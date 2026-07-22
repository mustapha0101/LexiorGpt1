# -*- coding: utf-8 -*-
"""FastAPI backend for the Lexior multi-agent legal assistant.

Endpoints:
    POST /api/chat           — SSE-streamed chat with the agent graph
    POST /api/dataset/generate — launch a dataset generation run
    GET  /api/dataset/runs   — list completed runs
    GET  /api/dataset/runs/{run_id}/rejections — rejection details for a run
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import dataclasses

from agentic_generation.config import load_config
from agentic_generation.schemas import (
    Message,
    Role,
    ScenarioSpec,
)
from agentic_generation.teacher_client import TeacherClient
from agentic_generation.tool_catalog import load_catalog
from lexior.agent_graph import (
    build_graph,
    initial_state,
)

# ── Resolve paths and build shared objects once at import time ───────────
_PHASE1 = Path(__file__).resolve().parents[2]
_REPO = _PHASE1.parent

_CFG = load_config(str(_PHASE1 / "configs" / "agentic_generation.yaml"))

_CATALOG_PATH = str(_REPO / "docs" / "mcp_tools_catalog.json")
if not Path(_CATALOG_PATH).exists():
    _CATALOG_PATH = str(_PHASE1 / "docs" / "mcp_tools_catalog.json")
_CATALOG = load_catalog(_CATALOG_PATH)

# Catalogue du chat : sans search_quebec_jurisprudence (serveur instable,
# renvoie des lois au lieu de décisions). La jurisprudence passe par
# search_legal_documents (a2aj).
from agentic_generation.tool_catalog import ToolCatalog  # noqa: E402

_CHAT_CATALOG = ToolCatalog(
    {
        **_CATALOG.raw,
        "tools": [
            t for t in _CATALOG.raw.get("tools", [])
            if t.get("canonicalName") != "search_quebec_jurisprudence"
        ],
    },
    path=_CATALOG_PATH,
)

_TEACHER = TeacherClient(_CFG.teacher, allow_remote_calls=True)
_CRITIC = (
    _TEACHER
    if _CFG.critic.model == "" or _CFG.critic.model == _CFG.teacher.model
    else TeacherClient(_CFG.critic, allow_remote_calls=True)
)

# Chat live : modèle plus fort que celui du pipeline dataset.
_CHAT_MODEL = os.environ.get("CHAT_TEACHER_MODEL", "gpt-4o")
_CHAT_TEACHER = TeacherClient(
    dataclasses.replace(_CFG.teacher, model=_CHAT_MODEL),
    allow_remote_calls=True,
)

# RAG local pour semantic_search_ccq/cpc (None tant que l'index n'est
# pas construit : ces deux outils renvoient alors une erreur d'outil,
# le planner bascule sur les autres).
_RAG = None
if _CFG.rag.enabled:
    try:
        from agentic_generation.legal_rag import (
            LegalRAG, OpenAIEmbedder, index_exists,
        )
        if index_exists(_CFG.rag.index_dir):
            _RAG = LegalRAG.load(
                _CFG.rag,
                OpenAIEmbedder(_CFG.rag, allow_remote_calls=True),
                reranker=_TEACHER,
            )
    except Exception as exc:
        print(f"[api] RAG indisponible: {type(exc).__name__}", flush=True)

app = FastAPI(title="Lexior API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DATA_ROOT = Path("data/runs")

_NODE_LABELS = {
    "plan": "Planning next step",
    "execute_tool": "Executing tool",
    "handle_clarification": "Handling clarification",
    "generate_answer": "Generating answer",
    "run_critics": "Evaluating quality",
    "repair": "Repairing answer",
    "validate_final": "Validating trajectory",
    "export": "Exporting result",
    "reject": "Processing rejection",
}


class ChatTurn(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    thread_id: Optional[str] = None
    mode: Optional[str] = "chat"
    jurisdiction: str = "quebec"
    history: list[ChatTurn] = []

    @property
    def text(self) -> str:
        return self.query or self.message or ""


class DatasetGenerateRequest(BaseModel):
    config_path: str = "configs/agentic_generation.yaml"
    count: int = 10


# ── SSE helper ───────────────────────────────────────────────────────────


def _sse(event_type: str, **fields) -> str:
    return json.dumps({"type": event_type, **fields}, ensure_ascii=False)


# ── Chat endpoint ────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat(request: ChatRequest):
    from sse_starlette.sse import EventSourceResponse

    async def _stream():
        scenario = ScenarioSpec(
            scenario_id=f"chat-{uuid.uuid4().hex[:8]}",
            scenario_family_id="chat",
            request_type="case_analysis",
            language="fr",
            user_query=request.text,
            jurisdiction=request.jurisdiction,
        )

        state = initial_state(
            scenario, mode="chat", max_tool_calls=4,
            system_prompt=(
                "Tu es Lexior, un assistant juridique canadien couvrant le "
                "droit québécois (CCQ, CPC, règlements) et le droit fédéral "
                "(lois fédérales, Code criminel). Réponds en français en "
                "citant les dispositions pertinentes. Quand la réponse "
                "dépend de la province et qu'elle est inconnue, demande où "
                "vit l'utilisateur avant de conclure."
            ),
        )

        # Contexte multi-tours : insérer l'historique avant le dernier
        # message utilisateur.
        hist_messages = []
        for turn in request.history:
            content = (turn.content or "").strip()
            if not content:
                continue
            role = Role.user if turn.role == "user" else Role.assistant
            hist_messages.append(Message(role=role, content=content))
        if hist_messages:
            msgs = state["messages"]
            state["messages"] = msgs[:-1] + hist_messages + msgs[-1:]

        yield _sse("thinking", content="Analyse de la question...")
        yield _sse("status", node="plan", label="Planning next step")

        try:
            from agentic_generation.planner_agent import PlannerAgent
            from agentic_generation.mcp_executor import MCPExecutor, RealMCPTransport
            from agentic_generation.trajectory_agent import TrajectoryAgent
            from agentic_generation.legal_critic import LegalCritic
            from agentic_generation.agentic_critic import AgenticCritic
            from agentic_generation.storage import JsonCache

            planner = PlannerAgent(_CHAT_CATALOG, client=_CHAT_TEACHER,
                                   chat_mode=True)
            transport = RealMCPTransport(_CFG.mcp_config_path)
            executor = MCPExecutor(
                _CHAT_CATALOG, transport=transport, allow_remote_calls=True,
                cache=JsonCache(Path(_CFG.data_root) / "cache" / "mcp-real"),
                max_response_chars=_CFG.max_tool_response_chars,
                rag=_RAG,
            )
            trajectory_agent = TrajectoryAgent(client=_CHAT_TEACHER,
                                               chat_mode=True)
            legal_critic = LegalCritic(client=_CRITIC)
            agentic_critic = AgenticCritic(client=_CRITIC)

            graph = build_graph(
                _CFG, _CHAT_CATALOG, planner, executor,
                trajectory_agent, legal_critic, agentic_critic,
            )

            q: queue.Queue = queue.Queue()

            def _run_graph():
                try:
                    for chunk in graph.stream(state):
                        q.put(("chunk", chunk))
                    q.put(("end", None))
                except Exception as exc:
                    q.put(("error", exc))

            thread = threading.Thread(target=_run_graph, daemon=True)
            thread.start()

            prev_tool_count = 0
            final_answer = ""
            final_thinking = ""
            rejection_reason = ""
            clarification_question = ""

            while True:
                try:
                    kind, data = await asyncio.to_thread(q.get, timeout=120)
                except Exception:
                    yield _sse("error", message="Timeout waiting for graph")
                    break

                if kind == "error":
                    yield _sse("error", message=str(data))
                    break

                if kind == "end":
                    break

                if not isinstance(data, dict):
                    continue

                for node_name, node_state in data.items():
                    if not isinstance(node_state, dict):
                        continue

                    label = _NODE_LABELS.get(node_name, node_name)
                    yield _sse("status", node=node_name, label=label)

                    tool_history = node_state.get("tool_history", [])
                    if len(tool_history) > prev_tool_count:
                        for obs in tool_history[prev_tool_count:]:
                            yield _sse(
                                "tool_call",
                                tool=obs.tool_name,
                                args=obs.arguments,
                            )
                            yield _sse(
                                "tool_result",
                                tool=obs.tool_name,
                                result=obs.normalized_response[:500],
                                ok=obs.ok,
                            )
                        prev_tool_count = len(tool_history)

                    if node_state.get("final_answer"):
                        final_answer = node_state["final_answer"]

                    if node_state.get("final_thinking"):
                        final_thinking = node_state["final_thinking"]

                    if node_state.get("status") == "clarification":
                        clarification_question = node_state.get(
                            "final_answer", "")

                    if (node_state.get("status") == "rejected"
                            and node_state.get("stop_reason")):
                        rejection_reason = node_state["stop_reason"]

            if clarification_question:
                yield _sse("clarification", question=clarification_question)
                yield _sse("done", accepted=True)
            elif final_answer:
                if final_thinking:
                    yield _sse("thinking", content=f"\n\n{final_thinking}")
                for i in range(0, len(final_answer), 20):
                    yield _sse("token", content=final_answer[i:i + 20])
                    await asyncio.sleep(0.01)
                yield _sse("done", accepted=True)
            elif rejection_reason:
                yield _sse("token",
                           content=f"Request could not be completed: "
                                   f"{rejection_reason}")
                yield _sse("done", accepted=False)
            else:
                yield _sse("done", accepted=True)

        except Exception as exc:
            yield _sse("error", message=str(exc))

    return EventSourceResponse(_stream())


# ── Dataset endpoints ────────────────────────────────────────────────────


@app.post("/api/dataset/generate")
async def dataset_generate(request: DatasetGenerateRequest):
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    return {"run_id": run_id, "status": "queued", "count": request.count}


@app.get("/api/dataset/runs")
async def dataset_runs():
    runs: list[dict[str, Any]] = []
    if not _DATA_ROOT.exists():
        return runs

    for manifest_file in sorted(
        (_DATA_ROOT / "manifests").glob("*.json"), reverse=True
    ):
        if manifest_file.name.endswith("_summary.json"):
            continue
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            counts = data.get("counts", {})
            total = counts.get("total", 0)
            accepted = counts.get("accepted", 0)
            runs.append({
                "run_id": data.get("run_id", manifest_file.stem),
                "total": total,
                "accepted": accepted,
                "rejected": counts.get("rejected", 0),
                "acceptance_rate": (
                    accepted / total if total > 0 else 0.0
                ),
                "created_at": data.get("created_at", ""),
                "teacher_model": data.get("teacher_model", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return runs


@app.get("/api/dataset/runs/{run_id}/rejections")
async def dataset_rejections(run_id: str):
    rejected_path = _DATA_ROOT / "rejected" / f"{run_id}.jsonl"
    if not rejected_path.exists():
        raise HTTPException(404, f"No rejections found for run {run_id}")

    rejections: list[dict[str, Any]] = []
    for line in rejected_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rejections.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return rejections


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
