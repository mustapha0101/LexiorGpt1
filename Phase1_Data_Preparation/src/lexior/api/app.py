# -*- coding: utf-8 -*-
"""FastAPI backend for the Lexior multi-agent legal assistant.

Endpoints:
    POST /api/chat           — SSE-streamed chat with the CENTRAL agent graph
    POST /api/dataset/generate — launch a dataset generation run
    GET  /api/dataset/runs   — list completed runs
    GET  /api/dataset/runs/{run_id}/rejections — rejection details for a run

Le backend ne contient AUCUNE logique d'orchestration : chaque tour de
chat est un run du graphe central (``lexior.agent_graph.GraphRunner``),
qui produit lui-même les événements streamés. Les clarifications live
utilisent ``interrupt()`` : le thread LangGraph reste suspendu et le
message suivant du même ``thread_id`` reprend l'exécution.
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
from agentic_generation.storage import JsonCache
from agentic_generation.teacher_client import TeacherClient
from agentic_generation.tool_catalog import ToolCatalog, load_catalog
from lexior.agent_graph import GraphRunner, build_context
from lexior.agent_graph.checkpointing import create_memory_checkpointer
from lexior.services import build_real_executor, build_services

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

# Chat live : modèle choisi par requête via le menu de l'UI.
#   gpt-4o / gpt-4o-mini : API OpenAI (endpoint teacher, modèle remplacé)
#   qwen-local           : serveur local compatible OpenAI (Ollama)
_DEFAULT_CHAT_MODEL = os.environ.get("CHAT_TEACHER_MODEL", "gpt-4o")
_LOCAL_BASE_URL = os.environ.get(
    "LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
_LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "qwen2.5:7b-16k")

_SYSTEM_PROMPT = (
    "Tu es Lexior, un assistant juridique canadien couvrant le "
    "droit québécois (CCQ, CPC, règlements) et le droit fédéral "
    "(lois fédérales, Code criminel). Réponds en français en "
    "citant les dispositions pertinentes. Quand la réponse "
    "dépend de la province et qu'elle est inconnue, demande où "
    "vit l'utilisateur avant de conclure."
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

# ── Un runner par modèle de chat — MÊME graphe, MÊMES services ───────────
# Le checkpointer est PARTAGÉ : un thread de conversation garde ses
# clarifications en attente même si l'utilisateur change de modèle.
_CHECKPOINTER = create_memory_checkpointer()
_RUNNERS: dict[str, GraphRunner] = {}
_RUNNERS_LOCK = threading.Lock()


def _chat_client(model_id: str) -> TeacherClient:
    if model_id == "qwen-local":
        endpoint = dataclasses.replace(
            _CFG.teacher,
            base_url=_LOCAL_BASE_URL,
            api_key="not-needed",
            model=_LOCAL_MODEL_NAME,
            timeout=600.0,
        )
    else:
        endpoint = dataclasses.replace(_CFG.teacher, model=model_id)
    return TeacherClient(endpoint, allow_remote_calls=True)


def _runner_for(model_id: Optional[str]) -> GraphRunner:
    key = (model_id
           if model_id in ("gpt-4o", "gpt-4o-mini", "qwen-local")
           else _DEFAULT_CHAT_MODEL)
    with _RUNNERS_LOCK:
        if key not in _RUNNERS:
            executor = build_real_executor(
                _CHAT_CATALOG, _CFG.mcp_config_path,
                cache=JsonCache(Path(_CFG.data_root) / "cache" / "mcp-real"),
                max_response_chars=_CFG.max_tool_response_chars,
                rag=_RAG,
            )
            services = build_services(
                _CFG, _CHAT_CATALOG,
                executor=executor,
                teacher=_chat_client(key),
                critic_client=_CRITIC,
            )
            _RUNNERS[key] = GraphRunner(
                build_context(_CFG, _CHAT_CATALOG, services),
                checkpointer=_CHECKPOINTER,
            )
        return _RUNNERS[key]


app = FastAPI(title="Lexior API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DATA_ROOT = Path("data/runs")


class ChatTurn(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    thread_id: Optional[str] = None
    mode: Optional[str] = "live"
    jurisdiction: str = ""
    history: list[ChatTurn] = []
    model: Optional[str] = None  # "gpt-4o" | "gpt-4o-mini" | "qwen-local"

    @property
    def text(self) -> str:
        return self.query or self.message or ""


class DatasetGenerateRequest(BaseModel):
    config_path: str = "configs/agentic_generation.yaml"
    count: int = 10


# ── SSE helpers ──────────────────────────────────────────────────────────


def _sse(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)


# ── Chat endpoint ────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat(request: ChatRequest):
    from sse_starlette.sse import EventSourceResponse

    runner = _runner_for(request.model)
    thread_id = request.thread_id or f"live-{uuid.uuid4().hex[:8]}"
    history = [
        {"role": turn.role, "content": turn.content}
        for turn in request.history
    ]
    # Un 7B local peut mettre plusieurs minutes par décision.
    queue_timeout = 900 if request.model == "qwen-local" else 120

    async def _stream():
        yield _sse({"type": "thinking",
                    "content": "Analyse de la question..."})

        events: queue.Queue = queue.Queue()

        def _run_graph():
            try:
                for event in runner.stream_live(
                        request.text,
                        thread_id=thread_id,
                        history=history,
                        system_prompt=_SYSTEM_PROMPT):
                    events.put(("event", event))
                events.put(("end", None))
            except Exception as exc:  # noqa: BLE001 — remonté au client
                events.put(("error", exc))

        thread = threading.Thread(target=_run_graph, daemon=True)
        thread.start()

        while True:
            try:
                kind, data = await asyncio.to_thread(
                    events.get, timeout=queue_timeout)
            except Exception:
                yield _sse({"type": "error",
                            "message": "Timeout waiting for graph"})
                break

            if kind == "error":
                yield _sse({"type": "error", "message": str(data)})
                break
            if kind == "end":
                break

            yield _sse(data)
            if data.get("type") == "token":
                await asyncio.sleep(0.01)

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
