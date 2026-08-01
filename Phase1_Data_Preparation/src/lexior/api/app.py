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
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import dataclasses

from lexior.agentic.config import load_config
from lexior.agentic.storage import JsonCache
from lexior.agentic.teacher_client import TeacherClient
from lexior.agentic.tool_catalog import ToolCatalog, load_catalog
from lexior.agent_graph import GraphRunner, build_context
from lexior.agent_graph.checkpointing import create_memory_checkpointer
from lexior.services import build_real_executor, build_services
from lexior.services.tool_coverage import get_coverage
from lexior.evaluation.human_40_recorder import Human40Recorder

# ── Resolve paths and build shared objects once at import time ───────────
# src/lexior/api/app.py -> api -> lexior -> src -> Phase1_Data_Preparation.
# parents[2] désignait `src` depuis la migration vers src/ : le catalogue
# était introuvable et load_config retombait en silence sur ses défauts.
_PHASE1 = Path(__file__).resolve().parents[3]
_REPO = _PHASE1.parent

_CFG = load_config(str(_PHASE1 / "configs" / "agentic_generation.yaml"))

_CATALOG_PATH = str(_REPO / "docs" / "mcp_tools_catalog.json")
if not Path(_CATALOG_PATH).exists():
    _CATALOG_PATH = str(_PHASE1 / "docs" / "mcp_tools_catalog.json")
_CATALOG = load_catalog(_CATALOG_PATH)


def _available_in_chat(tool_name: str) -> bool:
    """Un outil est-il exposé au chat ? ``tool_coverage`` fait foi.

    Un outil sans entrée de couverture est conservé : l'absence d'avis
    n'est pas un refus.
    """
    entry = get_coverage(tool_name)
    return entry is None or entry.is_available("live")

# Catalogue du chat : dérivé de la couverture déclarée, jamais d'une liste
# d'exclusion écrite ici. Une exclusion codée en dur avait survécu à la
# réactivation de search_quebec_jurisprudence, qui restait donc
# inaccessible en production alors que tool_coverage.py l'annonçait
# disponible — deux sources de vérité, une seule mise à jour.
_CHAT_CATALOG = ToolCatalog(
    {
        **_CATALOG.raw,
        "tools": [
            tool for tool in _CATALOG.raw.get("tools", [])
            if _available_in_chat(tool.get("canonicalName", ""))
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


def _effective_chat_model(model_id: Optional[str]) -> str:
    if model_id in ("gpt-4o", "gpt-4o-mini", "qwen-local"):
        return model_id
    return _DEFAULT_CHAT_MODEL


def _provider_model(model_id: str) -> str:
    return _LOCAL_MODEL_NAME if model_id == "qwen-local" else model_id

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
        from lexior.agentic.legal_rag import (
            LegalRAG, OpenAIEmbedder, index_exists,
        )
        if index_exists(_CFG.rag.index_dir):
            _RAG = LegalRAG.load(
                _CFG.rag,
                OpenAIEmbedder(_CFG.rag, allow_remote_calls=True),
                reranker=_CRITIC,
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
    key = _effective_chat_model(model_id)
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
_HUMAN_40 = Human40Recorder(
    _PHASE1 / "data" / "evaluations" / "human_40", _PHASE1)


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
    evaluation_run_id: Optional[str] = None
    evaluation_scenario_id: Optional[int] = None

    @property
    def text(self) -> str:
        return self.query or self.message or ""


class DatasetGenerateRequest(BaseModel):
    config_path: str = "configs/agentic_generation.yaml"
    count: int = 10


class HumanRunCreateRequest(BaseModel):
    run_id: Optional[str] = None


class HumanScenarioStartRequest(BaseModel):
    thread_id: Optional[str] = None


class HumanReviewPatch(BaseModel):
    expected_answer_note: Optional[str] = None
    overall_rating: Optional[str] = None
    route_quality: Optional[str] = None
    response_quality: Optional[list[str]] = None
    notes: Optional[str] = None
    observed_category: Optional[str] = None


class HumanEventRequest(BaseModel):
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class HumanRunPatch(BaseModel):
    action: str
    scenario_id: Optional[int] = None


# ── SSE helpers ──────────────────────────────────────────────────────────


def _sse(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)


def _evaluation_scenario(run_id: str, scenario_id: int) -> dict[str, Any]:
    try:
        run = _HUMAN_40.load_run(run_id)
        scenario = next(item for item in run["scenarios"]
                        if item["scenario_id"] == scenario_id)
    except (ValueError, FileNotFoundError, StopIteration) as exc:
        raise HTTPException(status_code=404, detail="evaluation scenario not found") from exc
    if scenario["status"] not in {"in_progress", "interrupted"}:
        raise HTTPException(status_code=409, detail="evaluation scenario is not active")
    return scenario


def _evaluation_context(request: ChatRequest) -> tuple[str, int, dict[str, Any]] | None:
    if request.mode != "human_40":
        return None
    if not request.evaluation_run_id or request.evaluation_scenario_id is None:
        raise HTTPException(
            status_code=422,
            detail="human_40 chat requires evaluation_run_id and evaluation_scenario_id",
        )
    if not request.thread_id:
        raise HTTPException(status_code=422, detail="human_40 chat requires thread_id")
    scenario = _evaluation_scenario(
        request.evaluation_run_id, request.evaluation_scenario_id)
    if scenario.get("thread_id") != request.thread_id:
        raise HTTPException(status_code=409, detail="thread does not belong to this scenario")
    return request.evaluation_run_id, request.evaluation_scenario_id, scenario


# ── Chat endpoint ────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat(request: ChatRequest):
    from sse_starlette.sse import EventSourceResponse

    evaluation = _evaluation_context(request)
    runner = _runner_for(request.model)
    thread_id = request.thread_id or f"live-{uuid.uuid4().hex[:8]}"
    history = [
        {"role": turn.role, "content": turn.content}
        for turn in request.history
    ]
    # Un 7B local peut mettre plusieurs minutes par décision.
    selected_model = _effective_chat_model(request.model)
    queue_timeout = 900 if selected_model == "qwen-local" else 120
    recording_error: str | None = None
    if evaluation:
        run_id, scenario_id, scenario_snapshot = evaluation
        previous = scenario_snapshot.get("conversation", [])
        previous_clarification = bool(
            previous and previous[-1].get("role") == "assistant"
            and previous[-1].get("message_type") == "clarification")
        try:
            _HUMAN_40.append_message(
                run_id, scenario_id=scenario_id, role="user",
                content=request.text,
                message_type=("clarification_answer" if previous_clarification
                              else "initial_query"),
                turn_index=len(previous) + 1,
            )
        except Exception as exc:
            recording_error = f"{type(exc).__name__}: {exc}"

    async def _stream():
        nonlocal recording_error
        # Public metadata for the raw log. This identifies the selected UI
        # model and the concrete provider model without exposing prompt data.
        request_started = {
            "type": "request_started",
            "model": selected_model,
            "provider_model": _provider_model(selected_model),
            "thread_id": thread_id,
        }
        if evaluation:
            try:
                _HUMAN_40.append_backend_event(
                    run_id, scenario_id=scenario_id, event=request_started)
            except Exception as exc:
                recording_error = f"{type(exc).__name__}: {exc}"
        yield _sse(request_started)
        yield _sse({"type": "thinking",
                    "content": "Analyse de la question..."})

        events: queue.Queue = queue.Queue()
        assistant_tokens: list[str] = []
        pending_clarification = False

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
                if evaluation:
                    try:
                        _HUMAN_40.append_backend_event(
                            run_id, scenario_id=scenario_id,
                            event={"type": "error", "message": str(data)},
                        )
                        _HUMAN_40.append_message(
                            run_id, scenario_id=scenario_id, role="assistant",
                            content=str(data), message_type="technical_error",
                        )
                    except Exception as exc:
                        recording_error = f"{type(exc).__name__}: {exc}"
                yield _sse({"type": "error", "message": str(data)})
                break
            if kind == "end":
                break

            if evaluation:
                try:
                    event_type = str(data.get("type", ""))
                    if event_type == "token":
                        assistant_tokens.append(str(data.get("content", "")))
                    elif event_type == "clarification":
                        pending_clarification = True
                        _HUMAN_40.append_message(
                            run_id, scenario_id=scenario_id, role="assistant",
                            content=str(data.get("question", "")),
                            message_type="clarification", accepted=True,
                        )
                    elif event_type == "done":
                        if assistant_tokens and not pending_clarification:
                            _HUMAN_40.append_final_answer(
                                run_id, scenario_id=scenario_id,
                                content="".join(assistant_tokens),
                                accepted=data.get("accepted"),
                            )
                        _HUMAN_40.append_backend_event(
                            run_id, scenario_id=scenario_id,
                            event={"type": "return_live_answer", **data},
                        )
                    else:
                        _HUMAN_40.append_backend_event(
                            run_id, scenario_id=scenario_id, event=data)
                except Exception as exc:
                    recording_error = f"{type(exc).__name__}: {exc}"
            yield _sse(data)
            if data.get("type") == "token":
                await asyncio.sleep(0.01)

        if recording_error:
            yield _sse({"type": "evaluation_save_error",
                        "message": "Evaluation log could not be saved completely."})

    return EventSourceResponse(_stream())


# ── Dataset endpoints ────────────────────────────────────────────────────


@app.post("/api/evaluations/human-40/runs")
async def human_40_create_run(request: HumanRunCreateRequest):
    try:
        return _HUMAN_40.create_run(request.run_id)
    except (ValueError, FileExistsError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/evaluations/human-40/runs/{run_id}")
async def human_40_get_run(run_id: str):
    try:
        return _HUMAN_40.load_run(run_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="evaluation run not found") from exc


@app.get("/api/evaluations/human-40/runs/{run_id}/file")
async def human_40_file(run_id: str):
    try:
        path = _HUMAN_40._path(run_id)
        if not path.exists():
            raise FileNotFoundError
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="evaluation run not found") from exc
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.post("/api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}/start")
async def human_40_start_scenario(run_id: str, scenario_id: int,
                                  request: HumanScenarioStartRequest):
    try:
        return _HUMAN_40.start_scenario(run_id, scenario_id, request.thread_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, FileExistsError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.patch("/api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}")
async def human_40_update_scenario(run_id: str, scenario_id: int,
                                   request: HumanReviewPatch):
    try:
        return _HUMAN_40.update_human_review(
            run_id, scenario_id, **request.model_dump(exclude_none=True))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}/events")
async def human_40_event(run_id: str, scenario_id: int,
                         request: HumanEventRequest):
    try:
        return _HUMAN_40.append_manual_event(
            run_id, scenario_id, request.event_type, request.payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch("/api/evaluations/human-40/runs/{run_id}")
async def human_40_patch_run(run_id: str, request: HumanRunPatch):
    if request.action != "interrupt_scenario" or request.scenario_id is None:
        raise HTTPException(status_code=422, detail="only interrupt_scenario is supported")
    try:
        return _HUMAN_40.interrupt_scenario(run_id, request.scenario_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}/complete")
async def human_40_complete_scenario(run_id: str, scenario_id: int):
    try:
        return _HUMAN_40.complete_scenario(run_id, scenario_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/evaluations/human-40/runs/{run_id}/complete")
async def human_40_complete_run(run_id: str):
    try:
        return _HUMAN_40.complete_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
