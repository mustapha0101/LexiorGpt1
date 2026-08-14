# -*- coding: utf-8 -*-
"""API minimale de la démonstration live Lexior.

Le backend expose uniquement l'état opérationnel et le chat SSE. Toute la
logique juridique reste dans le graphe central ``lexior.agent_graph``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lexior.agent_graph import GraphRunner, build_context
from lexior.agent_graph.checkpointing import create_memory_checkpointer
from lexior.agentic.config import load_config
from lexior.agentic.storage import JsonCache
from lexior.agentic.teacher_client import TeacherClient
from lexior.agentic.tool_catalog import ToolCatalog, load_catalog
from lexior.services import build_real_executor, build_services
from lexior.services.tool_coverage import get_coverage


_PHASE1 = Path(__file__).resolve().parents[3]
_REPO = _PHASE1.parent
_CFG = load_config(str(_PHASE1 / "configs" / "agentic_generation.yaml"))

_CATALOG_PATH = _REPO / "docs" / "mcp_tools_catalog.json"
if not _CATALOG_PATH.exists():
    _CATALOG_PATH = _PHASE1 / "docs" / "mcp_tools_catalog.json"
_CATALOG = load_catalog(str(_CATALOG_PATH))


def _available_in_chat(tool_name: str) -> bool:
    entry = get_coverage(tool_name)
    return entry is None or entry.is_available("live")


_CHAT_CATALOG = ToolCatalog(
    {
        **_CATALOG.raw,
        "tools": [
            tool for tool in _CATALOG.raw.get("tools", [])
            if _available_in_chat(tool.get("canonicalName", ""))
        ],
    },
    path=str(_CATALOG_PATH),
)

_TEACHER = TeacherClient(_CFG.teacher, allow_remote_calls=True)
_CRITIC = (
    _TEACHER
    if _CFG.critic.model == "" or _CFG.critic.model == _CFG.teacher.model
    else TeacherClient(_CFG.critic, allow_remote_calls=True)
)

_DEFAULT_CHAT_MODEL = os.environ.get("CHAT_TEACHER_MODEL", "gpt-4o")

# ``AgenticConfig.critic_is_teacher`` compare les valeurs du YAML
# (gpt-4o vs gpt-4o-mini) et ne se déclenche donc jamais en live, alors que
# le rédacteur EFFECTIF est ``CHAT_TEACHER_MODEL`` — gpt-4o par défaut,
# c'est-à-dire le modèle du critique. L'avertissement doit porter sur ce
# qui tourne réellement, pas sur ce qui est écrit dans le fichier.
if (not _CFG.no_critics
        and (_CFG.critic.model or _CFG.teacher.model) == _DEFAULT_CHAT_MODEL):
    print(
        f"[api] critique == rédacteur ({_DEFAULT_CHAT_MODEL}) : le même "
        f"modèle rédige la réponse ET la juge; le taux d'acceptation est "
        f"mécaniquement optimiste. Définir CHAT_TEACHER_MODEL sur un autre "
        f"modèle pour séparer les deux rôles.",
        flush=True)
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
    "vit l'utilisateur avant de conclure. En droit du travail, "
    "distingue le lieu de travail du régime fédéral ou provincial; "
    "si le secteur de l'employeur est inconnu, demande-le dans la "
    "même clarification que le lieu. Les autres faits d'application "
    "ne bloquent pas la recherche : réponds conditionnellement et "
    "pose les questions décisives à la fin."
)

_RAG = None
if _CFG.rag.enabled:
    try:
        from lexior.agentic.legal_rag import LegalRAG, OpenAIEmbedder, index_exists

        if index_exists(_CFG.rag.index_dir):
            _RAG = LegalRAG.load(
                _CFG.rag,
                OpenAIEmbedder(_CFG.rag, allow_remote_calls=True),
                reranker=_CRITIC,
            )
    except Exception as exc:  # noqa: BLE001 - état affiché au démarrage
        print(
            f"[api] RAG indisponible: {type(exc).__name__}: {str(exc)[:240]}",
            flush=True,
        )

if _RAG is None:
    print(
        "[api] RAG absent: fallback MCP lexical activé pour les recherches sémantiques",
        flush=True,
    )

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
                _CHAT_CATALOG,
                _CFG.mcp_config_path,
                cache=JsonCache(Path(_CFG.data_root) / "cache" / "mcp-real"),
                max_response_chars=_CFG.max_tool_response_chars,
                rag=_RAG,
            )
            services = build_services(
                _CFG,
                _CHAT_CATALOG,
                executor=executor,
                teacher=_chat_client(key),
                critic_client=_CRITIC,
            )
            _RUNNERS[key] = GraphRunner(
                build_context(_CFG, _CHAT_CATALOG, services),
                checkpointer=_CHECKPOINTER,
            )
        return _RUNNERS[key]


app = FastAPI(title="Lexior Live Demo API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "api_version": app.version,
        "rag": {
            "enabled": bool(_CFG.rag.enabled),
            "loaded": _RAG is not None,
            "index_dir": str(_CFG.rag.index_dir),
            "embedding_model": str(_CFG.rag.embedding_model),
            "fallback": "none" if _RAG is not None else "mcp_keyword_union",
        },
        "chat_tools": len(_CHAT_CATALOG.tools),
    }


class ChatTurn(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    query: Optional[str] = None
    message: Optional[str] = None
    thread_id: Optional[str] = None
    history: list[ChatTurn] = Field(default_factory=list)
    model: Optional[str] = None

    @property
    def text(self) -> str:
        return self.query or self.message or ""


def _sse(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    from sse_starlette.sse import EventSourceResponse

    runner = _runner_for(request.model)
    thread_id = request.thread_id or f"live-{uuid.uuid4().hex[:8]}"
    history = [
        {"role": turn.role, "content": turn.content}
        for turn in request.history
    ]
    selected_model = _effective_chat_model(request.model)
    # Délai INTER-ÉVÉNEMENT, pas durée totale du tour : le graphe émet un
    # « status » à chaque nœud. Une valeur trop basse coupait un tour vivant
    # dont un seul appel d'outil était simplement lent.
    queue_timeout = 900 if selected_model == "qwen-local" else 180

    async def _stream():
        yield _sse({
            "type": "request_started",
            "model": selected_model,
            "provider_model": _provider_model(selected_model),
            "critic_model": _CFG.critic.model or _CFG.teacher.model,
            "thread_id": thread_id,
        })
        yield _sse({"type": "thinking", "content": "Analyse de la question..."})

        events: queue.Queue = queue.Queue()
        # Posé quand le consommateur SSE s'en va. Le graphe ne peut pas être
        # interrompu au milieu d'un appel, mais il cesse de produire dès le
        # suivant : sans cela, un onglet fermé laissait un tour entier
        # consommer des crédits pour personne.
        abandoned = threading.Event()

        def _run_graph():
            try:
                for event in runner.stream_live(
                    request.text,
                    thread_id=thread_id,
                    history=history,
                    system_prompt=_SYSTEM_PROMPT,
                ):
                    if abandoned.is_set():
                        events.put(("abandoned", None))
                        return
                    events.put(("event", event))
                events.put(("end", None))
            except Exception as exc:  # noqa: BLE001 - transmis au client
                events.put(("error", exc))

        threading.Thread(
            target=_run_graph, daemon=True,
            name=f"lexior-graph-{thread_id}").start()

        try:
            while True:
                try:
                    kind, data = await asyncio.to_thread(
                        events.get, timeout=queue_timeout)
                except Exception:
                    abandoned.set()
                    print(f"[api] {thread_id}: aucun événement depuis "
                          f"{queue_timeout}s, tour abandonné", flush=True)
                    yield _sse({
                        "type": "error",
                        "message": ("Le traitement a dépassé le temps imparti. "
                                    "Reformulez la question ou réessayez.")})
                    yield _sse({"type": "done", "accepted": False,
                                "stop_reason": "timeout",
                                "thread_id": thread_id})
                    break

                if kind == "error":
                    # La trace interne va au journal serveur, jamais au
                    # navigateur : elle expose chemins et détails d'exécution.
                    print(f"[api] {thread_id}: {type(data).__name__}: {data}",
                          flush=True)
                    yield _sse({
                        "type": "error",
                        "message": ("Une erreur interne a interrompu le "
                                    "traitement de la question.")})
                    yield _sse({"type": "done", "accepted": False,
                                "stop_reason": "internal_error",
                                "thread_id": thread_id})
                    break
                if kind in ("end", "abandoned"):
                    break

                yield _sse(data)
                if data.get("type") == "token":
                    await asyncio.sleep(0.01)
        finally:
            # Déconnexion du client (fermeture d'onglet, navigation) :
            # EventSourceResponse ferme le générateur, on passe ici.
            abandoned.set()

    return EventSourceResponse(_stream())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
