# -*- coding: utf-8 -*-
"""Exécuteur MCP déterministe : validation, appel réel/mock, normalisation et cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .citations import COURT_SCOPE_PATTERN
from .response_verifier import strip_reader_directed
from .schemas import ToolCall, ToolObservation
from .storage import JsonCache, stable_hash
from .tool_catalog import CatalogError, ToolCatalog

URL_RE = re.compile(r"https?://[^\s<>\]\[\"')]+")
CITATION_RE = re.compile(
    rf"\b(?:\d{{4}}\s+(?:{COURT_SCOPE_PATTERN})\s+\d+|"
    r"RLRQ\s+c\s+[A-Za-z0-9.,\- ]+|[A-Z]-\d+(?:\.\d+)?(?:,\s*r\.\s*[^\n;]+)?)\b",
    re.IGNORECASE,
)
TEMP_PATH_RE = re.compile(r"(?:workspaceStorage|content\.txt|[A-Za-z]:\\[^\s]+)", re.IGNORECASE)

# SOQUIJ renvoie parfois « citoyens.soquij.qc.ca/ID=<hex> », sans le segment
# « /php/decision.php? ». L'identifiant est bon, le chemin manque : l'URL est
# irrésolvable en l'état et ferait échouer la validation. La forme correcte
# apparaît dans les mêmes runs, ce qui confirme la réécriture.
SOQUIJ_MALFORMED_RE = re.compile(
    r"(https?://citoyens\.soquij\.qc\.ca)/ID=([0-9A-Fa-f]{16,})")


def normalize_soquij_urls(text: str) -> str:
    """Rend résolvables les URLs SOQUIJ amputées de leur chemin."""
    return SOQUIJ_MALFORMED_RE.sub(r"\1/php/decision.php?ID=\2", text or "")
SECRET_RE = re.compile(r"(?i)(?:bearer\s+|api[_-]?key[=:]\s*)[^\s,;]+")
NORMALIZATION_VERSION = "mcp-normalize-1.2-strip-reader-directed"


class MCPExecutionError(Exception):
    pass


def safe_error_text(error: Exception) -> str:
    text = str(error)
    text = URL_RE.sub("[URL supprimée]", text)
    text = SECRET_RE.sub("[secret supprimé]", text)
    return text[:500]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _text_content(value: Any) -> str:
    value = _jsonable(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _text_content(value["content"])
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _text_content(item)))
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_mcp_response(raw: Any, max_chars: int) -> tuple[str, list[str], list[str], bool]:
    jsonable = _jsonable(raw)
    text = _text_content(jsonable).replace("\r\n", "\n").strip()
    # Les chemins temporaires ne sont jamais propagés au dataset.
    text = TEMP_PATH_RE.sub("[chemin local supprimé]", text)
    # Certains serveurs terminent leur réponse par des offres adressées au
    # lecteur (« Si vous souhaitez, je peux… ») : un modèle entraîné là-dessus
    # apprendrait qu'un outil de recherche lui pose des questions.
    text = strip_reader_directed(text)
    text = normalize_soquij_urls(text)
    structured_urls: list[str] = []
    structured_citations: list[str] = []
    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif isinstance(value, str):
            if "url" in key.casefold() and value.startswith(("http://", "https://")):
                structured_urls.append(normalize_soquij_urls(value))
            if "citation" in key.casefold() and value.strip():
                structured_citations.append(value.strip())
    collect(jsonable)
    urls = list(dict.fromkeys(structured_urls + URL_RE.findall(text)))
    citations = list(dict.fromkeys(structured_citations +
                                   [m.group(0).strip() for m in CITATION_RE.finditer(text)]))
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip() + "\n[TRONQUÉ: seule cette partie a été récupérée pour le contexte]"
    return text, urls, citations, truncated


def compact_legal_search_response(raw: Any) -> Any:
    """Retire snippets/licences volumineux tout en gardant un JSON valide."""
    try:
        payload = json.loads(_text_content(raw))
    except (TypeError, ValueError):
        return raw
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return raw
    kept = (
        "dataset", "citation_en", "citation_fr", "citation2_en", "citation2_fr",
        "name_en", "name_fr", "document_date_en", "document_date_fr",
        "source_url_en", "source_url_fr", "url_en", "url_fr", "score",
    )
    return {
        "results": [
            {key: result[key] for key in kept if key in result}
            for result in payload["results"]
            if isinstance(result, dict)
        ]
    }


class MockMCPTransport:
    """Transport explicitement réservé aux fixtures offline/dry-run."""

    def __init__(self, fixtures: Optional[dict[str, Any]] = None):
        self.fixtures = fixtures or {}
        self.calls: list[ToolCall] = []
        self.fail_next: Optional[Exception] = None
        self.empty_next: bool = False

    async def list_tools(self) -> dict[str, dict[str, Any]]:
        return {}

    async def call(self, server: str, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append(ToolCall(name=name, arguments=arguments))
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        if self.empty_next:
            self.empty_next = False
            return ""
        value = self.fixtures.get(name)
        if callable(value):
            value = value(arguments)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise MCPExecutionError(f"fixture MCP absente pour {name}")
        return value


class RealMCPTransport:
    """Client MCP standard pour serveurs SSE et Streamable HTTP."""

    def __init__(self, mcp_config_path: str, timeout: float = 45.0):
        self.path = Path(mcp_config_path)
        self.timeout = timeout
        if not self.path.exists():
            raise MCPExecutionError(f"configuration MCP introuvable : {self.path}")
        self.config = json.loads(self.path.read_text(encoding="utf-8"))

    def _server_config(self, observed_name: str, catalog_server: str) -> tuple[str, dict[str, Any]]:
        servers = self.config.get("mcpServers", {})
        aliases = [observed_name, catalog_server]
        lowered = catalog_server.lower()
        if "a2aj" in lowered or "canadian legal" in lowered:
            aliases.append("a2aj")
        if "quebec" in lowered or "legisquebec" in lowered:
            aliases.append("lexior-ccq")
        for alias in aliases:
            if alias and alias in servers:
                return alias, servers[alias]
        raise MCPExecutionError(f"serveur MCP non configuré pour {catalog_server}")

    async def _session(self, stack: AsyncExitStack, cfg: dict[str, Any]):
        from mcp import ClientSession
        server_type = str(cfg.get("type", "sse")).lower().replace("_", "-")
        url = cfg.get("url", "")
        if not url:
            raise MCPExecutionError("URL MCP manquante")
        if server_type == "sse":
            from mcp.client.sse import sse_client
            read, write = await stack.enter_async_context(sse_client(url=url, timeout=self.timeout))
        elif server_type in {"streamable-http", "streamablehttp", "http"}:
            from mcp.client.streamable_http import streamablehttp_client
            streams = await stack.enter_async_context(
                streamablehttp_client(url=url, timeout=self.timeout))
            read, write = streams[0], streams[1]
        else:
            raise MCPExecutionError(
                f"transport MCP non pris en charge : {server_type} (sse|streamable-http)")
        session = await stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=self.timeout)
        return session

    async def list_tools_for(self, observed_name: str, catalog_server: str) -> dict[str, dict[str, Any]]:
        _, cfg = self._server_config(observed_name, catalog_server)
        async with AsyncExitStack() as stack:
            session = await self._session(stack, cfg)
            result = await asyncio.wait_for(session.list_tools(), timeout=self.timeout)
            return {tool.name: _jsonable(tool.inputSchema) for tool in result.tools}

    async def call(self, observed_name: str, catalog_server: str, name: str, arguments: dict[str, Any]) -> Any:
        _, cfg = self._server_config(observed_name, catalog_server)
        async with AsyncExitStack() as stack:
            session = await self._session(stack, cfg)
            return await asyncio.wait_for(session.call_tool(name, arguments), timeout=self.timeout)


class MCPExecutor:
    def __init__(self, catalog: ToolCatalog, transport: MockMCPTransport | RealMCPTransport,
                 allow_remote_calls: bool = False, cache: Optional[JsonCache] = None,
                 max_response_chars: int = 6000, timeout: float = 45.0,
                 max_retries: int = 2, rag: Any = None):
        self.catalog = catalog
        self.transport = transport
        self.allow_remote_calls = allow_remote_calls
        self.cache = cache
        self.max_response_chars = max_response_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self.is_mock = isinstance(transport, MockMCPTransport)
        self.rag = rag

    async def verify_catalog(self) -> list[str]:
        if self.is_mock:
            return []
        if not self.allow_remote_calls:
            raise MCPExecutionError("connexion MCP refusée sans --allow-remote-calls")
        live: dict[str, dict[str, Any]] = {}
        for catalog_name, server in self.catalog.servers.items():
            if str(server.get("type", "")).casefold() == "local":
                continue
            observed = server.get("observedAs", "")
            live.update(await self.transport.list_tools_for(observed, catalog_name))
        problems = self.catalog.compare_with_live(live)
        if problems:
            raise CatalogError("catalogue MCP divergent : " + "; ".join(problems))
        return problems

    def _key(self, call: ToolCall) -> str:
        rag_hash = ""
        if self.catalog.is_local(call.name) and self.rag is not None:
            rag_hash = str(getattr(self.rag, "cache_signature", ""))
        return "mcp:" + stable_hash({
            "catalog": self.catalog.catalog_hash,
            "normalization": NORMALIZATION_VERSION,
            "rag_corpus": rag_hash,
            "name": call.name,
            "arguments": call.arguments,
        })

    async def aexecute(self, call: ToolCall) -> ToolObservation:
        errors = self.catalog.validate_call(call.name, call.arguments)
        if errors:
            raise MCPExecutionError("; ".join(errors))
        if not self.is_mock and not self.allow_remote_calls:
            raise MCPExecutionError("appel MCP distant refusé sans --allow-remote-calls")
        spec = self.catalog.tools[call.name]
        cache_key = self._key(call)
        cached = self.cache.get(cache_key) if self.cache else None
        if cached is not None:
            obs = ToolObservation.model_validate(cached)
            # Une fixture ne doit jamais contaminer un run réel via le cache.
            if obs.mock == self.is_mock:
                return obs

        started = time.monotonic()
        raw: Any = None
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.is_mock:
                    raw = await asyncio.wait_for(
                        self.transport.call(spec.server, call.name, call.arguments), self.timeout)
                elif self.catalog.is_local(call.name):
                    if self.rag is None:
                        raise MCPExecutionError(
                            f"index RAG non initialisé pour {call.name}")
                    raw = self.rag.call(call.name, call.arguments)
                else:
                    server_entry = self.catalog.servers.get(spec.server, {})
                    raw = await self.transport.call(server_entry.get("observedAs", ""), spec.server,
                                                    call.name, call.arguments)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 5))
        latency = (time.monotonic() - started) * 1000
        if last_error is not None:
            safe_message = safe_error_text(last_error)
            obs = ToolObservation(tool_name=call.name, server=spec.server,
                                  arguments=call.arguments, raw_response=None,
                                  normalized_response=json.dumps({"error": type(last_error).__name__,
                                                                  "message": safe_message}, ensure_ascii=False),
                                  error=f"{type(last_error).__name__}: {safe_message}", ok=False,
                                  mock=self.is_mock, latency_ms=latency).finalize_hash()
        else:
            normalized_raw = (
                compact_legal_search_response(raw)
                if call.name == "search_legal_documents" else raw
            )
            normalized, urls, citations, truncated = normalize_mcp_response(
                normalized_raw, self.max_response_chars)
            obs = ToolObservation(tool_name=call.name, server=spec.server,
                                  arguments=call.arguments, raw_response=_jsonable(raw),
                                  normalized_response=normalized, source_urls=urls,
                                  citations=citations, truncated=truncated, ok=True,
                                  mock=self.is_mock, latency_ms=latency).finalize_hash()
        if self.cache:
            self.cache.put(cache_key, obs.model_dump(mode="json"))
        return obs

    def execute(self, call: ToolCall) -> ToolObservation:
        return asyncio.run(self.aexecute(call))
