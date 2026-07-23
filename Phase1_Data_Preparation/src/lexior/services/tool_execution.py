# -*- coding: utf-8 -*-
"""Service d'exécution d'outils MCP — un exécuteur, les deux modes.

Enveloppe ``agentic_generation.mcp_executor.MCPExecutor`` (transport mock
ou réel) et l'injection de panne simulée des scénarios dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agentic_generation.mcp_executor import (
    MCPExecutionError,
    MCPExecutor,
    MockMCPTransport,
    RealMCPTransport,
)
from agentic_generation.schemas import ToolCall, ToolObservation
from agentic_generation.tool_catalog import ToolCatalog


class ToolExecutionService:
    def __init__(self, executor: MCPExecutor):
        self.executor = executor

    @property
    def transport(self):
        return self.executor.transport

    def execute(self, call: ToolCall) -> ToolObservation:
        return self.executor.execute(call)

    # ── Injection de panne (scénarios dataset, transport mock) ───────────

    def inject_planned_failure(self, scenario, call: ToolCall,
                               tool_history: list[ToolObservation]) -> None:
        """Arme la panne planifiée du scénario sur le transport mock."""
        if not isinstance(self.executor.transport, MockMCPTransport):
            return
        failure_mode = getattr(scenario, "effective_failure_mode", None)
        if failure_mode == "tool_error" and not tool_history:
            self.executor.transport.fail_next = MCPExecutionError(
                "panne MCP simulée")
        elif (failure_mode == "empty_result"
              and call.name in {"semantic_search_ccq", "semantic_search_cpc"}
              and sum(1 for o in tool_history
                      if o.tool_name == call.name) < 2):
            self.executor.transport.empty_next = True


# ── Fabriques ────────────────────────────────────────────────────────────


def build_mock_executor(catalog: ToolCatalog, fixtures: dict,
                        max_retries: int = 0,
                        max_response_chars: int = 6000) -> MCPExecutor:
    return MCPExecutor(
        catalog, MockMCPTransport(dict(fixtures)),
        max_retries=max_retries, max_response_chars=max_response_chars,
    )


def build_real_executor(catalog: ToolCatalog, mcp_config_path: str,
                        *, cache=None, max_response_chars: int = 6000,
                        rag=None,
                        allow_remote_calls: bool = True) -> MCPExecutor:
    return MCPExecutor(
        catalog, RealMCPTransport(mcp_config_path),
        allow_remote_calls=allow_remote_calls, cache=cache,
        max_response_chars=max_response_chars, rag=rag,
    )
