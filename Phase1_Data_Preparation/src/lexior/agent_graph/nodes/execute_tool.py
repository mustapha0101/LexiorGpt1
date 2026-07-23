# -*- coding: utf-8 -*-
"""execute_tool — exécution MCP brute. La vérification suit, séparée."""

from __future__ import annotations

from typing import Any

from agentic_generation.schemas import Message, PlannerDecision, Role, ToolCall

from ..context import GraphContext
from ..state import LexiorState

NAME = "execute_tool"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    decision = PlannerDecision.model_validate(state["latest_decision"])
    call = ToolCall(name=decision.next_tool, arguments=decision.arguments)

    tool_history = state.get("tool_history", [])
    ctx.services.tools.inject_planned_failure(
        state["scenario"], call, tool_history)

    observation = ctx.services.tools.execute(call)

    messages = list(state.get("messages", []))
    messages.append(Message(
        role=Role.assistant,
        thinking=decision.thinking_text or None,
        content=f"<tool_call>\n{call.render()}\n</tool_call>",
    ))

    return {
        "messages": messages,
        "tool_history": list(tool_history) + [observation],
        "last_tool_call": {"name": call.name, "arguments": call.arguments},
        "status": "planning",
    }
