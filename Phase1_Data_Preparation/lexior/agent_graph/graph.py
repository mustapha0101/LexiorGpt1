# -*- coding: utf-8 -*-
"""LangGraph StateGraph builder for the Lexior agent.

``build_graph`` assembles nodes and conditional edges into a compiled
graph.  Two modes share the same topology; mode-dependent behaviour
lives inside nodes (e.g. synthetic vs real clarification).
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph

from agentic_generation.agentic_critic import AgenticCritic
from agentic_generation.config import AgenticConfig
from agentic_generation.legal_critic import LegalCritic
from agentic_generation.mcp_executor import MCPExecutor
from agentic_generation.planner_agent import PlannerAgent
from agentic_generation.storage import RunStorage
from agentic_generation.tool_catalog import ToolCatalog
from agentic_generation.trajectory_agent import PRECISE_ARTICLE_TYPES, TrajectoryAgent

from .nodes import GraphNodes
from .state import LexiorState


# ── Conditional edge routing functions ───────────────────────────────────


def _route_after_plan(state: LexiorState) -> str:
    decision = state.get("current_decision") or {}
    decision_type = decision.get("decision", "")
    step = state.get("step", 0)
    max_steps = state.get("max_tool_calls", 4) + 2

    if state.get("status") == "rejected":
        return "reject"

    if step > max_steps:
        return "reject"

    if decision_type == "call_tool":
        return "execute_tool"

    if decision_type == "ask_clarification":
        if state.get("clarification_count", 0) >= 1:
            return "reject"
        return "handle_clarification"

    if decision_type in ("final_answer", "cannot_conclude"):
        return "generate_answer"

    return "reject"


def _route_after_execute(state: LexiorState) -> str:
    if state.get("status") == "rejected":
        return "reject"
    return "plan"


def _route_after_clarification(state: LexiorState) -> str:
    # Chat : la question part vers l'utilisateur réel, le tour se termine.
    if state.get("status") == "clarification":
        return "end"
    return "plan"


def _route_after_generate(state: LexiorState) -> str:
    if state.get("status") == "rejected":
        return "reject"
    return "run_critics"


def _route_after_critics(state: LexiorState) -> str:
    scenario = state.get("scenario")
    if not scenario:
        return "validate_final"

    if scenario.request_type in PRECISE_ARTICLE_TYPES:
        return "validate_final"

    legal = state.get("legal_critic_result")
    agentic = state.get("agentic_critic_result")
    if not legal and not agentic:
        return "validate_final"

    config_max = 1
    repair_needed = False
    if legal and (not legal.accepted or legal.score < 0.7):
        repair_needed = True
    if agentic and (not agentic.accepted or agentic.score < 0.7):
        repair_needed = True

    if repair_needed and state.get("repair_count", 0) < config_max:
        return "repair"

    return "validate_final"


def _route_after_final(state: LexiorState) -> str:
    acceptance = state.get("acceptance")
    if acceptance and acceptance.accepted:
        return "export"
    return "reject"


# ── Graph builder ────────────────────────────────────────────────────────


def build_graph(
    config: AgenticConfig,
    catalog: ToolCatalog,
    planner: PlannerAgent,
    executor: MCPExecutor,
    trajectory_agent: TrajectoryAgent,
    legal_critic: LegalCritic,
    agentic_critic: AgenticCritic,
    storage: Optional[RunStorage] = None,
    checkpointer=None,
):
    """Build and compile the Lexior agent graph.

    Parameters
    ----------
    checkpointer : optional
        A LangGraph checkpointer (e.g. ``SqliteSaver``).  If ``None``,
        the graph runs without persistence.

    Returns
    -------
    CompiledGraph
        Ready to invoke with ``graph.invoke(initial_state)``.
    """
    nodes = GraphNodes(
        config, catalog, planner, executor,
        trajectory_agent, legal_critic, agentic_critic, storage,
    )

    graph = StateGraph(LexiorState)

    graph.add_node("plan", nodes.plan)
    graph.add_node("execute_tool", nodes.execute_tool)
    graph.add_node("handle_clarification", nodes.handle_clarification)
    graph.add_node("generate_answer", nodes.generate_answer)
    graph.add_node("run_critics", nodes.run_critics)
    graph.add_node("repair", nodes.repair)
    graph.add_node("validate_final", nodes.validate_final)
    graph.add_node("export", nodes.export)
    graph.add_node("reject", nodes.reject)

    graph.set_entry_point("plan")

    graph.add_conditional_edges("plan", _route_after_plan, {
        "execute_tool": "execute_tool",
        "handle_clarification": "handle_clarification",
        "generate_answer": "generate_answer",
        "reject": "reject",
    })
    graph.add_conditional_edges("execute_tool", _route_after_execute, {
        "plan": "plan",
        "reject": "reject",
    })
    graph.add_conditional_edges(
        "handle_clarification", _route_after_clarification, {
            "end": END,
            "plan": "plan",
        })
    graph.add_conditional_edges("generate_answer", _route_after_generate, {
        "run_critics": "run_critics",
        "reject": "reject",
    })
    graph.add_conditional_edges("run_critics", _route_after_critics, {
        "repair": "repair",
        "validate_final": "validate_final",
    })
    graph.add_edge("repair", "run_critics")
    graph.add_conditional_edges("validate_final", _route_after_final, {
        "export": "export",
        "reject": "reject",
    })
    graph.add_edge("export", END)
    graph.add_edge("reject", END)

    return graph.compile(checkpointer=checkpointer)
