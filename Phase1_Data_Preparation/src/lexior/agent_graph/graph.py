# -*- coding: utf-8 -*-
"""Le graphe central Lexior — UN StateGraph, deux modes.

Topologie ::

    START → initialize → classify_request → classify_follow_up
          → update_active_task → resolve_jurisdiction → analyze_facts
          → plan → validate_plan
                     ├─ handle_clarification → (resolve_jurisdiction | build_answer_contract)
                     ├─ execute_tool → verify_tool_result → classify_tool_result
                     │      ├─ update_research_state → plan
                     │      ├─ reformulate_search   → plan
                     │      └─ repair_trajectory    → plan
                     ├─ build_answer_contract → generate_answer → run_critics
                     │      → classify_failures
                     │           ├─ repair_answer → run_critics
                     │           ├─ repair_trajectory → plan
                     │           ├─ resolve_jurisdiction (reprise du cycle)
                     │           ├─ handle_clarification
                     │           └─ validate_final → compute_acceptance
                     │                ├─ export_dataset     → END   (dataset)
                     │                ├─ return_live_answer → END   (live)
                     │                ├─ repair_trajectory  → plan
                     │                └─ reject → END
                     └─ reject → END

Les nœuds vivent dans ``nodes/`` (un module par nœud), le routage dans
``routing.py``. Les services calculent; LangGraph route; rien d'autre
ne boucle.
"""

from __future__ import annotations

import functools

from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from . import nodes as node_registry
from .context import GraphContext
from .routing import ROUTERS, CONDITIONAL_ROUTES
from .state import LexiorState

# Chaîne linéaire d'ouverture de tour.
_LINEAR_PREFIX = (
    "initialize",
    "classify_request",
    "classify_follow_up",
    "update_active_task",
    "resolve_jurisdiction",
    "analyze_facts",
    "plan",
)

# Arêtes fixes terminales. Les transitions internes passent par un routeur
# gardé afin qu'un Command(goto="reject") ne soit jamais concurrencé par une
# arête statique après une exception.
_STATIC_EDGES = (
    ("export_dataset", END),
    ("return_live_answer", END),
    ("reject", END),
)

_GUARDED_STATIC_EDGES = (
    ("verify_tool_result", "classify_tool_result"),
    ("update_research_state", "select_primary_authorities"),
    ("select_primary_authorities", "extract_rule_contract"),
    ("extract_rule_contract", "derive_rule_specific_facts"),
    ("derive_rule_specific_facts", "plan"),
    ("reformulate_search", "plan"),
    ("repair_trajectory", "plan"),
    ("build_answer_contract", "generate_answer"),
    ("run_critics", "classify_failures"),
    ("repair_answer", "run_critics"),
    ("validate_final", "compute_acceptance"),
)


def _wrap(node_name: str, fn, ctx: GraphContext):
    """Uniformise la gestion d'erreur des nœuds.

    Toute exception devient un rejet propre (jamais un crash de run) —
    sauf ``GraphInterrupt``, mécanisme de clarification live, qui DOIT
    se propager.
    """

    @functools.wraps(fn)
    def _node(state: LexiorState) -> dict | Command:
        try:
            update = fn(state, ctx)
            if isinstance(update, dict):
                # Make task identity available to every SSE update, including
                # nodes whose normal payload is otherwise intentionally small.
                update.setdefault("task_id", state.get("task_id", ""))
                update.setdefault("thread_id", state.get("thread_id", ""))
            return update
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — rejet contrôlé
            reason = f"{node_name}: {type(exc).__name__}: {exc}"
            return Command(
                update={
                    "status": "rejected",
                    "stop_reason": reason,
                    "deterministic_blockers": [reason],
                    "node_failed": node_name,
                    "error_type": type(exc).__name__,
                    "task_id": state.get("task_id", ""),
                    "thread_id": state.get("thread_id", ""),
                },
                goto="reject",
            )

    return _node


def build_graph(context: GraphContext, checkpointer=None):
    """Construit et compile LE graphe central (partagé par les modes).

    Parameters
    ----------
    context : GraphContext
        Services et configuration injectés dans chaque nœud.
    checkpointer : optional
        Checkpointer LangGraph. Requis pour ``interrupt()`` (live).
    """
    graph = StateGraph(LexiorState)

    for module in node_registry.NODE_MODULES:
        graph.add_node(module.NAME, _wrap(module.NAME, module.run, context))

    graph.set_entry_point(_LINEAR_PREFIX[0])
    for upstream, downstream in zip(_LINEAR_PREFIX, _LINEAR_PREFIX[1:]):
        # resolve_jurisdiction / analyze_facts sont aussi des cibles de
        # boucle; les arêtes linéaires restent valides pour LangGraph.
        graph.add_edge(upstream, downstream)

    for source, router in ROUTERS.items():
        graph.add_conditional_edges(source, router,
                                    dict(CONDITIONAL_ROUTES[source]))

    for upstream, downstream in _GUARDED_STATIC_EDGES:
        graph.add_conditional_edges(
            upstream,
            lambda state, target=downstream: (
                "reject" if state.get("status") == "rejected" else target),
            {downstream: downstream, "reject": "reject"},
        )

    for upstream, downstream in _STATIC_EDGES:
        graph.add_edge(upstream, downstream)

    return graph.compile(checkpointer=checkpointer)


def build_default_graph(config, catalog, *, executor,
                        teacher=None, critic_client=None, storage=None,
                        checkpointer=None):
    """Fabrique le contexte puis compile le graphe (raccourci)."""
    from .context import build_context

    context = build_context(
        config, catalog,
        executor=executor, teacher=teacher,
        critic_client=critic_client, storage=storage,
    )
    return build_graph(context, checkpointer=checkpointer), context


# Compatibilité : anciens noms de fonctions de routage réexportés.
from .routing import (  # noqa: E402,F401
    route_after_acceptance,
    route_after_classification,
    route_after_clarification,
    route_after_execute,
    route_after_failures,
    route_after_generate,
    route_after_plan,
    route_after_validate_plan,
)
