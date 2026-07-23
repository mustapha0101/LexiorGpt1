# -*- coding: utf-8 -*-
"""Routage du graphe central — LangGraph possède TOUT le routage.

Chaque fonction est pure : elle lit ``LexiorState`` et retourne le nom
du prochain nœud. Les services calculent; les nœuds écrivent l'état;
CES fonctions décident. Aucun autre moteur ne route.

Convention budgets : les compteurs (``repair_count``,
``reformulation_count``, ``clarification_count``) sont incrémentés par
les nœuds; les routes ne font que LIRE l'état.
"""

from __future__ import annotations

from .state import LexiorState

# Cible « fin de recherche » : le contrat de réponse précède toujours la
# rédaction.
ANSWER_PATH = "build_answer_contract"

# Statuts de résultat qui déclenchent une reformulation.
_REFORMULATE_STATUSES = {"irrelevant", "empty"}


def _rejected(state: LexiorState) -> bool:
    return state.get("status") == "rejected"


def _is_live(state: LexiorState) -> bool:
    return state.get("mode") in ("live", "chat")


# ── Après plan ───────────────────────────────────────────────────────────


def route_after_plan(state: LexiorState) -> str:
    if _rejected(state):
        return "reject"
    return "validate_plan"


# ── Après validate_plan ──────────────────────────────────────────────────


def route_after_validate_plan(state: LexiorState) -> str:
    if _rejected(state):
        return "reject"
    decision = state.get("latest_decision") or {}
    decision_type = decision.get("decision", "")
    if decision_type == "call_tool":
        return "execute_tool"
    if decision_type == "ask_clarification":
        return "handle_clarification"
    if decision_type in ("final_answer", "cannot_conclude"):
        return ANSWER_PATH
    return "reject"


# ── Après handle_clarification ───────────────────────────────────────────


def route_after_clarification(state: LexiorState) -> str:
    if _rejected(state):
        return "reject"
    if state.get("stop_reason") == "clarification_required":
        # Dataset sans réponse synthétique : la trajectoire d'entraînement
        # se termine sur la question elle-même.
        return ANSWER_PATH
    # Réponse obtenue (synthétique ou reprise après interrupt) : elle peut
    # changer la juridiction — on repasse par la résolution.
    return "resolve_jurisdiction"


# ── Route outil ──────────────────────────────────────────────────────────


def route_after_execute(state: LexiorState) -> str:
    if _rejected(state):
        return "reject"
    return "verify_tool_result"


def route_after_classification(state: LexiorState) -> str:
    """usable → enregistrement; inutilisable → reformulation ou réparation.

    ``tool_error`` retourne à l'enregistrement : l'exécuteur a déjà
    épuisé ses reprises internes et le planner possède la stratégie de
    repli (changement d'outil ou réponse honnête d'échec).
    """
    if _rejected(state):
        return "reject"
    status = state.get("last_tool_result_status", "")
    reformulations = state.get("reformulation_count", 0)
    max_reformulations = state.get("max_reformulations", 1)
    if (status in _REFORMULATE_STATUSES
            and reformulations < max_reformulations):
        return "reformulate_search"
    if (status == "wrong_document_type"
            and state.get("repair_count", 0) < state.get("max_repairs", 1)):
        return "repair_trajectory"
    return "update_research_state"


# ── Route réponse ────────────────────────────────────────────────────────


def route_after_generate(state: LexiorState) -> str:
    if _rejected(state):
        return "reject"
    return "run_critics"


def route_after_failures(state: LexiorState) -> str:
    """Chaque catégorie d'échec retourne au premier nœud invalide."""
    if _rejected(state):
        return "reject"
    target = state.get("repair_from_node", "")
    if target in ("repair_answer", "repair_trajectory",
                  "resolve_jurisdiction", "handle_clarification",
                  ANSWER_PATH):
        return target
    return "validate_final"


# ── Route finale ─────────────────────────────────────────────────────────


def route_after_acceptance(state: LexiorState) -> str:
    if _rejected(state):
        return "reject"
    acceptance = state.get("acceptance_result")
    accepted = bool(acceptance and acceptance.accepted)
    if accepted:
        return "return_live_answer" if _is_live(state) else "export_dataset"
    # Rejet réparable : un premier pas invalide identifié + budget restant.
    if (state.get("first_invalid_step") is not None
            and state.get("repair_count", 0) < state.get("max_repairs", 1)
            and not _is_live(state)):
        return "repair_trajectory"
    return "reject"


# ── Table des routes (pour les tests et la documentation) ────────────────

CONDITIONAL_ROUTES: dict[str, dict[str, str]] = {
    "plan": {
        "validate_plan": "validate_plan",
        "reject": "reject",
    },
    "validate_plan": {
        "execute_tool": "execute_tool",
        "handle_clarification": "handle_clarification",
        ANSWER_PATH: ANSWER_PATH,
        "reject": "reject",
    },
    "handle_clarification": {
        "resolve_jurisdiction": "resolve_jurisdiction",
        ANSWER_PATH: ANSWER_PATH,
        "reject": "reject",
    },
    "execute_tool": {
        "verify_tool_result": "verify_tool_result",
        "reject": "reject",
    },
    "classify_tool_result": {
        "update_research_state": "update_research_state",
        "reformulate_search": "reformulate_search",
        "repair_trajectory": "repair_trajectory",
        "reject": "reject",
    },
    "generate_answer": {
        "run_critics": "run_critics",
        "reject": "reject",
    },
    "classify_failures": {
        "repair_answer": "repair_answer",
        "repair_trajectory": "repair_trajectory",
        "resolve_jurisdiction": "resolve_jurisdiction",
        "handle_clarification": "handle_clarification",
        ANSWER_PATH: ANSWER_PATH,
        "validate_final": "validate_final",
        "reject": "reject",
    },
    "compute_acceptance": {
        "export_dataset": "export_dataset",
        "return_live_answer": "return_live_answer",
        "repair_trajectory": "repair_trajectory",
        "reject": "reject",
    },
}

ROUTERS = {
    "plan": route_after_plan,
    "validate_plan": route_after_validate_plan,
    "handle_clarification": route_after_clarification,
    "execute_tool": route_after_execute,
    "classify_tool_result": route_after_classification,
    "generate_answer": route_after_generate,
    "classify_failures": route_after_failures,
    "compute_acceptance": route_after_acceptance,
}
