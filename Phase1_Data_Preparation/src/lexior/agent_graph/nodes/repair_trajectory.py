# -*- coding: utf-8 -*-
"""repair_trajectory — la recherche est fautive; on répare le PROCESSUS.

Prépare le retour vers ``plan`` avec un correctif explicite (mauvaise
loi récupérée, route incomplète, mauvais type de document). La cause et
la cible sont consignées dans ``repair_history``.
"""

from __future__ import annotations

from typing import Any

from ..context import GraphContext
from ..state import LexiorState

NAME = "repair_trajectory"


def _build_feedback(state: LexiorState) -> str:
    feedback = state.get("planner_feedback", "")
    if feedback:
        return feedback

    acceptance = state.get("acceptance_result")
    if acceptance is not None and getattr(acceptance, "blocking_errors", None):
        return ("La trajectoire a été refusée : "
                + "; ".join(acceptance.blocking_errors)
                + ". Corrige la recherche en conséquence.")

    assessment = state.get("last_tool_assessment") or {}
    if assessment.get("search_status") == "wrong_document_type":
        return ("Le dernier résultat est du mauvais type de document. "
                "Reformule pour viser le bon type (décision vs loi).")

    blockers = state.get("deterministic_blockers", [])
    if blockers:
        return "; ".join(blockers)
    return "La trajectoire de recherche est invalide; corrige-la."


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    feedback = _build_feedback(state)
    first_invalid = state.get("first_invalid_step")

    return {
        "repair_count": state.get("repair_count", 0) + 1,
        "repair_from_node": "",
        "planner_feedback": feedback,
        "status": "planning",
        "stop_reason": "",
        "repair_history": list(state.get("repair_history", [])) + [{
            "from_node": NAME,
            "category": "retrieval",
            "first_invalid_step": first_invalid,
            "feedback": feedback,
        }],
    }
