# -*- coding: utf-8 -*-
"""reformulate_search — correctif de recherche transmis au planner.

Le résultat inutilisable (non pertinent ou vide) reste dans la trace,
mais un retour correctif explicite est préparé pour la PROCHAINE
proposition du planner. Le compteur borne les reformulations.
"""

from __future__ import annotations

from typing import Any

from lexior.services.result_verification import ToolResultAssessment

from ..context import GraphContext
from ..state import LexiorState

NAME = "reformulate_search"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    raw = state.get("last_tool_assessment") or {}
    assessment = ToolResultAssessment(
        tool_name=raw.get("tool_name", ""),
        tool_call_succeeded=raw.get("tool_call_succeeded", True),
        search_status=raw.get("search_status", "irrelevant"),
        expected_document=raw.get("expected_document", ""),
        returned_document=raw.get("returned_document", ""),
        usable_as_evidence=False,
        reason=raw.get("reason", ""),
    )
    attempt = state.get("reformulation_count", 0) + 1
    feedback = ctx.services.research.build_reformulation_feedback(
        assessment, attempt)

    return {
        "planner_feedback": feedback,
        "reformulation_count": attempt,
        "status": "planning",
        "repair_history": list(state.get("repair_history", [])) + [{
            "from_node": NAME,
            "category": "retrieval",
            "reason": assessment.reason or assessment.search_status,
        }],
    }
