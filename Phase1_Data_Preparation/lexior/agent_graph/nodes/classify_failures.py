# -*- coding: utf-8 -*-
"""classify_failures — chaque échec retourne au premier nœud invalide.

Classe les sorties des critiques en catégories typées (writing,
retrieval, jurisdiction, clarification, ...) et désigne le nœud cible.
Convention budget : ce nœud paie (incrémente ``repair_count``) pour les
redirections vers des nœuds PARTAGÉS (resolve_jurisdiction,
handle_clarification, build_answer_contract); les nœuds ``repair_*``
paient eux-mêmes à l'entrée.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from lexior.services.critics import CriticsOutcome

from ..context import GraphContext
from ..state import LexiorState

NAME = "classify_failures"

_SHARED_TARGETS = {
    "resolve_jurisdiction", "handle_clarification", "build_answer_contract",
}
_REPAIR_TARGETS = {"repair_answer", "repair_trajectory"}


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    critics = state.get("critic_results", {}) or {}
    outcome = CriticsOutcome(
        legal=critics.get("legal"), agentic=critics.get("agentic"))

    reports = ctx.services.repair.classify_failures(outcome)
    updates: dict[str, Any] = {
        "failure_reports": [asdict(r) for r in reports],
    }

    primary = ctx.services.repair.primary(reports)
    repair_count = state.get("repair_count", 0)
    max_repairs = state.get("max_repairs", 1)

    if primary is None or repair_count >= max_repairs:
        updates["repair_from_node"] = "validate_final"
        return updates

    target = primary.target_node
    live = state.get("mode") in ("live", "chat")
    clarification_cap = 2 if live else 1
    if (target == "handle_clarification"
            and state.get("clarification_count", 0) >= clarification_cap):
        updates["repair_from_node"] = "validate_final"
        return updates

    updates["repair_from_node"] = target
    updates["repair_history"] = list(state.get("repair_history", [])) + [{
        "from_node": NAME,
        "category": primary.category,
        "target_node": target,
        "instructions": list(primary.instructions),
    }]
    if target in _SHARED_TARGETS:
        updates["repair_count"] = repair_count + 1
    if target == "repair_trajectory":
        updates["planner_feedback"] = "; ".join(primary.instructions)

    return updates
