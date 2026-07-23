# -*- coding: utf-8 -*-
"""reject — rejet final, avec raisons complètes et trace persistée."""

from __future__ import annotations

from typing import Any

from lexior.agentic.schemas import (
    RejectionDetail,
    RejectionRecord,
    RepairReport,
)

from ..context import GraphContext
from ..state import LexiorState, to_trajectory

NAME = "reject"

_PLANNER_MARKERS = (
    "décision", "planner", "clarification répétée", "outil requis",
    "route", "limite de",
)


def _stage(state: LexiorState, reasons: list[str]) -> str:
    acceptance = state.get("acceptance_result")
    if acceptance is not None and getattr(
            acceptance, "blocking_errors", None):
        return "validator"
    joined = " ".join(reasons).lower()
    if any(marker in joined for marker in _PLANNER_MARKERS):
        return "planner"
    return "graph"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    scenario = state["scenario"]

    acceptance = state.get("acceptance_result")
    if acceptance is not None and getattr(
            acceptance, "blocking_errors", None):
        reasons = list(acceptance.blocking_errors)
    elif state.get("deterministic_blockers"):
        reasons = list(state.get("deterministic_blockers"))
    else:
        reasons = [state.get("stop_reason") or "rejet sans raison"]

    stage = _stage(state, reasons)
    trajectory = to_trajectory(state) if stage == "validator" else None

    rejection = RejectionRecord(
        scenario_id=scenario.scenario_id,
        request_type=scenario.request_type,
        stage=stage,
        reasons=list(dict.fromkeys(reasons)),
        trajectory=(trajectory.model_dump(mode="json")
                    if trajectory else None),
    )
    ctx.services.export.export_rejected(rejection)

    repair = state.get("repair", RepairReport())
    detail = state.get("rejection_detail") or RejectionDetail(
        scenario_id=scenario.scenario_id,
        blocking_reason=reasons[0] if reasons else "",
        repair_attempted=repair.attempted,
        repair_successful=repair.status == "successful",
        first_invalid_step=state.get("first_invalid_step"),
    )

    return {
        "status": "rejected",
        "stop_reason": reasons[0] if reasons else "",
        "rejection_detail": detail,
        "trajectory": (trajectory.model_dump(mode="json")
                       if trajectory else state.get("trajectory")),
        "export_result": {"exported": ctx.services.export.storage
                          is not None,
                          "rejection": rejection.model_dump(mode="json")},
    }
