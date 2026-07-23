# -*- coding: utf-8 -*-
"""resolve_jurisdiction — l'unique écrivain de la juridiction résolue.

Live : détection déterministe sur toute la conversation (le signal
explicite le plus récent l'emporte et VERROUILLE la valeur — une
juridiction verrouillée ne change jamais silencieusement).
Dataset : amorçage depuis le scénario; les décisions du planner
raffinent ensuite via ``validate_plan``.
"""

from __future__ import annotations

from typing import Any

from lexior.services.jurisdiction import JurisdictionResolution
from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState

NAME = "resolve_jurisdiction"


def _previous(state: LexiorState) -> JurisdictionResolution:
    return JurisdictionResolution(
        value=state.get("resolved_jurisdiction", ""),
        basis=state.get("jurisdiction_basis", ""),
        locked=state.get("jurisdiction_locked", False),
        verified=state.get("jurisdiction_verified", False),
    )


def _updates(resolution: JurisdictionResolution) -> dict[str, Any]:
    return {
        "resolved_jurisdiction": resolution.value,
        "jurisdiction_status": resolution.status,
        "jurisdiction_basis": resolution.basis,
        "jurisdiction_locked": resolution.locked,
        "jurisdiction_verified": resolution.verified,
        "candidate_jurisdiction": resolution.value,
    }


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    service = ctx.services.jurisdiction
    previous = _previous(state)

    if is_live(state.get("mode", "")):
        resolution = service.resolve_live(
            state.get("messages", []), previous)
        updates = _updates(resolution)
        if resolution.basis == "explicit_user_statement":
            updates["work_location"] = resolution.value
        return updates

    resolution = service.resolve_dataset(state["scenario"], "", previous)
    return _updates(resolution)
