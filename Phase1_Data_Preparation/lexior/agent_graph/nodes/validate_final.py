# -*- coding: utf-8 -*-
"""validate_final — validation déterministe de la trajectoire complète.

Dataset : validation stricte (grounding, doublons, budget, séquence).
Live : les mêmes validateurs tournent en OBSERVATION (les problèmes
sont consignés dans ``validation_issues`` mais ne bloquent pas la
livraison — les bloqueurs live sont gérés par le contrat de réponse).
"""

from __future__ import annotations

from typing import Any

from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState, to_trajectory

NAME = "validate_final"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    live = is_live(state.get("mode", ""))
    trajectory = to_trajectory(state)
    exempt = state.get("exempt_tools") or (
        ctx.services.validation.compute_exempt_tools(
            state.get("tool_history", [])))

    validation = ctx.services.validation.validate_trajectory(
        trajectory,
        allow_mock=(ctx.config.offline or ctx.config.dry_run or live),
        max_tool_calls=state.get("max_tool_calls", 4),
        exempt_tools=exempt,
    )
    validation.warnings.extend(ctx.services.validation.sequence_warnings(
        state["scenario"].request_type,
        [o.tool_name for o in state.get("tool_history", [])],
    ))

    return {
        "validation_result": validation,
        "validation_issues": list(validation.errors)
        + list(validation.warnings),
        "deterministic_blockers": ([] if live else list(validation.errors)),
        "deterministic_validation": bool(validation.valid),
        "exempt_tools": exempt,
    }
