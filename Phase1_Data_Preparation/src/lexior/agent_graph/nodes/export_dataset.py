# -*- coding: utf-8 -*-
"""export_dataset — persistance JSONL intermédiaire (mode dataset).

Écrit le format « agentic-2.0 » inchangé. La conversion ChatML reste un
pas déterministe séparé (``agentic_generation.training_formatter``).
"""

from __future__ import annotations

from typing import Any

from ..context import GraphContext
from ..state import LexiorState, to_trajectory

NAME = "export_dataset"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    trajectory = to_trajectory(state)
    trajectory.quality.acceptance = state.get("acceptance_result")
    trajectory.quality.accepted_for_intermediate = True

    exported = ctx.services.export.export_accepted(trajectory)

    return {
        "status": "accepted",
        "trajectory": trajectory.model_dump(mode="json"),
        "export_result": {
            "exported": exported,
            "format": "agentic-2.0",
        },
    }
