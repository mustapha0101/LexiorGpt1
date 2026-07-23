# -*- coding: utf-8 -*-
"""return_live_answer — livraison de la réponse live avec ses sources.

Pas d'export dataset automatique en mode live; une trace comparable à
la trajectoire dataset est néanmoins conservée dans l'état
(``trajectory``) pour l'observabilité.
"""

from __future__ import annotations

from typing import Any

from ..context import GraphContext
from ..state import LexiorState, to_trajectory

NAME = "return_live_answer"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    sources = list(dict.fromkeys(state.get("sources", [])))
    trajectory = to_trajectory(state)
    trajectory.quality.acceptance = state.get("acceptance_result")

    return {
        "status": "accepted",
        "sources": sources,
        "trajectory": trajectory.model_dump(mode="json"),
        "export_result": None,  # jamais d'export automatique en live
    }
