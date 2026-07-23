# -*- coding: utf-8 -*-
"""classify_request — type de demande et type de sortie attendus.

Dataset : le type vient du scénario (vérité de génération).
Live : heuristique déterministe sur le dernier message (pas de LLM ici;
le planner affine ensuite, mais l'état porte déjà une classification
exploitable par le contrat de réponse).
"""

from __future__ import annotations

from typing import Any

from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState
from ._common import is_greeting, requested_output_type

NAME = "classify_request"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    scenario = state["scenario"]
    latest = state.get("latest_user_message", "")

    request_type = scenario.request_type
    if is_live(state.get("mode", "")) and is_greeting(latest):
        request_type = "non_legal"

    return {
        "request_type": request_type,
        "requested_output_type": requested_output_type(latest),
    }
