# -*- coding: utf-8 -*-
"""run_critics — évaluation juridique et agentique (les deux modes)."""

from __future__ import annotations

from typing import Any

from ..context import GraphContext
from ..state import LexiorState, to_research_state

NAME = "run_critics"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    outcome = ctx.services.critics.evaluate(
        to_research_state(state), state.get("final_answer", ""))
    return {
        "critic_results": {
            "legal": outcome.legal,
            "agentic": outcome.agentic,
        },
    }
