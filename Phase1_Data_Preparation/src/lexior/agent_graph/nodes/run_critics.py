# -*- coding: utf-8 -*-
"""run_critics — évaluation juridique et agentique (les deux modes)."""

from __future__ import annotations

from typing import Any

from ..context import GraphContext
from ..state import (
    LexiorState,
    to_research_state,
)

NAME = "run_critics"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    answer = state.get("final_answer", "")
    outcome = ctx.services.critics.evaluate(
        to_research_state(state), answer)
    return {
        "critic_results": {
            "legal": outcome.legal,
            "agentic": outcome.agentic,
        },
        # Claim verification has one authority: validate_final builds the
        # canonical ClaimLedger once. Critics score writing/agentic quality;
        # they do not run a competing proposition judge.
        "preflight_grounding_issues": [],
    }
