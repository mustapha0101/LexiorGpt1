# -*- coding: utf-8 -*-
"""run_critics — évaluation juridique et agentique (les deux modes)."""

from __future__ import annotations

from typing import Any

from lexior.services.assertion_grounding import textes_recuperes

from ..context import GraphContext
from ..state import (
    LexiorState,
    canonical_case_description,
    to_research_state,
    visible_tool_history,
)

NAME = "run_critics"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    answer = state.get("final_answer", "")
    outcome = ctx.services.critics.evaluate(
        to_research_state(state), answer)
    verdicts = ctx.services.assertion_grounding.verifier(
        answer,
        textes_recuperes(visible_tool_history(state)),
        faits=canonical_case_description(state),
    )
    grounding_issues = [verdict.probleme() for verdict in verdicts
                        if not verdict.soutenue]
    return {
        "critic_results": {
            "legal": outcome.legal,
            "agentic": outcome.agentic,
        },
        "preflight_grounding_issues": grounding_issues,
    }
