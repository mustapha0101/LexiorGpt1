# -*- coding: utf-8 -*-
"""run_critics — évaluation juridique et agentique (les deux modes)."""

from __future__ import annotations

from typing import Any

from lexior.services.assertion_grounding import textes_recuperes

from ..context import GraphContext
from ..state import LexiorState, to_research_state

NAME = "run_critics"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    answer = state.get("final_answer", "")
    outcome = ctx.services.critics.evaluate(
        to_research_state(state), answer)
    verdicts = ctx.services.assertion_grounding.verifier(
        answer,
        textes_recuperes(state.get("tool_history", [])),
        faits=(state.get("active_issue")
               or state.get("latest_user_message", "")),
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
