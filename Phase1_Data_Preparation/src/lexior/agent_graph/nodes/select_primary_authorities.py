# -*- coding: utf-8 -*-
"""Select an evidence allowlist before any answer contract is built."""

from __future__ import annotations

from lexior.services.evidence_first import select_primary_authorities

from ..context import GraphContext
from ..state import LexiorState, visible_tool_history

NAME = "select_primary_authorities"


def run(state: LexiorState, ctx: GraphContext) -> dict:
    task_id = state.get("task_id", "")
    if not ctx.config.evidence_first_enabled:
        return {"task_id": task_id, "status": "planning"}
    selection = select_primary_authorities(
        visible_tool_history(state), state.get("article_reviews", {}),
        task_id=task_id,
        case_description=state.get("active_issue") or state.get(
            "latest_user_message", ""),
        facts=state.get("facts") or {},
        maximum_primary=min(
            ctx.config.evidence_first_maximum_primary_sources,
            ctx.config.evidence_first_initial_candidate_count,
            ctx.config.evidence_first_initial_fetch_count,
        ),
        maximum_secondary=ctx.config.evidence_first_maximum_secondary_sources,
    )
    return {
        "thread_id": state.get("thread_id", ""),
        "task_id": task_id,
        "primary_authority_selection": selection,
        "status": "planning",
    }
