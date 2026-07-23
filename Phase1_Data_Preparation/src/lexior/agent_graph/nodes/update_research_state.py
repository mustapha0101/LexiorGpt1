# -*- coding: utf-8 -*-
"""update_research_state — enregistrement d'une preuve dans les
collections appropriées.

Met à jour les drapeaux de recherche (règle officielle récupérée,
jurisprudence filtrée par le gate) et enregistre les coverage gaps
avant de retourner à ``plan``.

Only results classified as ``usable`` may enter ``usable_evidence``.
Retrieval-only semantic-search outputs never enter ``citable_sources``
or ``usable_evidence``. Official article text must still pass relevance
verification before entering ``usable_evidence``.
"""

from __future__ import annotations

from typing import Any

from agentic_generation.case_law_gate import gate_search_results

from ..context import GraphContext
from ..state import LexiorState

NAME = "update_research_state"

_OFFICIAL_RULE_TOOLS = ("get_ccq_articles", "get_cpc_articles")


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    tool_history = state.get("tool_history", [])
    if not tool_history:
        return {"status": "planning"}

    observation = tool_history[-1]
    updates: dict[str, Any] = {"status": "planning"}

    if observation.tool_name in _OFFICIAL_RULE_TOOLS and observation.ok:
        updates["official_rule_retrieved"] = True
        updates["official_rule_sources"] = list(
            state.get("official_rule_sources", [])
        ) + [observation.tool_name]

    if (observation.tool_name == "search_quebec_jurisprudence"
            and observation.ok):
        article_nums = [
            str(o.arguments.get("start_article", ""))
            for o in tool_history
            if o.tool_name in _OFFICIAL_RULE_TOOLS
            and o.ok and o.arguments.get("start_article")
        ]
        usable, status = gate_search_results(
            observation.normalized_response,
            article_nums,
            state["scenario"].user_query,
        )
        updates["usable_case_sources"] = list(
            state.get("usable_case_sources", [])
        ) + list(usable)
        updates["case_law_search_status"] = (
            status.value if hasattr(status, "value") else str(status))

    return updates
