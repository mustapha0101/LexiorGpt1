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

from lexior.agentic.case_law_gate import gate_search_results
from lexior.services.assertion_grounding import (
    articles_incompatibles_deterministes,
    textes_recuperes,
)
from lexior.services.provenance import numeros_demandes

from ..context import GraphContext
from ..state import (
    LexiorState,
    canonical_case_description,
    visible_tool_history,
)

NAME = "update_research_state"

_OFFICIAL_RULE_TOOLS = ("get_ccq_articles", "get_cpc_articles")


def _review_status(verdict: Any) -> str:
    """Traduit le vocabulaire du relecteur en contrat stable du dossier."""
    status = str(getattr(verdict, "statut", "") or "").strip().lower()
    return {
        "applicable": "applicable",
        "incertain": "conditionally_applicable",
        "incompatible": "incompatible",
    }.get(status, "unreviewed")


def _append_once(values: list[Any], value: Any) -> list[Any]:
    signature = (
        getattr(value, "tool_name", ""),
        getattr(value, "content_hash", ""),
        getattr(value, "normalized_response", ""),
    )
    for item in values:
        if signature == (
                getattr(item, "tool_name", ""),
                getattr(item, "content_hash", ""),
                getattr(item, "normalized_response", "")):
            return values
    return [*values, value]


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    tool_history = state.get("tool_history", [])
    if not tool_history:
        return {"status": "planning"}

    observation = tool_history[-1]
    updates: dict[str, Any] = {"status": "planning"}
    context = dict(state.get("case_context") or {})
    visible_history = visible_tool_history(state)
    case_description = canonical_case_description(state)

    if observation.tool_name in _OFFICIAL_RULE_TOOLS and observation.ok:
        official_texts = textes_recuperes(visible_history)
        deterministic = articles_incompatibles_deterministes(
            official_texts, faits=case_description)
        selection = ctx.services.assertion_grounding.selectionner_articles(
            official_texts, faits=case_description)
        reviews = dict(state.get("article_reviews", {}))
        for number, text in official_texts.items():
            verdict = deterministic.get(number) or selection.get(number)
            status = (_review_status(verdict) if verdict else "unreviewed")
            reviews[number] = {
                "status": status,
                "reason": str(getattr(verdict, "motif", "") or "")[:300],
                "source": observation.tool_name,
                "reviewed": bool(verdict),
                "text_available": bool(text.strip()),
            }

        # Une récupération officiellement réussie devient une preuve durable
        # uniquement si son texte a été effectivement parsé. Les résultats
        # sémantiques et les candidats de jurisprudence restent hors dossier.
        if official_texts:
            prior_evidence = list(state.get("prior_evidence", []))
            prior_evidence = _append_once(prior_evidence, observation)
            sources = list(state.get("official_rule_sources", []))
            if observation.tool_name not in sources:
                sources.append(observation.tool_name)
            updates.update({
                "official_rule_retrieved": True,
                "official_rule_sources": sources,
                "prior_evidence": prior_evidence,
                "article_reviews": reviews,
            })
            context.update({
                "prior_evidence": prior_evidence,
                "article_reviews": reviews,
                "official_rule_retrieved": True,
                "official_rule_sources": sources,
            })

    if (observation.tool_name == "search_quebec_jurisprudence"
            and observation.ok):
        reviews = updates.get("article_reviews") or state.get(
            "article_reviews", {})
        article_nums = [
            number for number, review in reviews.items()
            if review.get("status") in {
                "applicable", "conditionally_applicable"}
        ]
        if not article_nums:
            article_nums = list(dict.fromkeys(
                number
                for item in visible_history
                if item.tool_name in _OFFICIAL_RULE_TOOLS and item.ok
                for number in numeros_demandes(item.tool_name, item.arguments)
            ))
        usable, status = gate_search_results(
            observation.normalized_response,
            article_nums,
            case_description,
        )
        existing_cases = list(state.get("usable_case_sources", []))
        updates["usable_case_sources"] = existing_cases + list(usable)
        updates["case_law_search_status"] = (
            "candidates_pending_fetch" if usable else
            (status.value if hasattr(status, "value") else str(status)))
        context.update({
            "usable_case_sources": updates["usable_case_sources"],
            "case_law_search_status": updates["case_law_search_status"],
        })

    if context:
        updates["case_context"] = context

    return updates
