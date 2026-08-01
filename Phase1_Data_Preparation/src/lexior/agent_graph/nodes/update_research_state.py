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

from lexior.agentic.case_law_gate import (
    gate_search_results, is_verified_quebec_decision,
)
from lexior.services.article_review import (
    assess_legislative_sufficiency, enrich_article_review,
)
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
    def field(item: Any, name: str) -> Any:
        if isinstance(item, dict):
            return item.get(name, "")
        return getattr(item, name, "")

    signature = (
        field(value, "tool_name"),
        field(value, "content_hash"),
        field(value, "normalized_response"),
    )
    for item in values:
        if signature == (
                field(item, "tool_name"),
                field(item, "content_hash"),
                field(item, "normalized_response")):
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
        for rank, (number, text) in enumerate(official_texts.items(), start=1):
            verdict = deterministic.get(number) or selection.get(number)
            status = (_review_status(verdict) if verdict else "unreviewed")
            reviews[number] = enrich_article_review(
                article_number=number,
                status=status,
                reason=str(getattr(verdict, "motif", "") or ""),
                text=text,
                facts=case_description,
                rank=rank,
                source=observation.tool_name,
            )

        sufficiency = assess_legislative_sufficiency(
            reviews, case_description, state.get("facts") or {},
            remaining_candidates=any(
                item.tool_name in {"semantic_search_ccq", "semantic_search_cpc"}
                and item.ok for item in visible_history))

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
                "legislative_sufficiency": sufficiency.to_dict(),
            })
            context.update({
                "prior_evidence": prior_evidence,
                "article_reviews": reviews,
                "official_rule_retrieved": True,
                "official_rule_sources": sources,
                "legislative_sufficiency": sufficiency.to_dict(),
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
        classified, status = gate_search_results(
            observation.normalized_response,
            article_nums,
            case_description,
            source_urls=list(observation.source_urls),
        )
        candidate_status = getattr(status, "value", str(status))
        accepted = [item for item in classified
                    if item.source_url and (
                        item.usable or candidate_status == "candidate_pending_fetch")][:2]
        existing_cases = list(state.get("usable_case_sources", []))
        signatures = {(item.citation, item.source_url)
                      for item in existing_cases}
        for item in accepted:
            if (item.citation, item.source_url) not in signatures:
                existing_cases.append(item)
                signatures.add((item.citation, item.source_url))
        updates["usable_case_sources"] = existing_cases
        updates["case_law_search_status"] = (
            "candidate_pending_fetch" if accepted and not any(item.usable for item in accepted) else
            "candidates_pending_fetch" if accepted else
            ("candidates_without_url" if any(item.usable for item in classified)
             else (status.value if hasattr(status, "value") else str(status))))
        context.update({
            "usable_case_sources": updates["usable_case_sources"],
            "case_law_search_status": updates["case_law_search_status"],
        })

    if (observation.tool_name == "get_quebec_regulation"
            and observation.ok
            and is_verified_quebec_decision(observation.normalized_response)
            and (state.get("last_tool_assessment") or {}).get(
                "usable_as_evidence", False)):
        verified = _append_once(
            list(state.get("case_law_verified", [])), observation)
        prior_evidence = _append_once(
            list(state.get("prior_evidence", [])), observation)
        updates.update({
            "case_law_verified": verified,
            "prior_evidence": prior_evidence,
            "case_law_search_status": "verified",
        })
        context.update({
            "case_law_verified": verified,
            "prior_evidence": prior_evidence,
            "case_law_search_status": "verified",
        })

    if context:
        updates["case_context"] = context

    return updates
