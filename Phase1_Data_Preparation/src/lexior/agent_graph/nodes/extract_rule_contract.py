# -*- coding: utf-8 -*-
"""Build a source-bounded RuleContract from the selected authority."""

from __future__ import annotations

from lexior.services.evidence_first import (
    build_rule_contract,
    decide_source_sufficiency,
    normative_references,
    retrieved_articles,
    validate_rule_contract,
)

from ..context import GraphContext
from ..state import LexiorState, visible_tool_history

NAME = "extract_rule_contract"


def run(state: LexiorState, ctx: GraphContext) -> dict:
    if not ctx.config.evidence_first_enabled:
        return {"task_id": state.get("task_id", ""), "status": "planning"}
    selection = state.get("primary_authority_selection")
    if isinstance(selection, dict):
        from lexior.agentic.schemas import PrimaryAuthoritySelection
        selection = PrimaryAuthoritySelection.model_validate(selection)
    if selection is None:
        from lexior.agentic.schemas import PrimaryAuthoritySelection
        selection = PrimaryAuthoritySelection(task_id=state.get("task_id", ""))
    contract = build_rule_contract(
        selection, visible_tool_history(state), state.get("article_reviews", {}),
        state.get("facts", {}), task_id=state.get("task_id", ""),
    )
    sufficiency = decide_source_sufficiency(
        selection, contract, visible_tool_history(state),
        task_id=state.get("task_id", ""),
        jurisprudence_requested=state.get("request_type") == "case_law_research",
    )
    articles = retrieved_articles(visible_tool_history(state))
    contract_errors = validate_rule_contract(
        contract, set(articles),
        set(item.source_id for item in selection.rejected_sources),
    )
    context = dict(state.get("case_context") or {})
    context.update({
        "task_id": state.get("task_id", ""),
        "primary_authority_selection": selection.model_dump(mode="json"),
        "rule_contract": contract.model_dump(mode="json"),
        "source_sufficiency_decision": sufficiency.model_dump(mode="json"),
        "normative_references": normative_references(visible_tool_history(state)),
    })
    return {
        "task_id": state.get("task_id", ""),
        "rule_contract": contract,
        "source_sufficiency_decision": sufficiency,
        "normative_references": normative_references(visible_tool_history(state)),
        "case_context": context,
        "deterministic_blockers": list(dict.fromkeys(
            [*state.get("deterministic_blockers", []), *contract_errors])),
        "missing_facts_before_application": list(contract.decisive_facts_needed),
        "status": "planning",
    }
