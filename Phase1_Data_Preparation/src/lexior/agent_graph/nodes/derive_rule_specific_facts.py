# -*- coding: utf-8 -*-
"""Compare selected rule elements with facts, without adding facts."""

from __future__ import annotations

from ..context import GraphContext
from ..state import LexiorState

NAME = "derive_rule_specific_facts"


def run(state: LexiorState, ctx: GraphContext) -> dict:
    contract = state.get("rule_contract")
    if isinstance(contract, dict):
        decisive = list(contract.get("decisive_facts_needed", []))
        not_required = list(contract.get("facts_not_required", []))
        elements = list(contract.get("elements", []))
    else:
        decisive = list(getattr(contract, "decisive_facts_needed", []))
        not_required = list(getattr(contract, "facts_not_required", []))
        elements = [element.model_dump(mode="json") for element in getattr(contract, "elements", [])]
    facts = state.get("facts", {}) or {}
    known = {str(key) for key, value in facts.items() if value not in (None, "", [], {})}
    missing = [fact for fact in decisive if str(fact) not in known and str(fact) not in not_required]
    analysis = dict(state.get("fact_analysis") or {})
    analysis["legal_elements"] = elements
    analysis["decisive_facts_missing"] = missing
    analysis["facts_not_required"] = not_required
    return {
        "task_id": state.get("task_id", ""),
        "missing_facts_before_application": missing,
        "missing_critical_facts": missing,
        "fact_analysis": analysis,
        "status": "planning",
    }
