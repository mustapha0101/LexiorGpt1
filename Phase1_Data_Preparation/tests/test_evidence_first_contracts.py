# -*- coding: utf-8 -*-
"""Focused tests for the evidence-first contracts."""

from lexior.agentic.schemas import ToolObservation
from lexior.services.evidence_first import (
    build_claim_ledger,
    build_rule_contract,
    merge_failures,
    normalize_and_repair_tool_args,
    select_primary_authorities,
)


def _article(text: str) -> ToolObservation:
    return ToolObservation(
        tool_name="get_ccq_articles",
        arguments={"articles": ["100", "101"]},
        normalized_response=text,
    )


def test_primary_authorities_are_an_allowlist_and_are_bounded():
    observations = [_article(
        "Article 100\nToute personne doit réparer le préjudice causé par sa faute.\n"
        "Article 101\nLe propriétaire doit exécuter les travaux nécessaires pour éviter un danger."
    )]
    reviews = {
        "100": {"status": "applicable", "retrieval_group": "primary", "rule_roles": ["general_liability_basis"]},
        "101": {"status": "incompatible", "reason": "événement déjà réalisé"},
    }
    selection = select_primary_authorities(observations, reviews, task_id="t1")
    assert selection.task_id == "t1"
    assert selection.primary_sources == ["ccq:100"]
    assert all(source.source_id != "ccq:101" for source in selection.rejected_sources) is False


def test_rule_contract_uses_only_retrieved_source_propositions():
    observations = [_article(
        "Article 100\nToute personne doit réparer le préjudice causé par sa faute."
    )]
    selection = select_primary_authorities(
        observations, {"100": {"status": "conditionally_applicable", "retrieval_group": "primary", "missing_fact_keys": ["causation"]}},
        task_id="t1",
    )
    contract = build_rule_contract(selection, observations, {
        "100": {"status": "conditionally_applicable", "retrieval_group": "primary", "missing_fact_keys": ["causation"]}
    }, {}, task_id="t1")
    assert contract.primary_source_ids == ["ccq:100"]
    assert contract.decisive_facts_needed == []
    assert "causation" in contract.facts_not_required
    assert contract.extraction_status == "source_bounded"
    assert contract.elements[0].supporting_passages
    assert all(source.startswith("ccq:") for element in contract.elements for source in element.support_source_ids)


def test_search_query_is_repaired_from_the_active_task():
    class Catalog:
        tools = {"semantic_search_ccq": type("Spec", (), {"properties": {"query": {"type": "string"}}})()}

        @staticmethod
        def validate_call(_tool, arguments):
            return [] if arguments.get("query") else ["query missing"]

    args, audit, errors = normalize_and_repair_tool_args(
        Catalog(), "semantic_search_ccq", {"unexpected": True},
        active_task={"normalized_query": "responsabilité civile"},
    )
    assert args == {"query": "responsabilité civile"}
    assert audit["removed_fields"] == ["unexpected"]
    assert audit["repaired_fields"] == ["query"]
    assert errors == []


def test_failure_merge_is_append_only_and_claim_ledger_keeps_unsupported_claim():
    merged = merge_failures([{"failure_type": "wrong_source_type", "reason": "x"}], [], node="compute_acceptance")
    assert merged == [{"failure_type": "wrong_source_type", "reason": "x"}]
    selection = select_primary_authorities(
        [_article("Article 100\nToute personne doit réparer le préjudice causé par sa faute.")],
        {"100": {"status": "applicable", "retrieval_group": "primary"}}, task_id="t1")
    ledger = build_claim_ledger(
        "Selon l'article 999, le propriétaire doit toujours indemniser.",
        selection, {"ccq:100": "Toute personne doit réparer le préjudice causé par sa faute."},
        task_id="t1")
    assert ledger.claims[0].verification_status == "failed"
