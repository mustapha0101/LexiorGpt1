# -*- coding: utf-8 -*-
"""Regression tests for the source-bounded evidence-first path."""

from lexior.agent_graph.events import StreamTranslator
from lexior.agentic.schemas import RuleContract, ToolObservation
from lexior.services.evidence_first import (
    article_budget,
    build_clarification_decision,
    build_claim_ledger,
    build_rule_contract,
    select_primary_authorities,
)


def _observation(text: str) -> ToolObservation:
    return ToolObservation(
        tool_name="get_ccq_articles",
        arguments={"articles": ["1", "2"]},
        normalized_response=text,
    )


def test_generic_damage_words_do_not_create_fault_or_prior_knowledge():
    observation = _observation(
        "Article 1\nLe gardien est tenu de réparer le préjudice causé par le bien."
    )
    selection = select_primary_authorities(
        [observation], {"1": {"status": "applicable"}},
        case_description="un bien a causé un dommage",
    )
    contract = build_rule_contract(selection, [observation], {
        "1": {"status": "applicable"}}, {},
    )
    assert not contract.blocking_facts
    assert not contract.conditional_facts or all(
        "connaissance" not in fact.description.casefold()
        for fact in contract.conditional_facts
    )
    assert "prior_knowledge" in contract.facts_not_required
    assert "failure_to_take_reasonable_action" in contract.facts_not_required


def test_special_source_is_ranked_by_fact_alignment_not_article_identifier():
    observations = [_observation(
        "Article 1\nLe propriétaire de l'animal est tenu de réparer le préjudice "
        "causé par le fait autonome de celui-ci.\n"
        "Article 2\nToute personne doit respecter les règles de conduite applicables."
    )]
    selection = select_primary_authorities(
        observations,
        {"1": {"status": "applicable"}, "2": {"status": "applicable"}},
        case_description="un animal a causé un préjudice",
    )
    assert selection.primary_sources[0] == "ccq:1"


def test_conditional_source_does_not_authorize_clarification():
    contract = RuleContract(
        task_id="task-1",
        conditional_branches=["Si le texte vise cette situation, alors la conséquence source s'applique."],
    )
    decision = build_clarification_decision(contract, {}, [], task_id="task-1")
    assert decision.needed is False
    assert decision.answerable_conditionally is True


def test_uncertain_answer_is_already_asked_and_cannot_repeat():
    contract = RuleContract(
        task_id="task-1",
        blocking_facts=[{
            "fact_id": "identity",
            "description": "l'identité de la personne visée",
            "source_ids": ["ccq:1"],
            "supporting_passages": ["La personne visée"],
            "status": "blocking",
        }],
    )
    decision = build_clarification_decision(
        contract, {"identity": {"value": None, "status": "asked_but_uncertain"}},
        [{"clarification_id": "rule-fact-identity", "fact_keys": ["identity"],
          "status": "asked_but_uncertain"}],
    )
    assert decision.needed is False
    assert decision.already_asked is True
    assert decision.user_answer_status == "asked_but_uncertain"


def test_evidence_first_budget_is_three_then_one_more_batch():
    class Config:
        evidence_first_enabled = True
        evidence_first_initial_candidate_count = 5
        evidence_first_initial_fetch_count = 3
        evidence_first_maximum_article_batches = 2

    first = article_budget(Config(), fetched_count=0, batch_index=0)
    second = article_budget(Config(), fetched_count=3, batch_index=1)
    third = article_budget(Config(), fetched_count=6, batch_index=2)
    assert first["candidate_count"] == 5
    assert first["fetch_count"] == 3
    assert first["allow_next"] is True
    assert second["allow_next"] is False
    assert third["allow_next"] is False


def test_claim_ledger_requires_a_supporting_passage():
    selection = select_primary_authorities(
        [_observation("Article 1\nLe gardien peut réparer le préjudice causé.")],
        {"1": {"status": "applicable"}},
        case_description="gardien bien",
    )
    ledger = build_claim_ledger(
        "La loi impose toujours une indemnisation automatique.", selection,
        {"ccq:1": "Le gardien peut réparer le préjudice causé."},
    )
    assert ledger.claims[0].verification_status == "failed"


def test_observability_exposes_rule_and_normative_events():
    events = list(StreamTranslator(thread_id="thread-1").translate_chunk({
        "extract_rule_contract": {
            "task_id": "task-1",
            "thread_id": "thread-1",
            "normative_references": [{"status": "resolved"}],
            "rule_contract": {"extraction_status": "source_bounded"},
        }
    }))
    observations = [item["event"] for item in events
                    if item.get("type") == "observability"]
    assert observations
    assert {
        "rule_contract_built", "source_sufficiency_decided",
        "normative_reference_detected", "normative_reference_resolved",
    } <= set(observations[0]["event_names"])
