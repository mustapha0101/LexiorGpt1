# -*- coding: utf-8 -*-
"""Modes d'application et fallbacks juridiques génériques."""

from lexior.agent_graph.nodes.validate_final import (
    _conditional_evidence_fallback,
    _safe_unresolved_fallback,
)
from lexior.agentic.schemas import (
    PrimaryAuthoritySelection,
    RuleContract,
    RuleElement,
    RuleFact,
    ToolObservation,
)
from lexior.services.evidence_first import (
    LegalClaimVerificationService,
    build_rule_contract,
    decide_source_sufficiency,
)


def _rule(*, unknown: bool) -> RuleContract:
    return RuleContract(
        task_id="t",
        primary_source_ids=["ccq:1"],
        elements=[RuleElement(
            id="e1", description="règle", support_source_ids=["ccq:1"],
            supporting_passages=["Texte officiel"], status="present",
        )],
        conditional_facts=[RuleFact(
            fact_id="f1", description="date déterminante",
            value_status="unknown" if unknown else "known_true",
            status="conditional", question="À quelle date?",
        )],
        conditional_branches=["Si la date est antérieure, la première branche s'applique."],
    )


def test_sources_suffisantes_avec_fait_inconnu_donnent_mode_conditionnel():
    decision = decide_source_sufficiency(
        PrimaryAuthoritySelection(primary_sources=["ccq:1"]),
        _rule(unknown=True), [],
    )
    assert decision.application_mode == "conditional_application"


def test_sources_suffisantes_sans_fait_inconnu_donnent_mode_direct():
    decision = decide_source_sufficiency(
        PrimaryAuthoritySelection(primary_sources=["ccq:1"]),
        _rule(unknown=False), [],
    )
    assert decision.application_mode == "direct_application"


def test_absence_de_source_donne_mode_insuffisant():
    decision = decide_source_sufficiency(
        PrimaryAuthoritySelection(), _rule(unknown=True), [],
    )
    assert decision.application_mode == "insufficient_legal_evidence"


def test_fallback_conditionnel_ne_contient_aucun_conseil_metier_code_en_dur():
    contract = {
        "articles_retenus": ["1"],
        "rule_contract": _rule(unknown=True).model_dump(mode="json"),
        "questions_decisives": [{"question": "À quelle date?"}],
    }
    answer = _conditional_evidence_fallback(
        {"1": "Le texte officiel de la règle."}, contract)
    folded = answer.casefold()
    assert "article 1" in folded
    assert "à quelle date" in folded
    assert "photos" not in folded
    assert "responsabilité n'est donc pas automatique" not in folded


def test_fallback_sans_source_reste_neutre_et_reutilise_la_question():
    answer = _safe_unresolved_fallback({
        "questions_decisives": [{"question": "Quel document avez-vous reçu?"}],
    })
    assert "Quel document avez-vous reçu?" in answer
    assert "photos" not in answer.casefold()


def test_source_secondaire_complete_le_contrat_et_ses_faits_conditionnels():
    observation = ToolObservation(
        tool_name="get_ccq_articles",
        arguments={"articles": [1, 2]},
        normalized_response=(
            "Article 1\nLe vendeur doit garantir le bien.\n\n"
            "Article 2\nSi le vendeur connaissait le défaut, il doit réparer."
        ),
        ok=True,
    )
    selection = PrimaryAuthoritySelection(
        primary_sources=["ccq:1"], secondary_sources=["ccq:2"])
    contract = build_rule_contract(
        selection, [observation], {
            "1": {"status": "applicable"},
            "2": {
                "status": "conditionally_applicable",
                "required_application_facts": [{
                    "id": "seller_knowledge",
                    "description": "le vendeur connaissait le défaut",
                    "question": "Le vendeur connaissait-il le défaut?",
                    "passage_source": "Si le vendeur connaissait le défaut",
                }],
            },
        }, {}, task_id="t",
    )
    assert contract.secondary_source_ids == ["ccq:2"]
    assert {sid for element in contract.elements
            for sid in element.support_source_ids} == {"ccq:1", "ccq:2"}
    assert [fact.fact_id for fact in contract.conditional_facts] == [
        "seller_knowledge"]
    assert contract.supporting_facts == []
    assert contract.facts_not_required == []


def test_question_factuelle_nest_pas_une_affirmation_juridique():
    ledger = LegalClaimVerificationService().verify_answer(
        "Avez-vous dénoncé le problème dans un délai raisonnable?",
        PrimaryAuthoritySelection(primary_sources=["ccq:1"]),
        {"ccq:1": "Article 1\nLe délai doit être raisonnable."},
        rule_contract=RuleContract(primary_source_ids=["ccq:1"]),
    )
    assert ledger.claims == []
