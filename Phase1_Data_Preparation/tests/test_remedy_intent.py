# -*- coding: utf-8 -*-

from lexior.agent_graph.events import StreamTranslator
from lexior.agentic.schemas import PrimaryAuthoritySelection
from lexior.services.evidence_first import build_claim_ledger
from lexior.services.remedy_intent import classify_remedy_intent


SOURCE = {
    "ccq:1457": (
        "Toute personne a le devoir de respecter les règles de conduite. "
        "Elle est tenue de réparer le préjudice qu'elle cause par sa faute."
    )
}


def test_ambiguous_plainte_is_answerable_with_civil_distinction():
    intent = classify_remedy_intent("Je veux porter plainte.", source_texts=SOURCE)

    assert intent.ambiguity_detected is True
    assert intent.clarification_required is False
    assert intent.answerable_with_distinctions is True
    assert "civil_compensation" in intent.supported_remedy_types
    assert "police_or_criminal_complaint" in intent.unsupported_remedy_types
    assert intent.distinction_to_explain


def test_explicit_indemnisation_is_classified_as_civil_compensation():
    intent = classify_remedy_intent(
        "Je veux obtenir une indemnisation.", source_texts=SOURCE)

    assert intent.ambiguity_detected is False
    assert intent.clarification_required is False
    assert intent.supported_remedy_types == ["civil_compensation"]


def test_explicit_police_complaint_is_not_supported_by_civil_source():
    intent = classify_remedy_intent(
        "Je veux porter plainte à la police.", source_texts=SOURCE)

    assert intent.primary_interpretation == "police_or_criminal_complaint"
    assert intent.supported_remedy_types == []
    assert "police_or_criminal_complaint" in intent.unsupported_remedy_types
    assert intent.clarification_required is True


def test_municipal_report_is_not_inferred_from_civil_source():
    intent = classify_remedy_intent(
        "Je veux signaler le problème à la ville.", source_texts=SOURCE)

    assert intent.primary_interpretation == "municipal_or_administrative_report"
    assert intent.supported_remedy_types == []
    assert intent.clarification_required is True


def test_ambiguous_poursuivre_prefers_supported_civil_outcome():
    intent = classify_remedy_intent(
        "Puis-je poursuivre mon voisin?", source_texts=SOURCE)

    assert intent.ambiguity_detected is True
    assert intent.primary_interpretation == "civil_compensation"
    assert "civil_compensation" in intent.supported_remedy_types
    assert "civil_proceeding" in intent.unsupported_remedy_types


def test_composite_claim_is_split_and_procedure_is_not_hidden_by_compensation():
    intent = classify_remedy_intent("porter plainte", source_texts=SOURCE)
    ledger = build_claim_ledger(
        "Vous pouvez porter plainte pour obtenir une indemnisation.",
        PrimaryAuthoritySelection(primary_sources=["ccq:1457"]),
        SOURCE,
        remedy_intent=intent,
    )

    assert len(ledger.claims) == 2
    complaint, compensation = ledger.claims
    assert complaint.claim_category == "procedure"
    assert complaint.verification_status == "failed"
    assert compensation.claim_category == "remedy_type"
    assert compensation.verification_status == "verified"


def test_mock_dog_case_remedy_intent_has_no_scenario_specific_branch():
    intent = classify_remedy_intent(
        "Mon chien a causé un dommage et je veux porter plainte.",
        source_texts=SOURCE,
    )

    assert intent.raw_expression == "porter plainte"
    assert intent.primary_interpretation == "civil_compensation"
    assert intent.supported_remedy_evidence["civil_compensation"][0]["source_id"] == "ccq:1457"


def test_remedy_observability_contains_required_fields():
    chunks = StreamTranslator(thread_id="thread-1").translate_chunk({
        "build_answer_contract": {
            "task_id": "task-1",
            "remedy_events": [{
                "event_name": "remedy_ambiguity_detected",
                "raw_expression": "porter plainte",
                "possible_remedy_types": ["civil_compensation"],
                "supported_remedy_types": ["civil_compensation"],
                "unsupported_remedy_types": ["police_or_criminal_complaint"],
                "reason": "distinction",
            }],
        }
    })
    events = [item["event"] for item in chunks if item.get("type") == "observability"]
    remedy = [event for event in events
              if event.get("event_name") == "remedy_ambiguity_detected"][0]

    assert remedy["task_id"] == "task-1"
    assert remedy["thread_id"] == "thread-1"
    assert remedy["raw_expression"] == "porter plainte"
    assert remedy["possible_remedy_types"]
    assert remedy["supported_remedy_types"]
    assert remedy["unsupported_remedy_types"]
    assert remedy["timestamp"]
