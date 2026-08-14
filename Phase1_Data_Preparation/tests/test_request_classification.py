# -*- coding: utf-8 -*-
"""Contrat de classification live, sans appel réseau."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lexior.agent_graph.nodes.classify_request import run
from lexior.agent_graph.nodes.classify_follow_up import run as classify_follow_up
from lexior.agent_graph.runner import GraphRunner
from lexior.agent_graph.state import initial_state, to_research_state, to_trajectory
from lexior.agentic.schemas import RequestIntent, ScenarioSpec
from lexior.services.request_classifier import RequestClassifierService


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete_json(self, role, messages, temperature=0.0):
        self.calls.append((role, messages, temperature))
        if isinstance(self.output, Exception):
            raise self.output
        return dict(self.output)


@pytest.mark.parametrize("text", [
    "bonjout", "bonsoir", "coucou", "hey", "merci!",
])
def test_salutation_est_classee_par_le_sens_sans_liste_locale(text):
    client = FakeClient({
        "intent": "greeting",
        "request_type": "case_analysis",
        "legal_domain": "unknown",
        "jurisdiction_material": True,
        "employment_regime_material": True,
        "confidence": 0.98,
        "reason": "prise de contact",
    })
    result = RequestClassifierService(client).classify(text)
    assert result.intent is RequestIntent.greeting
    assert result.request_type == "non_legal"
    assert result.legal_domain == "none"
    assert not result.jurisdiction_material
    assert not result.employment_regime_material
    assert client.calls[0][0] == "request_classifier"


def test_question_juridique_en_langage_courant_est_conservee():
    client = FakeClient({
        "intent": "legal",
        "request_type": "case_analysis",
        "legal_domain": "civil_liability",
        "jurisdiction_material": True,
        "employment_regime_material": False,
        "confidence": 0.94,
        "reason": "responsabilité possible",
    })
    result = RequestClassifierService(client).classify(
        "Mon fils a cassé la vitrine du dépanneur")
    assert result.intent is RequestIntent.legal
    assert result.request_type == "case_analysis"
    assert result.jurisdiction_material


def test_panne_du_classificateur_devient_ambiguite_neutre():
    result = RequestClassifierService(FakeClient(RuntimeError("boom"))).classify(
        "bonjout")
    assert result.intent is RequestIntent.ambiguous
    assert result.request_type == "unknown"
    assert not result.jurisdiction_material


def test_classification_peu_confiante_ne_declenche_pas_le_droit():
    result = RequestClassifierService(FakeClient({
        "intent": "legal",
        "request_type": "case_analysis",
        "legal_domain": "housing",
        "jurisdiction_material": True,
        "employment_regime_material": False,
        "confidence": 0.4,
        "reason": "interprétation incertaine",
    }), minimum_confidence=0.65).classify("message très ambigu")
    assert result.intent is RequestIntent.ambiguous
    assert result.request_type == "unknown"
    assert not result.jurisdiction_material


def test_domaine_non_emploi_ne_peut_pas_exiger_le_secteur():
    client = FakeClient({
        "intent": "legal",
        "request_type": "case_analysis",
        "legal_domain": "housing",
        "jurisdiction_material": True,
        "employment_regime_material": True,
        "confidence": 0.9,
        "reason": "bail",
    })
    result = RequestClassifierService(client).classify(
        "Mon propriétaire augmente mon loyer")
    assert not result.employment_regime_material


def test_noeud_ecrit_toute_la_classification():
    classifier = RequestClassifierService(FakeClient({
        "intent": "legal",
        "request_type": "case_analysis",
        "legal_domain": "housing",
        "jurisdiction_material": True,
        "employment_regime_material": False,
        "confidence": 0.93,
        "reason": "litige locatif",
    }))
    scenario = ScenarioSpec(
        scenario_id="t", scenario_family_id="t", request_type="unknown",
        user_query="Mon loyer augmente", language="fr")
    state = initial_state(scenario, mode="live")
    context = SimpleNamespace(services=SimpleNamespace(
        request_classifier=classifier))
    update = run(state, context)
    assert update["request_type"] == "case_analysis"
    assert update["legal_domain"] == "housing"
    assert update["jurisdiction_material"] is True


def test_projection_transmet_la_classification_au_planner():
    scenario = ScenarioSpec(
        scenario_id="t", scenario_family_id="t", request_type="unknown",
        user_query="Question", language="fr")
    state = initial_state(scenario, mode="live")
    state.update({
        "request_type": "procedure_guidance",
        "request_intent": "legal",
        "legal_domain": "procedure",
        "jurisdiction_material": True,
        "classification_confidence": 0.91,
    })
    projected = to_research_state(state)
    assert projected.scenario.request_type == "procedure_guidance"
    assert projected.request_intent == "legal"
    assert projected.legal_domain == "procedure"


def test_runner_ne_force_plus_case_analysis():
    assert GraphRunner.build_live_state.__kwdefaults__["request_type"] == "unknown"


def test_trajectoire_live_utilise_le_type_semantique_et_non_le_seed_unknown():
    scenario = ScenarioSpec(
        scenario_id="t", scenario_family_id="t", request_type="unknown",
        user_query="Question", language="fr")
    state = initial_state(scenario, mode="live")
    state.update({
        "request_type": "case_analysis",
        "legal_domain": "housing",
    })
    trajectory = to_trajectory(state)
    assert trajectory.request_type == "case_analysis"
    assert trajectory.legal_domain == "housing"


def test_reponse_a_la_clarification_dintention_reste_dans_la_meme_tache():
    scenario = ScenarioSpec(
        scenario_id="t", scenario_family_id="t", request_type="unknown",
        user_query="Pouvez-vous préciser?", language="fr")
    state = initial_state(scenario, mode="live")
    state.update({
        "latest_user_message": "Je veux connaître mes recours après la vente",
        "clarification_answer": "Je veux connaître mes recours après la vente",
        "last_clarification_category": "request_intent",
    })
    update = classify_follow_up(state, SimpleNamespace())
    assert update["refers_to_previous_answer"] is True
