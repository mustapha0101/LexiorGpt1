# -*- coding: utf-8 -*-
"""Non-régressions de clarification live, sans appel réseau."""

from __future__ import annotations

from pathlib import Path

import pytest

from lexior.agent_graph import build_context
from lexior.agent_graph.nodes.validate_plan import run
from lexior.agent_graph.state import initial_state
from lexior.agentic.config import load_config
from lexior.agentic.schemas import (
    Decision, Message, PlannerDecision, Role, ScenarioSpec,
)
from lexior.agentic.tool_catalog import load_catalog
from lexior.services import build_services
from lexior.services.jurisdiction import detect_jurisdiction_hint


PHASE1 = Path(__file__).resolve().parent.parent
REPO = PHASE1.parent


class _ExecuteurInerte:
    def execute(self, call):  # pragma: no cover - jamais appelé ici
        raise AssertionError("aucun outil ne doit être exécuté")


@pytest.mark.parametrize("text", [
    "Au Québec, la situation s'est produite hier.",
    "J'habite à Montréal.",
])
def test_lieu_quebecois_accentue_est_reconnu(text):
    assert detect_jurisdiction_hint([
        Message(role=Role.user, content=text)
    ]) == "Québec"


@pytest.fixture(scope="module")
def contexte():
    cfg = load_config(str(PHASE1 / "configs" / "agentic_generation.yaml"))
    catalog = load_catalog(str(REPO / "docs" / "mcp_tools_catalog.json"))
    services = build_services(
        cfg, catalog, executor=_ExecuteurInerte(), teacher=None,
        critic_client=None,
    )
    return build_context(cfg, catalog, services)


def _state(*, intent: str, domain: str = "unknown",
           jurisdiction_material: bool = False,
           employment_material: bool = False,
           decision: Decision = Decision.final_answer,
           scope: str = "none"):
    scenario = ScenarioSpec(
        scenario_id="policy", scenario_family_id="policy",
        request_type="unknown", user_query="Question de test", language="fr",
    )
    state = initial_state(scenario, mode="live")
    state.update({
        "request_intent": intent,
        "request_type": "case_analysis" if intent == "legal" else "non_legal",
        "legal_domain": domain,
        "jurisdiction_material": jurisdiction_material,
        "employment_regime_material": employment_material,
        "latest_decision": PlannerDecision(
            request_type="case_analysis",
            decision=decision,
            clarification_scope=scope,
            clarification_question="Question produite par le planner?"
            if decision == Decision.ask_clarification else None,
        ).model_dump(mode="json"),
    })
    return state


@pytest.mark.parametrize("intent", ["greeting", "non_legal"])
def test_hors_droit_ne_demande_jamais_la_juridiction(contexte, intent):
    update = run(_state(
        intent=intent, jurisdiction_material=True,
        employment_material=True, decision=Decision.ask_clarification,
        scope="legal_regime",
    ), contexte)
    decision = update["latest_decision"]
    assert decision["decision"] == "final_answer"
    assert decision["clarification_scope"] == "none"
    assert not update.get("pending_clarification")


def test_demande_ambigue_recoit_une_question_neutre(contexte):
    update = run(_state(intent="ambiguous"), contexte)
    pending = update["pending_clarification"]
    assert pending["category"] == "request_intent"
    assert pending["question"] == "Que souhaitez-vous savoir ou faire?"
    assert "province" not in pending["question"].casefold()
    assert "employeur" not in pending["question"].casefold()


def test_logement_demande_seulement_le_lieu_si_necessaire(contexte):
    update = run(_state(
        intent="legal", domain="housing", jurisdiction_material=True,
        employment_material=False, decision=Decision.ask_clarification,
        scope="legal_regime",
    ), contexte)
    pending = update["pending_clarification"]
    assert pending["category"] == "jurisdiction"
    assert "province" in pending["question"].casefold()
    assert "employeur" not in pending["question"].casefold()
    assert "travail" not in pending["question"].casefold()


def test_emploi_demande_lieu_et_activite_de_lemployeur(contexte):
    update = run(_state(
        intent="legal", domain="employment", jurisdiction_material=True,
        employment_material=True,
    ), contexte)
    pending = update["pending_clarification"]
    assert pending["category"] == "legal_regime"
    assert "province" in pending["question"].casefold()
    assert "activité principale" in pending["question"].casefold()
    assert "employeur" in pending["question"].casefold()


def test_dependance_structurelle_ecrase_une_question_factuelle_du_planner(contexte):
    update = run(_state(
        intent="legal", domain="housing", jurisdiction_material=True,
        employment_material=False, decision=Decision.ask_clarification,
        scope="application_fact",
    ), contexte)
    assert update["pending_clarification"]["category"] == "jurisdiction"


def test_regime_emploi_regroupe_meme_si_le_planner_demande_seulement_le_lieu(
        contexte):
    update = run(_state(
        intent="legal", domain="employment", jurisdiction_material=True,
        employment_material=True, decision=Decision.ask_clarification,
        scope="jurisdiction",
    ), contexte)
    pending = update["pending_clarification"]
    assert pending["category"] == "legal_regime"
    assert pending["fact_keys"] == ["work_location", "employment_sector"]


def test_question_juridique_independante_du_lieu_ne_force_rien(contexte):
    update = run(_state(
        intent="legal", domain="federal", jurisdiction_material=False,
        employment_material=False,
    ), contexte)
    assert update["latest_decision"]["decision"] == "final_answer"
    assert not update.get("pending_clarification")
