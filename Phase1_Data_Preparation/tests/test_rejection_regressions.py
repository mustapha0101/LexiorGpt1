import json

import pytest

from agentic_generation.agentic_critic import AGENTIC_CRITIC_SYSTEM
from agentic_generation.legal_critic import LEGAL_CRITIC_SYSTEM
from agentic_generation.planner_agent import PlannerAgent
from agentic_generation.scenario_generator import ScenarioGenerator
from agentic_generation.schemas import Decision, Message, ResearchState, Role, ScenarioSpec, TrainingTrajectory
from agentic_generation.trajectory_agent import TrajectoryAgent, normalize_final_answer
from agentic_generation.validators import validate_tool_route, validate_trajectory


class JsonClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, *args, **kwargs):
        return dict(self.payload)


class TextClient:
    def __init__(self, answer):
        self.answer = answer
        self.messages = None

    def complete(self, role, messages, temperature=None):
        self.messages = messages
        return self.answer


def _scenario_payload(query, **extra):
    payload = {
        "user_query": query,
        "legal_domain": "domaine incorrect fourni par le modèle",
        "facts_provided": {"entreprise": "XYZ Inc.", "montant": "5000 $"},
        "facts_missing": [],
        "clarification_answer": None,
        "request_type": "catégorie_falsifiée",
        "expected_jurisdiction": "juridiction falsifiée",
    }
    payload.update(extra)
    return payload


def test_parking_ticket_cannot_become_ccq_clarification_case():
    client = JsonClient(_scenario_payload(
        "Puis-je contester une amende de stationnement?",
        clarification_answer="Elle date du 15 septembre 2023.",
    ))
    scenario = ScenarioGenerator(client=client, seed=3407).generate(
        "topic_research")
    assert "stationnement" not in scenario.user_query.casefold()
    assert "code civil" in scenario.user_query.casefold()
    assert scenario.request_type == "topic_research"


def test_penal_question_cannot_become_civil_procedure_search():
    client = JsonClient(_scenario_payload(
        "Quels délais du CPC s'appliquent à une amende de stationnement?"
    ))
    scenario = ScenarioGenerator(client=client, seed=3407).generate(
        "topic_research")
    assert "stationnement" not in scenario.user_query.casefold()
    assert "code civil" in scenario.user_query.casefold()


def test_non_federal_case_gets_a_clear_federal_anchor():
    client = JsonClient(_scenario_payload(
        "Mon voisin refuse de réparer notre clôture. Que faire?"
    ))
    scenario = ScenarioGenerator(client=client, seed=3407).generate(
        "case_analysis")
    assert "banque" in scenario.user_query.casefold()
    assert scenario.jurisdiction_status == "supported_federal"


def test_vague_article_explanation_names_the_ccq():
    client = JsonClient(_scenario_payload(
        "Peux-tu m'expliquer l'article 11 de la loi?"
    ))
    scenario = ScenarioGenerator(client=client, seed=3407).generate(
        "article_explanation")
    assert "code civil du québec" in scenario.user_query.casefold()


def test_vague_federal_law_request_gets_an_identifiable_statute():
    client = JsonClient(_scenario_payload(
        "Comment obtenir des documents selon la loi fédérale canadienne?"
    ))
    scenario = ScenarioGenerator(client=client, seed=3407).generate("law_or_regulation_identification")
    assert "loi sur les banques" in scenario.user_query.casefold()


def test_hidden_scenario_facts_are_not_sent_to_trajectory_writer():
    client = TextClient('{"answer":"Réponse fondée uniquement sur le message."}')
    scenario = ScenarioSpec(
        scenario_id="s", scenario_family_id="f", request_type="non_legal",
        user_query="Bonjour!", facts_provided={"entreprise": "XYZ Inc.", "montant": "5000 $"},
    )
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content="Bonjour!")],
    )
    _thinking, answer = TrajectoryAgent(client=client).final_answer(state)
    serialized_prompt = client.messages[-1]["content"]
    assert answer == "Réponse fondée uniquement sur le message."
    assert "XYZ Inc." not in serialized_prompt
    assert "5000 $" not in serialized_prompt


def test_json_answer_wrapper_is_unwrapped_and_guarded(catalog):
    assert normalize_final_answer('{"answer":"Texte en prose."}') == "Texte en prose."
    row = TrainingTrajectory(
        scenario_id="s", scenario_family_id="f", request_type="non_legal",
        messages=[
            Message(role=Role.user, content="Bonjour"),
            Message(role=Role.assistant, content='{"answer":"Bonjour!"}'),
        ],
    )
    result = validate_trajectory(row, catalog, allow_mock=True)
    assert "réponse finale enveloppée dans un objet JSON" in result.errors


def test_non_route_tool_caught_by_validation(catalog):
    """A tool not in the expected route is caught by route validation even
    if the route policy allows it (no forbidden tools)."""
    errors = validate_tool_route("case_analysis", ["search_ccq_keywords"])
    assert any("requis absent" in error for error in errors)


def test_critic_policies_cover_observed_false_rejections():
    assert "non_legal" in LEGAL_CRITIC_SYSTEM
    assert "N'exige JAMAIS" in LEGAL_CRITIC_SYSTEM
    assert "qualité du scénario" in LEGAL_CRITIC_SYSTEM
    assert "jamais la qualité" in AGENTIC_CRITIC_SYSTEM
    assert "étapes marquées optionnelles" in AGENTIC_CRITIC_SYSTEM


def test_non_legal_route_requires_no_tool():
    assert validate_tool_route("non_legal", []) == []
    assert validate_tool_route("non_legal", ["get_ccq_articles"])
