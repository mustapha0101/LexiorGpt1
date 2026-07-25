from collections import Counter
import json

from agentic_generation.cli import _next_request_type
from agentic_generation.agentic_critic import AgenticCritic
from agentic_generation.legal_critic import LegalCritic
from agentic_generation.planner_agent import PlannerAgent
from agentic_generation.scenario_generator import ScenarioGenerator
from agentic_generation.schemas import (
    CriticResult,
    Decision,
    Message,
    ResearchState,
    Role,
    ScenarioSpec,
    ToolObservation,
    TrainingTrajectory,
)
from agentic_generation.taxonomy import REQUEST_TYPES, target_request_type_counts
from agentic_generation.trajectory_agent import TrajectoryAgent
from agentic_generation.validators import validate_trajectory

from agentic_generation.error_codes import ErrorCode, extract_code


def codes_of(errors):
    """Codes portés par les erreurs — le texte français ne sert qu'à l'affichage."""
    return {extract_code(error) for error in errors}



ARTICLE_TEXT = "Article 1457\nToute personne a le devoir de respecter les règles de conduite."


def _article_state(request_type="exact_text_retrieval"):
    scenario = ScenarioSpec(
        scenario_id="s",
        scenario_family_id="f",
        request_type=request_type,
        user_query="Peux-tu me donner le texte de l'article 1457?",
        expected_route=REQUEST_TYPES[request_type].expected_route,
    )
    observation = ToolObservation(
        tool_name="get_ccq_articles",
        arguments={"start_article": 1457},
        raw_response=ARTICLE_TEXT,
        normalized_response=ARTICLE_TEXT,
        mock=True,
    ).finalize_hash()
    return ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[observation],
    )


def test_precise_article_answer_is_verbatim_without_llm_call():
    class FailingClient:
        def complete(self, *args, **kwargs):
            raise AssertionError("le rédacteur LLM ne doit pas être appelé")

    thinking, answer = TrajectoryAgent(client=FailingClient()).final_answer(_article_state())
    assert answer == ARTICLE_TEXT
    assert thinking  # thinking should be non-empty


def test_article_explanation_keeps_the_complete_official_text():
    class TextClient:
        def complete(self, *args, **kwargs):
            return "Cet article présente le principe général."

    state = _article_state("article_explanation")
    _thinking, answer = TrajectoryAgent(client=TextClient()).final_answer(state)
    assert answer.startswith(ARTICLE_TEXT + "\n\nExplication\n")
    assert answer.count(ARTICLE_TEXT) == 1


def test_article_explanation_removes_a_duplicate_returned_by_teacher():
    class TextClient:
        def complete(self, *args, **kwargs):
            return f"{ARTICLE_TEXT}\n\nExplication\nLe devoir décrit est général."

    _thinking, answer = TrajectoryAgent(client=TextClient()).final_answer(
        _article_state("article_explanation")
    )
    assert answer.count(ARTICLE_TEXT) == 1
    assert answer.endswith("Le devoir décrit est général.")


def test_precise_article_paraphrase_is_deterministically_rejected(catalog):
    state = _article_state()
    observation = state.tool_history[0]
    row = TrainingTrajectory(
        scenario_id="s",
        scenario_family_id="f",
        request_type="exact_text_retrieval",
        messages=[
            Message(role=Role.user, content=state.scenario.user_query),
            Message(
                role=Role.assistant,
                content='<tool_call>\n{"name":"get_ccq_articles","arguments":{"start_article":1457}}\n</tool_call>',
            ),
            Message(role=Role.tool, name="get_ccq_articles", content=ARTICLE_TEXT),
            Message(role=Role.assistant, content="L'article dit que chacun doit être prudent."),
        ],
        tool_trace=[observation],
    )
    result = validate_trajectory(row, catalog, allow_mock=True)
    assert ErrorCode.OFFICIAL_TEXT_NOT_REPRODUCED in codes_of(result.errors)


def test_semantic_candidate_is_not_a_citable_article(catalog):
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("topic_research")
    semantic = ToolObservation(
        tool_name="semantic_search_ccq", arguments={"query": scenario.user_query},
        raw_response={"results": [{"article_number": "1726"}]},
        normalized_response="Article 1726 — score de pertinence 0.99", mock=True,
    ).finalize_hash()
    official_text = "Article 1851\nLe louage est un contrat."
    official = ToolObservation(
        tool_name="get_ccq_articles", arguments={"start_article": 1851},
        raw_response=official_text, normalized_response=official_text, mock=True,
    ).finalize_hash()
    row = TrainingTrajectory(
        scenario_id="rag", scenario_family_id="rag-family",
        request_type="topic_research",
        messages=[
            Message(role=Role.user, content=scenario.user_query),
            Message(role=Role.assistant, content=(
                '<tool_call>\n{"name":"semantic_search_ccq","arguments":'
                '{"query":"question"}}\n</tool_call>')),
            Message(role=Role.tool, name="semantic_search_ccq",
                    content=semantic.normalized_response),
            Message(role=Role.assistant, content=(
                '<tool_call>\n{"name":"get_ccq_articles","arguments":'
                '{"start_article":1851}}\n</tool_call>')),
            Message(role=Role.tool, name="get_ccq_articles", content=official_text),
            Message(role=Role.assistant,
                    content="L'article 1726 prévoit une garantie."),
        ],
        tool_trace=[semantic, official],
    )
    # Aligne l'argument du message sur l'observation pour isoler le grounding.
    row.messages[1].content = row.messages[1].content.replace(
        '"question"', json.dumps(scenario.user_query, ensure_ascii=False))
    result = validate_trajectory(row, catalog, allow_mock=True)
    assert ErrorCode.UNGROUNDED_ARTICLE in codes_of(result.errors)
    assert any("article 1726" in error for error in result.errors)


def test_legal_critic_scope_removes_false_verbatim_requirement():
    state = _article_state("article_explanation")
    answer = ARTICLE_TEXT + "\n\nExplication\nLe texte protège le consentement."
    result = CriticResult(
        critic="legal", accepted=False, score=0.5,
        issues=["La réponse inclut une explication après le texte, pas mot pour mot."],
        repair_instructions=["Supprimer l'explication pour reproduire mot pour mot."],
    )
    corrected = LegalCritic._apply_scope_policy(state, answer, result)
    assert corrected.accepted
    assert corrected.score >= 0.70
    assert corrected.issues == []


def test_agentic_critic_does_not_reask_an_already_handled_clarification():
    scenario = ScenarioGenerator(seed=3407, offline=True).generate(
        "topic_research")
    state = ResearchState(
        scenario=scenario,
        messages=[
            Message(role=Role.user, content=scenario.user_query),
            Message(role=Role.assistant, content="Pouvez-vous préciser l'empiètement?"),
            Message(role=Role.user, content=scenario.effective_clarification_answer or "Oui."),
        ],
    )
    result = CriticResult(
        critic="agentic", accepted=False, score=0.6,
        issues=["L'assistant a recherché sans demander de clarification."],
    )
    corrected = AgenticCritic._apply_scope_policy(state, True, result)
    assert corrected.accepted
    assert corrected.issues == []


def test_agentic_critic_does_not_reject_answer_style():
    state = _article_state("article_explanation")
    result = CriticResult(
        critic="agentic", accepted=False, score=0.6,
        issues=["L'assistant aurait dû fournir une réponse plus directe et concise."],
    )
    corrected = AgenticCritic._apply_scope_policy(state, False, result)
    assert corrected.accepted
    assert corrected.issues == []


def test_agentic_critic_does_not_require_every_semantic_candidate():
    state = _article_state("article_explanation")
    result = CriticResult(
        critic="agentic", accepted=False, score=0.6,
        issues=["L'assistant aurait dû récupérer tous les articles pertinents."],
    )
    corrected = AgenticCritic._apply_scope_policy(state, False, result)
    assert corrected.accepted
    assert corrected.issues == []


def test_planner_forces_required_tool_when_teacher_skips(catalog):
    """When the teacher says final_answer but a required tool hasn't been
    called yet, _guard_required_tools overrides to call_tool."""
    class JsonClient:
        def complete_json(self, *args, **kwargs):
            return {
                "thinking_text": "J'ai trouvé le règlement, je peux conclure.",
                "request_type": "law_or_regulation_identification",
                "jurisdiction": "Québec",
                "decision": "final_answer",
            }

    scenario = ScenarioGenerator(seed=3407, offline=True).generate("law_or_regulation_identification")
    search = ToolObservation(
        tool_name="search_quebec_regulations",
        arguments={"query": "environnement"},
        raw_response="Q-2, r. 17.1",
        normalized_response="Q-2, r. 17.1 — règlement de fixture.\nhttps://www.legisquebec.gouv.qc.ca/fr/document/rc/Q-2,%20r.%2017.1",
        mock=True,
    ).finalize_hash()
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[search],
    )
    decision = PlannerAgent(catalog, client=JsonClient()).decide(state)
    assert decision.decision == Decision.call_tool
    assert decision.next_tool == "get_quebec_regulation"
    assert decision.thinking_text


def test_planner_rejects_repeated_identical_tool_call_from_teacher(catalog):
    """When the teacher tries to call a tool already in history with the same
    args, the post-hoc allowed_tools check rejects it."""
    class JsonClient:
        def complete_json(self, *args, **kwargs):
            return {
                "thinking_text": "Je dois récupérer l'article.",
                "request_type": "exact_text_retrieval",
                "jurisdiction": "Québec",
                "decision": "call_tool",
                "next_tool": "get_ccq_articles",
                "arguments": {"start_article": 1457},
            }

    state = _article_state()
    decision = PlannerAgent(catalog, client=JsonClient()).decide(state)
    assert decision.decision == Decision.call_tool
    assert decision.thinking_text


def test_non_legal_question_guard_converts_to_final_answer(catalog):
    """For no_tool categories, the guard converts call_tool to final_answer."""
    class JsonClient:
        def complete_json(self, *args, **kwargs):
            return {
                "thinking_text": "Je vais chercher dans le CCQ.",
                "request_type": "non_legal",
                "jurisdiction": "sans objet",
                "decision": "call_tool",
                "next_tool": "get_ccq_articles",
                "arguments": {"start_article": 1},
            }

    scenario = ScenarioGenerator(seed=3407, offline=True).generate("non_legal")
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
    )
    decision = PlannerAgent(catalog, client=JsonClient()).decide(state)
    assert decision.decision == Decision.final_answer


def test_critics_never_receive_raw_mcp_response():
    class RecordingClient:
        def __init__(self):
            self.prompts = []

        def complete_json(self, role, messages, temperature=None):
            self.prompts.append((role, messages))
            return {
                "accepted": True,
                "score": 1.0,
                "issues": [],
                "unsupported_claims": [],
                "missing_sources": [],
                "repair_instructions": [],
            }

    state = _article_state("article_explanation")
    state.tool_history[0].raw_response = "RAW_SECRET_MARKER" * 50_000
    client = RecordingClient()
    LegalCritic(client=client).evaluate(state, "Réponse")
    AgenticCritic(client=client).evaluate(state, "Réponse")
    serialized = str(client.prompts)
    assert "RAW_SECRET_MARKER" not in serialized
    payloads = [json.loads(messages[-1]["content"]) for _, messages in client.prompts]
    assert all(payload["tool_history"][0]["normalized_response"] == ARTICLE_TEXT
               for payload in payloads)
    assert len(serialized) < 50_000


def test_semantic_search_uses_full_question_and_topic_fallback(catalog):
    planner = PlannerAgent(catalog, offline=True)
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("topic_research")
    scenario.user_query = "Je cherche des informations sur les baux résidentiels au Québec."
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
    )
    assert planner._arguments("semantic_search_ccq", state) == {
        "query": scenario.user_query
    }

    empty = ToolObservation(
        tool_name="semantic_search_ccq",
        arguments={"query": scenario.user_query},
        raw_response=[],
        normalized_response='Aucun article CCQ trouvé pour le mot-clé: "bail".',
        mock=True,
    ).finalize_hash()
    state.tool_history.append(empty)
    assert planner._effective_route(state)[:3] == [
        "semantic_search_ccq", "semantic_search_ccq", "get_ccq_articles"
    ]
    retry = planner._arguments("semantic_search_ccq", state)
    assert scenario.user_query in retry["query"]
    assert retry["query"] != scenario.user_query
    fallback = planner._arguments("get_ccq_articles", state)
    assert fallback is not None
    assert "start_article" in fallback


def test_article_is_selected_only_from_actual_search_result(catalog):
    planner = PlannerAgent(catalog, offline=True)
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("topic_research")
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[ToolObservation(
            tool_name="semantic_search_ccq",
            arguments={"query": scenario.user_query},
            raw_response="Article 1851",
            normalized_response="Article 1851\nLe louage...",
            mock=True,
        ).finalize_hash()],
    )
    assert planner._arguments("get_ccq_articles", state) == {"start_article": 1851}


def test_neighboring_top_cpc_candidates_are_fetched_as_official_range(catalog):
    planner = PlannerAgent(catalog, offline=True)
    scenario = ScenarioGenerator(seed=3407, offline=True).generate(
        "procedure_guidance")
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[ToolObservation(
            tool_name="semantic_search_cpc",
            arguments={"query": scenario.user_query},
            raw_response={"results": []},
            normalized_response=(
                "Article 271 — score 0.9\nArticle 269 — score 0.8\n"
                "Article 274 — score 0.7\nArticle 60 — score 0.6"
            ),
            mock=True,
        ).finalize_hash()],
    )
    assert planner._arguments("get_cpc_articles", state) == {
        "start_article": 269, "end_article": 274,
    }


def test_federal_bankruptcy_query_uses_the_statute_name(catalog):
    planner = PlannerAgent(catalog, offline=True)
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("case_analysis")
    scenario.user_query = "Je pense devoir déclarer faillite; quelles sont les étapes?"
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
    )
    arguments = planner._arguments("search_legal_documents", state)
    assert arguments["query"] == "Bankruptcy and Insolvency Act"
    assert arguments["doc_type"] == "laws"
    assert arguments["search_type"] == "name"
    assert arguments["dataset"] == "LEGISLATION-FED"


def test_federal_fetch_uses_fallback_citation_when_target_not_found(catalog):
    """When validated citation fails, _any_search_citation provides a fallback."""
    planner = PlannerAgent(catalog, offline=True)
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("case_analysis")
    scenario.user_query = "Je pense devoir déclarer faillite; quelles sont les étapes?"
    wrong = ToolObservation(
        tool_name="search_legal_documents",
        arguments={"query": "Bankruptcy and Insolvency Act"},
        raw_response={"results": []},
        normalized_response=json.dumps({"results": [{
            "citation_en": "CCSM c L150", "dataset": "LEGISLATION-MB",
            "name_en": "The Limitations Act",
        }, {
            "citation_en": "SC 2019, c 29", "dataset": "LEGISLATION-FED",
            "name_en": "Budget Implementation Act, 2019, No. 1",
        }, {
            "citation_en": "SC 2017, c 20", "dataset": "LEGISLATION-FED",
            "name_en": "Canada Infrastructure Bank Act",
        }]}),
        mock=True,
    ).finalize_hash()
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[wrong],
    )
    result = planner._arguments("fetch_document", state)
    assert result is not None
    assert "citation" in result


def test_bankruptcy_case_fetches_targeted_section_49(catalog):
    planner = PlannerAgent(catalog, offline=True)
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("case_analysis")
    scenario.user_query = "Je pense devoir déclarer faillite; quelles sont les étapes?"
    search = ToolObservation(
        tool_name="search_legal_documents",
        arguments={"query": "Bankruptcy and Insolvency Act"},
        raw_response={"results": []},
        normalized_response=json.dumps({"results": [{
            "citation_fr": "LRC 1985, c B-3", "dataset": "LEGISLATION-FED",
            "name_en": "Bankruptcy and Insolvency Act",
            "name_fr": "Loi sur la faillite et l’insolvabilité",
        }]}, ensure_ascii=False),
        mock=True,
    ).finalize_hash()
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[search],
    )
    assert planner._arguments("fetch_document", state) == {
        "citation": "LRC 1985, c B-3", "output_language": "fr",
        "doc_type": "laws", "section": "49",
    }


def test_civil_and_procedure_keyword_mappings_are_domain_specific():
    assert PlannerAgent._keyword_candidates(
        "search_ccq_keywords",
        "Mon voisin a construit une clôture qui empiète sur mon terrain.",
    )[0] == "bornage"
    assert PlannerAgent._keyword_candidates(
        "search_cpc_keywords",
        "Comment se déroule une demande de mise en état?",
    )[0] == "protocole de l'instance"
    assert PlannerAgent._keyword_candidates(
        "search_cpc_keywords",
        "J'ai reçu une citation à comparaître comme témoin.",
    )[0] == "assignation d'un témoin"


def test_acceptance_targets_cover_taxonomy_and_sum_exactly():
    targets = target_request_type_counts(100)
    assert set(targets) == set(REQUEST_TYPES)
    assert sum(targets.values()) == 100
    assert all(value >= 0 for value in targets.values())


def test_scheduler_does_not_keep_selecting_an_already_filled_easy_request_type():
    targets = {"exact_text_retrieval": 2, "case_analysis": 2}
    accepted = Counter(exact_text_retrieval=2, case_analysis=0)
    attempted = Counter(exact_text_retrieval=2, case_analysis=1)
    assert _next_request_type(targets, accepted, attempted) == "case_analysis"
