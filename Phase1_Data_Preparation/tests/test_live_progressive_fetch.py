from lexior.agentic.planner_agent import PlannerAgent
from lexior.agentic.schemas import (
    Decision,
    Message,
    PlannerDecision,
    ResearchState,
    Role,
    ScenarioSpec,
    ToolObservation,
)
from lexior.services.provenance import numeros_demandes
from lexior.services.result_verification import ResultVerificationService
from lexior.agent_graph.nodes.handle_clarification import _clarification_category
from lexior.agent_graph.state import (
    canonical_case_description,
    initial_state,
    to_research_state,
    visible_tool_history,
)


def _state(numbers):
    scenario = ScenarioSpec(
        scenario_id="live-progressive",
        scenario_family_id="live",
        request_type="case_analysis",
        user_query="Question factuelle",
    )
    search = ToolObservation(
        tool_name="semantic_search_ccq",
        arguments={"query": scenario.user_query},
        normalized_response="\n".join(
            f"Article {number} — candidat" for number in numbers),
        ok=True,
    ).finalize_hash()
    return ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[search],
        current_turn_tool_count=1,
    )


def test_live_fetches_ranked_articles_in_progressive_batches(catalog):
    planner = PlannerAgent(
        catalog, chat_mode=True, initial_article_fetch_k=6,
        article_fetch_batch_size=6, max_articles_per_issue=20)
    state = _state(range(1, 21))

    assert planner._article_fetch_arguments(state, "get_ccq_articles") == {
        "articles": [1, 2, 3, 4, 5, 6]}

    first = ToolObservation(
        tool_name="get_ccq_articles", arguments={"articles": [1, 2, 3, 4, 5, 6]},
        normalized_response="Article 1\nTexte", ok=True,
    ).finalize_hash()
    widened = state.model_copy(update={"tool_history": [*state.tool_history, first]})
    assert planner._article_fetch_arguments(widened, "get_ccq_articles") == {
        "articles": [7, 8, 9, 10, 11, 12]}


def test_article_range_provenance_contains_every_requested_article():
    assert numeros_demandes("get_ccq_articles", {
        "start_article": 1457, "end_article": 1460,
    }) == ("1457", "1458", "1459", "1460")


def test_live_dossier_keeps_official_evidence_without_spending_new_turn_budget():
    scenario = ScenarioSpec(
        scenario_id="live-dossier", scenario_family_id="live",
        request_type="case_analysis", user_query="L'arbre est tombé sur mon garage.",
    )
    state = initial_state(scenario, mode="live", system_prompt="test")
    official = ToolObservation(
        tool_name="get_ccq_articles", arguments={"articles": [1457]},
        normalized_response="Article 1457\nTexte officiel.", ok=True,
    ).finalize_hash()
    current_search = ToolObservation(
        tool_name="semantic_search_ccq", arguments={"query": "arbre"},
        normalized_response="Article 985 — candidat", ok=True,
    ).finalize_hash()
    state.update({
        "active_issue": scenario.user_query,
        "case_context": {"facts": {"user_statements": [
            "Le voisin savait que l'arbre était dangereux."]}},
        "prior_evidence": [official],
        "tool_history": [current_search],
    })

    assert [item.tool_name for item in visible_tool_history(state)] == [
        "get_ccq_articles", "semantic_search_ccq"]
    assert "voisin savait" in canonical_case_description(state)
    assert to_research_state(state).tool_calls_made() == 1


def test_clarification_categories_keep_jurisdiction_and_facts_distinct():
    assert _clarification_category("Dans quelle province êtes-vous?", []) == "jurisdiction"
    assert _clarification_category("Quel était l'état du bien?", []) == "fact"


def test_case_law_search_is_candidate_until_full_decision_is_fetched(catalog):
    scenario = ScenarioSpec(
        scenario_id="case-law", scenario_family_id="live",
        request_type="case_analysis", user_query="Faits du dossier",
    )
    search = ToolObservation(
        tool_name="search_quebec_jurisprudence", arguments={"query": "article 1457"},
        normalized_response="2022 QCCQ 1\nArticle 1457\nDécision candidate.",
        source_urls=["https://canlii.example/decision"], ok=True,
    ).finalize_hash()
    assessment = ResultVerificationService().assess(search, user_query="Faits")
    assert assessment.evidence_level == "candidate"
    assert not assessment.citable and not assessment.usable_as_evidence

    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        tool_history=[search],
        article_reviews={"1457": {"status": "applicable"}},
        case_description="Faits du dossier",
        current_turn_tool_count=1,
    )
    final = PlannerDecision(
        request_type="case_analysis", decision=Decision.final_answer,
    )
    planned = PlannerAgent(catalog, chat_mode=True)._guard_live_source_completeness(
        state, final)
    assert planned.next_tool == "get_quebec_regulation"
    assert planned.arguments == {"url": "https://canlii.example/decision"}
