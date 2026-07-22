import pytest

from agentic_generation.agentic_critic import AgenticCritic
from agentic_generation.config import AgenticConfig
from agentic_generation.fixtures import MOCK_MCP_FIXTURES
from agentic_generation.legal_critic import LegalCritic
from agentic_generation.mcp_executor import MCPExecutor, MockMCPTransport
from agentic_generation.orchestrator import AgenticOrchestrator
from agentic_generation.acceptance import _critic_failure_reasons
from agentic_generation.planner_agent import PlannerAgent
from agentic_generation.scenario_generator import ScenarioGenerator
from agentic_generation.schemas import CriticResult, Decision, Message, ResearchState, Role
from agentic_generation.trajectory_agent import TrajectoryAgent
from agentic_generation.taxonomy import REQUEST_TYPES


def run_offline(catalog, category, max_tool_calls=4, **gen_kw):
    config = AgenticConfig(seed=3407, offline=True, dry_run=True,
                           max_tool_calls=max_tool_calls, no_critics=False)
    transport = MockMCPTransport(dict(MOCK_MCP_FIXTURES))
    executor = MCPExecutor(catalog, transport, max_retries=0)
    orchestrator = AgenticOrchestrator(
        config, catalog, PlannerAgent(catalog, offline=True), executor,
        TrajectoryAgent(offline=True), LegalCritic(offline=True), AgenticCritic(offline=True))
    scenario = ScenarioGenerator(seed=3407, offline=True).generate(category, **gen_kw)
    return orchestrator.run(scenario), transport


@pytest.mark.parametrize("category,expected", [
    ("exact_text_retrieval", ["get_ccq_articles"]),
    ("topic_research", ["semantic_search_ccq", "get_ccq_articles"]),
    ("case_analysis", ["semantic_search_ccq", "get_ccq_articles"]),
    ("law_or_regulation_identification", ["search_quebec_regulations",
                                          "get_quebec_regulation"]),
])
def test_required_routes(catalog, category, expected):
    result, transport = run_offline(catalog, category)
    assert result.accepted, result.rejection
    assert [call.name for call in transport.calls] == expected


def test_quebec_case_analysis_routes_quebec_tools(catalog):
    result, transport = run_offline(catalog, "case_analysis")
    assert result.accepted, result.rejection
    names = [call.name for call in transport.calls]
    assert "get_ccq_articles" in names


def test_incomplete_question_clarifies_before_search(catalog):
    result, transport = run_offline(
        catalog, "case_analysis",
        clarification_stage="before_search")
    assert result.accepted, result.rejection
    messages = result.trajectory.messages
    clarification_msg = [m for m in messages
                         if m.role.value == "assistant"
                         and m.content.rstrip().endswith("?")]
    assert len(clarification_msg) >= 1


def test_greeting_uses_no_tool(catalog):
    result, transport = run_offline(catalog, "non_legal")
    assert result.accepted, result.rejection
    assert transport.calls == []


def test_mcp_failure_never_produces_a_fabricated_answer(catalog):
    result, _ = run_offline(catalog, "case_analysis",
                            failure_mode="tool_error")
    assert result.accepted, result.rejection
    observation = result.trajectory.tool_trace[0]
    assert not observation.ok
    assert "ne vais pas fabriquer" in result.trajectory.final_answer()


def test_empty_result_allows_only_one_reformulation(catalog):
    result, transport = run_offline(catalog, "topic_research",
                                    failure_mode="empty_result")
    assert result.accepted, result.rejection
    assert [call.name for call in transport.calls] == ["semantic_search_ccq", "semantic_search_ccq"]
    assert transport.calls[0].arguments != transport.calls[1].arguments


def test_max_tool_calls_stops_without_loop(catalog):
    result, transport = run_offline(catalog, "topic_research", max_tool_calls=1)
    assert len(transport.calls) == 1
    assert len({(call.name, str(call.arguments)) for call in transport.calls}) == 1
    assert not result.accepted  # route incomplète, rejetée proprement


class _JsonClient:
    def __init__(self, payload):
        self.payload = payload

    def complete_json(self, *args, **kwargs):
        return dict(self.payload)


def test_planner_normalizes_tool_name_used_as_decision(catalog):
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("topic_research")
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        max_tool_calls=4,
    )
    client = _JsonClient({
        "request_type": "topic_research",
        "jurisdiction": "Québec",
        "decision": "semantic_search_ccq",
        "next_tool": "semantic_search_ccq",
        "arguments": {"query": scenario.user_query},
        "decision_trace": {"next_action": "semantic_search_ccq"},
    })
    decision = PlannerAgent(catalog, client=client).decide(state)
    assert decision.decision == Decision.call_tool
    assert decision.next_tool == "semantic_search_ccq"


def test_validate_arguments_replaces_teacher_keywords_with_full_query(catalog):
    """_validate_arguments reconstructs the semantic search query from
    the user's full question when the teacher provides a short keyword."""
    scenario = ScenarioGenerator(seed=3407, offline=True).generate("topic_research")
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content=scenario.user_query)],
        max_tool_calls=4,
    )
    client = _JsonClient({
        "thinking_text": "Je recherche dans le CCQ les articles pertinents.",
        "request_type": "topic_research",
        "jurisdiction": "Québec",
        "decision": "call_tool",
        "next_tool": "semantic_search_ccq",
        "arguments": {"query": "clôture"},
    })

    decision = PlannerAgent(catalog, client=client).decide(state)

    assert decision.arguments == {"query": scenario.user_query}
    assert decision.next_tool == "semantic_search_ccq"


def test_online_precise_article_scenario_replaces_unverified_number():
    client = _JsonClient({
        "user_query": "Donnez-moi l'article 3408 du Code civil du Québec.",
        "legal_domain": "droit civil",
        "facts_provided": {},
        "facts_missing": [],
        "clarification_answer": None,
    })
    scenario = ScenarioGenerator(client=client, seed=3407).generate("exact_text_retrieval")
    article = scenario.facts_provided["article_number"]
    assert str(article) in scenario.user_query
    assert "3408" not in scenario.user_query


def test_low_critic_score_always_has_an_explicit_reason():
    result = CriticResult(critic="legal", accepted=False, score=0.0)
    reasons = _critic_failure_reasons("legal_critic", result, 0.70)
    assert "legal_critic: décision rejected" in reasons
    assert "score 0.00" in reasons[1] and "0.70" in reasons[1]


def test_critic_requires_decision_and_score():
    with pytest.raises(Exception):
        CriticResult.model_validate({"critic": "legal", "issues": []})


def test_repeated_clarification_is_bounded(catalog):
    class RepeatingPlanner:
        def __init__(self):
            self.calls = 0

        def decide(self, state):
            from agentic_generation.schemas import PlannerDecision
            self.calls += 1
            return PlannerDecision(
                request_type=state.scenario.request_type,
                jurisdiction="Québec",
                decision=Decision.ask_clarification,
                clarification_question="Pouvez-vous préciser le problème?",
            )

    config = AgenticConfig(seed=3407, offline=True, dry_run=True,
                           max_tool_calls=4)
    planner = RepeatingPlanner()
    transport = MockMCPTransport(dict(MOCK_MCP_FIXTURES))
    orchestrator = AgenticOrchestrator(
        config, catalog, planner, MCPExecutor(catalog, transport),
        TrajectoryAgent(offline=True), LegalCritic(offline=True),
        AgenticCritic(offline=True),
    )
    scenario = ScenarioGenerator(seed=3407, offline=True).generate(
        "topic_research", clarification_stage="before_search")
    result = orchestrator.run(scenario)
    assert not result.accepted
    assert planner.calls == 2
    assert "clarification répétée" in result.rejection.reasons[0]


def test_all_taxonomy_categories_complete_offline(catalog):
    failures = {}
    for category in REQUEST_TYPES:
        result, _ = run_offline(catalog, category)
        if not result.accepted:
            failures[category] = result.rejection.reasons
    assert failures == {}
