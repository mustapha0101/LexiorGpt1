# -*- coding: utf-8 -*-
"""Regression tests for the agentic generation pipeline overhaul.

28+ tests covering: capability routing, case-law gate, training formatter,
critic profiles, source tracking, thinking export, decision enum repair,
next-action validation, hard/soft failure separation, and scenario-specific
behaviour.
"""

import json
import re

import pytest

from agentic_generation.agentic_critic import AgenticCritic
from agentic_generation.case_law_gate import (
    classify_case_result, filter_usable_cases, gate_search_results,
)
from agentic_generation.config import AgenticConfig
from agentic_generation.critic_profiles import (
    AGENTIC_RUBRICS, LEGAL_RUBRICS, REQUEST_TYPE_TO_PROFILE, get_profile,
)
from agentic_generation.fixtures import MOCK_MCP_FIXTURES
from agentic_generation.legal_critic import LegalCritic
from agentic_generation.mcp_executor import MCPExecutor, MockMCPTransport
from agentic_generation.orchestrator import AgenticOrchestrator
from agentic_generation.planner_agent import PlannerAgent
from agentic_generation.scenario_generator import ScenarioGenerator
from agentic_generation.schemas import (
    CaseRelevanceResult, CaseLawSearchStatus, ConditionalTool, CriticProfile,
    CriticResult, Decision, ExpectedRoute, Message, PlannerDecision,
    ResearchState, Role, RoutePolicy, TrainingTrajectory,
)
from agentic_generation.taxonomy import REQUEST_TYPES, NO_JURISPRUDENCE
from agentic_generation.tool_catalog import load_catalog
from agentic_generation.training_formatter import (
    build_loss_mask, export_trajectory_for_training, format_for_finetuning,
    merge_thinking_into_content,
)
from agentic_generation.trajectory_agent import TrajectoryAgent
from agentic_generation.validators import validate_next_action, validate_tool_route


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _catalog_path():
    from pathlib import Path
    return str(Path(__file__).resolve().parents[2] / "docs" / "mcp_tools_catalog.json")


@pytest.fixture
def catalog(_catalog_path):
    return load_catalog(_catalog_path)


def _run_offline(catalog, request_type, max_tool_calls=4):
    config = AgenticConfig(seed=3407, offline=True, dry_run=True,
                           max_tool_calls=max_tool_calls, no_critics=False)
    transport = MockMCPTransport(dict(MOCK_MCP_FIXTURES))
    executor = MCPExecutor(catalog, transport, max_retries=0)
    orchestrator = AgenticOrchestrator(
        config, catalog, PlannerAgent(catalog, offline=True), executor,
        TrajectoryAgent(offline=True), LegalCritic(offline=True),
        AgenticCritic(offline=True))
    scenario = ScenarioGenerator(seed=3407, offline=True).generate(request_type)
    return orchestrator.run(scenario), transport


# ---------------------------------------------------------------------------
# 1. RoutePolicy schema
# ---------------------------------------------------------------------------

class TestRoutePolicy:
    def test_route_policy_allows_tool_not_forbidden(self):
        policy = RoutePolicy(forbidden_tools=["search_legal_documents"])
        assert policy.allows_tool("get_ccq_articles")
        assert not policy.allows_tool("search_legal_documents")

    def test_route_policy_no_tool(self):
        policy = RoutePolicy(no_tool=True)
        assert policy.no_tool

    def test_route_policy_with_capabilities(self):
        policy = RoutePolicy(
            required_capabilities=["official_text_retrieval"],
            preferred_initial_tools=["get_ccq_articles"],
        )
        assert "official_text_retrieval" in policy.required_capabilities
        assert policy.allows_tool("get_ccq_articles")

    def test_conditional_tool_model(self):
        ct = ConditionalTool(tool="search_quebec_jurisprudence",
                             condition="facts warrant it")
        assert ct.tool == "search_quebec_jurisprudence"
        assert ct.condition == "facts warrant it"


# ---------------------------------------------------------------------------
# 2. Every request type has a route_policy
# ---------------------------------------------------------------------------

def test_all_request_types_have_route_policy():
    for name, rt in REQUEST_TYPES.items():
        assert hasattr(rt, "route_policy"), f"{name} missing route_policy"
        assert isinstance(rt.route_policy, RoutePolicy), f"{name} wrong type"


# ---------------------------------------------------------------------------
# 3. Capability-based validation
# ---------------------------------------------------------------------------

class TestCapabilityValidation:
    def test_forbidden_tool_rejected(self):
        errors = validate_tool_route("exact_text_retrieval",
                                     ["get_ccq_articles", "search_legal_documents"])
        assert any("interdit" in e for e in errors)

    def test_valid_route_passes(self):
        errors = validate_tool_route("exact_text_retrieval", ["get_ccq_articles"])
        assert errors == []

    def test_case_analysis_route_with_jurisprudence(self):
        errors = validate_tool_route(
            "case_analysis",
            ["semantic_search_ccq", "get_ccq_articles",
             "search_quebec_jurisprudence"])
        assert errors == []

    def test_no_tool_request_type_rejects_any_tool(self):
        errors = validate_tool_route("non_legal",
                                     ["semantic_search_ccq"])
        assert any("interdit" in e for e in errors)

    def test_missing_required_tool(self):
        errors = validate_tool_route("case_analysis",
                                     ["semantic_search_ccq"])
        assert any("requis absent" in e for e in errors)


# ---------------------------------------------------------------------------
# 4. Next-action-only validation
# ---------------------------------------------------------------------------

class TestNextActionValidation:
    def test_allowed_tool_passes(self):
        errors = validate_next_action("exact_text_retrieval", "get_ccq_articles")
        assert errors == []

    def test_forbidden_tool_fails(self):
        errors = validate_next_action("exact_text_retrieval",
                                      "search_legal_documents")
        assert any("interdit" in e for e in errors)

    def test_no_tool_request_type(self):
        errors = validate_next_action("non_legal",
                                      "semantic_search_ccq")
        assert any("interdit" in e for e in errors)


# ---------------------------------------------------------------------------
# 5. Case-law relevance gate
# ---------------------------------------------------------------------------

class TestCaseLawGate:
    def test_classify_usable_result(self):
        text = ("Dupont c. Martin, 2020 QCCS 1234. "
                "Le tribunal a appliqué l'article 1726 du Code civil.")
        result = classify_case_result(text, ["1726"], "vice caché fondation")
        assert result.correct_jurisdiction
        assert result.mentions_target_provision
        assert result.usable

    def test_classify_wrong_jurisdiction(self):
        text = "Smith v. Jones, 2020 SCC 45. Federal court decision."
        result = classify_case_result(text, ["1726"], "vice caché")
        assert not result.correct_jurisdiction
        assert not result.usable

    def test_classify_no_target_provision(self):
        text = "Dupont c. Martin, 2020 QCCS 1234. Décision sur le bail."
        result = classify_case_result(text, ["1726"], "vice caché")
        assert result.correct_jurisdiction
        assert not result.mentions_target_provision
        assert not result.usable

    def test_gate_empty_results(self):
        results, status = gate_search_results("", ["1726"], "vice caché")
        assert status == CaseLawSearchStatus.empty
        assert results == []

    def test_gate_usable_results(self):
        text = "Dupont c. Martin, 2020 QCCS 1234. Article 1726 vice caché."
        results, status = gate_search_results(text, ["1726"], "vice caché")
        assert status == CaseLawSearchStatus.usable

    def test_filter_usable_cases(self):
        usable = CaseRelevanceResult(usable=True, relevance_score=0.8)
        not_usable = CaseRelevanceResult(usable=False, relevance_score=0.3)
        filtered = filter_usable_cases([not_usable, usable])
        assert len(filtered) == 1
        assert filtered[0].usable


# ---------------------------------------------------------------------------
# 6. Training formatter — thinking merge
# ---------------------------------------------------------------------------

class TestTrainingFormatter:
    def test_merge_thinking_into_assistant_content(self):
        msg = Message(role=Role.assistant, content="Final answer.",
                      thinking="I need to check the law.")
        merged = merge_thinking_into_content(msg)
        assert "<thinking>" in merged.content
        assert "I need to check the law." in merged.content
        assert "Final answer." in merged.content
        assert merged.thinking is None

    def test_merge_thinking_preserves_non_assistant(self):
        msg = Message(role=Role.user, content="Question?")
        merged = merge_thinking_into_content(msg)
        assert merged.content == "Question?"

    def test_merge_thinking_no_thinking(self):
        msg = Message(role=Role.assistant, content="Answer.")
        merged = merge_thinking_into_content(msg)
        assert merged.content == "Answer."

    def test_loss_mask_system_user_masked(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "name": "t", "content": "r"},
            {"role": "assistant", "content": "final"},
        ]
        mask = build_loss_mask(messages)
        assert mask == [-100, -100, 1, -100, 1]


# ---------------------------------------------------------------------------
# 7. Critic profiles
# ---------------------------------------------------------------------------

class TestCriticProfiles:
    def test_all_request_types_mapped(self):
        for name in REQUEST_TYPES:
            profile = get_profile(name)
            assert isinstance(profile, CriticProfile), f"{name} has no profile"

    def test_exact_text_profile(self):
        assert get_profile("exact_text_retrieval") == CriticProfile.exact_text

    def test_factual_analysis_profile(self):
        assert get_profile("case_analysis") == CriticProfile.factual_legal_analysis
        assert get_profile("topic_research") == CriticProfile.factual_legal_analysis

    def test_comparative_and_case_law_profiles(self):
        assert get_profile("comparative_law") == CriticProfile.comparative
        assert get_profile("case_law_research") == CriticProfile.case_law_request

    def test_unknown_defaults_to_general(self):
        assert get_profile("unknown_request_type_xyz") == CriticProfile.general

    def test_all_profiles_have_legal_rubric(self):
        for profile in CriticProfile:
            assert profile in LEGAL_RUBRICS, f"{profile} missing legal rubric"

    def test_all_profiles_have_agentic_rubric(self):
        for profile in CriticProfile:
            assert profile in AGENTIC_RUBRICS, f"{profile} missing agentic rubric"


# ---------------------------------------------------------------------------
# 8. Source tracking on ResearchState
# ---------------------------------------------------------------------------

def test_research_state_source_tracking():
    from agentic_generation.schemas import ScenarioSpec
    scenario = ScenarioSpec(scenario_id="test", scenario_family_id="test",
                            request_type="exact_text_retrieval",
                            user_query="Article 1726")
    state = ResearchState(scenario=scenario)
    assert state.official_rule_retrieved is False
    assert state.official_rule_sources == []
    assert state.usable_case_sources == []
    assert state.case_law_search_status == CaseLawSearchStatus.not_required


# ---------------------------------------------------------------------------
# 9. CriticResult hard/soft failures
# ---------------------------------------------------------------------------

class TestCriticResultHardSoft:
    def test_hard_failures_field(self):
        result = CriticResult(
            critic="legal", accepted=False, score=0.3,
            hard_failures=["invented law"],
            soft_issues=["minor wording"],
        )
        assert len(result.hard_failures) == 1
        assert len(result.soft_issues) == 1

    def test_empty_hard_failures_defaults(self):
        result = CriticResult(critic="legal", accepted=True, score=0.9)
        assert result.hard_failures == []
        assert result.soft_issues == []


# ---------------------------------------------------------------------------
# 10. Decision enum — no pipe-separated values in prompt
# ---------------------------------------------------------------------------

def test_planner_prompt_no_pipe_enum(_catalog_path):
    from agentic_generation.prompts import planner_system_prompt
    cat = load_catalog(_catalog_path)
    prompt = planner_system_prompt(cat)
    assert "ask_clarification|call_tool" not in prompt


# ---------------------------------------------------------------------------
# 11. CaseRelevanceResult model
# ---------------------------------------------------------------------------

def test_case_relevance_result_model():
    result = CaseRelevanceResult(
        source_type="jurisprudence",
        correct_jurisdiction=True,
        mentions_target_provision=True,
        usable=True,
        case_name="Dupont c. Martin",
        citation="2020 QCCS 1234",
        court="QCCS",
        relevance_score=0.85,
    )
    assert result.usable
    assert result.relevance_score == 0.85


# ---------------------------------------------------------------------------
# 12. CaseLawSearchStatus enum
# ---------------------------------------------------------------------------

def test_case_law_search_status_values():
    assert CaseLawSearchStatus.usable == "usable"
    assert CaseLawSearchStatus.failed == "failed"
    assert CaseLawSearchStatus.not_required == "not_required"
    assert CaseLawSearchStatus.irrelevant == "irrelevant"
    assert CaseLawSearchStatus.empty == "empty"


# ---------------------------------------------------------------------------
# 13. Offline critic uses hard_failures for acceptance
# ---------------------------------------------------------------------------

def test_offline_legal_critic_hard_failure():
    from agentic_generation.schemas import ScenarioSpec, ToolObservation
    scenario = ScenarioSpec(scenario_id="t", scenario_family_id="t",
                            request_type="case_analysis",
                            planned_failure_mode="tool_error",
                            user_query="Article 1726")
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content="Article 1726")],
        tool_history=[
            ToolObservation(tool_name="get_ccq_articles", ok=False,
                            error="panne MCP", content_hash="x",
                            normalized_response=""),
        ],
    )
    critic = LegalCritic(offline=True)
    result = critic.evaluate(state, "Voici l'article 1726 du CCQ...")
    assert not result.accepted
    assert len(result.hard_failures) > 0


def test_offline_agentic_critic_hard_failure():
    from agentic_generation.schemas import ScenarioSpec, ToolObservation
    scenario = ScenarioSpec(
        scenario_id="t", scenario_family_id="t",
        request_type="topic_research",
        user_query="vice caché",
        expected_route=ExpectedRoute(steps=[
            {"tool": "semantic_search_ccq"},
            {"tool": "get_ccq_articles"},
        ]),
    )
    state = ResearchState(
        scenario=scenario,
        messages=[Message(role=Role.user, content="vice caché")],
        tool_history=[
            ToolObservation(tool_name="semantic_search_ccq", ok=True,
                            content_hash="x", normalized_response="Article 1726"),
        ],
        max_tool_calls=4,
    )
    critic = AgenticCritic(offline=True)
    result = critic.evaluate(state, "Réponse")
    assert not result.accepted
    assert len(result.hard_failures) > 0


# ---------------------------------------------------------------------------
# 14. Shortened training system prompt
# ---------------------------------------------------------------------------

def test_training_prompt_shorter(_catalog_path):
    from agentic_generation.prompts import agent_system_prompt
    cat = load_catalog(_catalog_path)
    prompt = agent_system_prompt(cat)
    prose_part = prompt.split("Outils")[0]
    assert len(prose_part) < 200


# ---------------------------------------------------------------------------
# 15. Prompt version updated
# ---------------------------------------------------------------------------

def test_prompt_version():
    from agentic_generation.prompts import PROMPT_VERSION
    assert "4.0" in PROMPT_VERSION


# ---------------------------------------------------------------------------
# 16-23. Offline routing for all key request types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("request_type", [
    "exact_text_retrieval",
    "topic_research",
    "case_analysis",
    "procedure_guidance",
    "law_or_regulation_identification",
])
def test_quebec_request_type_routes_offline(catalog, request_type):
    result, transport = _run_offline(catalog, request_type)
    assert result.accepted, f"{request_type}: {result.rejection}"
    tool_names = [call.name for call in transport.calls]
    policy = REQUEST_TYPES[request_type].route_policy
    for tool in tool_names:
        assert policy.allows_tool(tool), f"{tool} forbidden in {request_type}"


# ---------------------------------------------------------------------------
# 24. NO_JURISPRUDENCE request types forbid jurisprudence in route policy
# ---------------------------------------------------------------------------

def test_no_jurisprudence_request_types_forbid_in_policy():
    for name in NO_JURISPRUDENCE:
        rt = REQUEST_TYPES.get(name)
        if rt is None:
            continue
        policy = rt.route_policy
        if policy.forbidden_tools:
            assert "search_quebec_jurisprudence" in policy.forbidden_tools, (
                f"{name} should forbid jurisprudence")


# ---------------------------------------------------------------------------
# 25. Comparative law allows both Quebec and federal tools
# ---------------------------------------------------------------------------

def test_comparative_law_allows_both_jurisdictions():
    policy = REQUEST_TYPES["comparative_law"].route_policy
    assert policy.allows_tool("semantic_search_ccq"), (
        "comparative_law should allow Quebec search")
    assert policy.allows_tool("get_ccq_articles"), (
        "comparative_law should allow Quebec article retrieval")
    assert policy.allows_tool("search_legal_documents"), (
        "comparative_law should allow federal search")


# ---------------------------------------------------------------------------
# 26. Property-boundary rejection: exact_text_retrieval rejects forbidden tools
# ---------------------------------------------------------------------------

def test_exact_text_retrieval_rejects_forbidden_tools():
    errors = validate_tool_route(
        "exact_text_retrieval",
        ["get_ccq_articles", "search_legal_documents"])
    assert len(errors) > 0
    assert any("interdit" in e for e in errors)


# ---------------------------------------------------------------------------
# 27. Thinking export produces valid ChatML
# ---------------------------------------------------------------------------

def test_export_trajectory_for_training():
    trajectory = TrainingTrajectory(
        scenario_id="test-1",
        scenario_family_id="fam-1",
        request_type="exact_text_retrieval",
        messages=[
            Message(role=Role.system, content="System prompt"),
            Message(role=Role.user, content="Article 1726 du CCQ"),
            Message(role=Role.assistant, content="<tool_call>...</tool_call>",
                    thinking="Je cherche l'article 1726"),
            Message(role=Role.tool, name="get_ccq_articles",
                    content="Article 1726\nTexte officiel"),
            Message(role=Role.assistant, content="Article 1726\nTexte officiel",
                    thinking="J'ai trouvé le texte"),
        ],
    )
    export = export_trajectory_for_training(trajectory)
    assert export["scenario_id"] == "test-1"
    assert len(export["messages"]) == 5
    assert len(export["loss_mask"]) == 5
    assert export["loss_mask"][0] == -100
    assert export["loss_mask"][2] == 1
    assert "<thinking>" in export["messages"][2]["content"]


# ---------------------------------------------------------------------------
# 28. All request types still complete offline
# ---------------------------------------------------------------------------

def test_all_request_types_complete_offline(catalog):
    failures = {}
    for request_type in REQUEST_TYPES:
        result, _ = _run_offline(catalog, request_type)
        if not result.accepted:
            failures[request_type] = result.rejection.reasons
    assert failures == {}, f"Failing request types: {failures}"


# ---------------------------------------------------------------------------
# 29. All request types have valid clarification weights
# ---------------------------------------------------------------------------

def test_all_request_types_have_clarification_weights():
    for name, rt in REQUEST_TYPES.items():
        assert rt.clarification_weights, f"{name} missing clarification_weights"
        total = sum(rt.clarification_weights.values())
        assert abs(total - 1.0) < 1e-6, (
            f"{name} clarification_weights sum to {total}, expected 1.0")


# ---------------------------------------------------------------------------
# 30. CriticProfile enum covers all request types
# ---------------------------------------------------------------------------

def test_critic_profile_covers_all_request_types():
    for name in REQUEST_TYPES:
        assert name in REQUEST_TYPE_TO_PROFILE, f"{name} missing from profile mapping"
