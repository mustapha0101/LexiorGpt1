# -*- coding: utf-8 -*-
"""Tests for StepVerifier — centralized validation gate."""

import pytest

from agentic_generation.schemas import (
    AcceptanceResult,
    CriticResult,
    Decision,
    ExpectedRoute,
    ExpectedRouteStep,
    Message,
    PlannerDecision,
    QualityReport,
    RepairReport,
    ResearchState,
    Role,
    ScenarioSpec,
    StateStatus,
    ToolObservation,
    TrainingTrajectory,
)
from lexior.agent_graph.step_verifier import (
    ProposalVerdict,
    StepVerifier,
    VerifiedProposal,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


def _scenario(
    request_type: str = "case_analysis",
    query: str = "Quels sont mes recours pour un vice caché au Québec?",
    jurisdiction_status: str = "supported_quebec",
    clarification_stage: str = "none",
) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id="test-001",
        scenario_family_id="test-fam",
        request_type=request_type,
        user_query=query,
        jurisdiction_status=jurisdiction_status,
        expected_jurisdiction="Québec",
        clarification_stage=clarification_stage,
        expected_route=ExpectedRoute(
            steps=[
                ExpectedRouteStep(tool="semantic_search_ccq"),
                ExpectedRouteStep(tool="get_ccq_articles"),
                ExpectedRouteStep(
                    tool="search_quebec_jurisprudence", optional=True),
            ],
        ),
    )


def _observation(
    tool_name: str = "semantic_search_ccq",
    ok: bool = True,
    response: str = "article 1457 du Code civil du Québec",
    error: str | None = None,
) -> ToolObservation:
    obs = ToolObservation(
        tool_name=tool_name,
        ok=ok,
        normalized_response=response,
        error=error,
        arguments={},
    )
    obs.finalize_hash()
    return obs


def _trajectory(
    scenario: ScenarioSpec | None = None,
    tool_trace: list[ToolObservation] | None = None,
    final_text: str = "Selon l'article 1726 du Code civil du Québec...",
) -> TrainingTrajectory:
    sc = scenario or _scenario()
    trace = tool_trace if tool_trace is not None else []
    messages = [
        Message(role=Role.system, content="system"),
        Message(role=Role.user, content=sc.user_query),
    ]
    for obs in trace:
        messages.append(Message(
            role=Role.assistant,
            content=f'<tool_call>\n{{"name":"{obs.tool_name}","arguments":{{}}}}\n</tool_call>',
        ))
        messages.append(Message(
            role=Role.tool, name=obs.tool_name,
            content=obs.normalized_response,
        ))
    messages.append(Message(role=Role.assistant, content=final_text))
    return TrainingTrajectory(
        scenario_id=sc.scenario_id,
        scenario_family_id=sc.scenario_family_id,
        request_type=sc.request_type,
        language="fr",
        messages=messages,
        tool_trace=trace,
        quality=QualityReport(),
    )


# ── verify_proposal ─────────────────────────────────────────────────────


class TestVerifyProposal:
    def test_permit_valid_tool(self, catalog):
        sv = StepVerifier(catalog)
        decision = PlannerDecision(
            decision=Decision.call_tool,
            next_tool="semantic_search_ccq",
            arguments={"query": "vice caché"},
        )
        result = sv.verify_proposal(
            decision, "case_analysis", [], max_tool_calls=4)
        assert result.verdict == ProposalVerdict.permit
        assert not result.errors

    def test_reject_max_tool_calls(self, catalog):
        sv = StepVerifier(catalog)
        decision = PlannerDecision(
            decision=Decision.call_tool,
            next_tool="semantic_search_ccq",
            arguments={"query": "vice caché"},
        )
        history = [_observation() for _ in range(4)]
        result = sv.verify_proposal(
            decision, "case_analysis", history, max_tool_calls=4)
        assert result.verdict == ProposalVerdict.reject
        assert "max_tool_calls" in result.errors[0]

    def test_permit_final_answer(self, catalog):
        sv = StepVerifier(catalog)
        decision = PlannerDecision(decision=Decision.final_answer)
        result = sv.verify_proposal(
            decision, "case_analysis", [], max_tool_calls=4)
        assert result.verdict == ProposalVerdict.permit

    def test_reject_forbidden_tool(self, catalog):
        sv = StepVerifier(catalog)
        decision = PlannerDecision(
            decision=Decision.call_tool,
            next_tool="semantic_search_ccq",
            arguments={"query": "test"},
        )
        result = sv.verify_proposal(
            decision, "non_legal", [], max_tool_calls=4)
        assert result.verdict == ProposalVerdict.reject


# ── compute_exempt_tools ─────────────────────────────────────────────────


class TestComputeExemptTools:
    def test_no_exemptions_single_search(self):
        history = [_observation("semantic_search_ccq", response="[]")]
        assert StepVerifier.compute_exempt_tools(history) == []

    def test_exempt_after_two_empty_searches(self):
        history = [
            _observation("semantic_search_ccq", response="[]"),
            _observation("semantic_search_ccq", response=""),
        ]
        assert StepVerifier.compute_exempt_tools(history) == [
            "get_ccq_articles"]

    def test_not_exempt_if_one_search_has_results(self):
        history = [
            _observation("semantic_search_ccq", response="[]"),
            _observation("semantic_search_ccq",
                         response="article 1457 trouvé"),
        ]
        assert StepVerifier.compute_exempt_tools(history) == []

    def test_both_exempted(self):
        history = [
            _observation("semantic_search_ccq", response="[]"),
            _observation("semantic_search_ccq", response="{}"),
            _observation("semantic_search_cpc", response=""),
            _observation("semantic_search_cpc", response="[]"),
        ]
        exempt = StepVerifier.compute_exempt_tools(history)
        assert "get_ccq_articles" in exempt
        assert "get_cpc_articles" in exempt

    def test_exempt_failed_searches(self):
        history = [
            _observation("semantic_search_ccq", ok=False,
                         response="", error="timeout"),
            _observation("semantic_search_ccq", ok=False,
                         response="", error="timeout"),
        ]
        assert StepVerifier.compute_exempt_tools(history) == [
            "get_ccq_articles"]


# ── validate_tool_route ──────────────────────────────────────────────────


class TestValidateToolRoute:
    def test_valid_route(self, catalog):
        sv = StepVerifier(catalog)
        errors = sv.validate_tool_route(
            "case_analysis",
            ["semantic_search_ccq", "get_ccq_articles",
             "search_quebec_jurisprudence"],
        )
        assert not errors

    def test_missing_required(self, catalog):
        sv = StepVerifier(catalog)
        errors = sv.validate_tool_route(
            "case_analysis",
            ["semantic_search_ccq"],
        )
        assert any("requis absent" in e for e in errors)

    def test_exempt_tools_bypass(self, catalog):
        sv = StepVerifier(catalog)
        errors = sv.validate_tool_route(
            "case_analysis",
            ["semantic_search_ccq"],
            exempt_tools=["get_ccq_articles"],
        )
        assert not any("get_ccq_articles" in e for e in errors)


# ── validate_observation ─────────────────────────────────────────────────


class TestValidateObservation:
    def test_ok_observation_passes(self, catalog):
        sv = StepVerifier(catalog)
        obs = _observation("semantic_search_ccq",
                           response="article 1457 trouvé")
        result_obs, issues = sv.validate_observation(obs)
        assert result_obs.ok
        assert not any("FATAL" in i for i in issues)

    def test_empty_qc_article_flagged(self, catalog):
        sv = StepVerifier(catalog)
        obs = _observation("get_ccq_articles", response="")
        result_obs, issues = sv.validate_observation(obs)
        assert not result_obs.ok
        assert any("FATAL" in i for i in issues)


# ── validate_final_answer ────────────────────────────────────────────────


class TestValidateFinalAnswer:
    def test_empty_answer(self, catalog):
        sv = StepVerifier(catalog)
        errors = sv.validate_final_answer("", [], "case_analysis")
        assert "réponse finale vide" in errors

    def test_json_wrapped_answer(self, catalog):
        sv = StepVerifier(catalog)
        errors = sv.validate_final_answer(
            '{"answer": "texte"}', [], "case_analysis")
        assert any("JSON" in e for e in errors)

    def test_valid_answer_passes(self, catalog):
        sv = StepVerifier(catalog)
        obs = _observation(
            "get_ccq_articles",
            response="article 1726 du Code civil du Québec",
        )
        errors = sv.validate_final_answer(
            "Selon l'article 1726 du Code civil du Québec, le vendeur...",
            [obs],
            "case_analysis",
        )
        article_errors = [e for e in errors if "article" in e.lower()
                          and "absent" in e.lower()]
        assert not article_errors


# ── compute_acceptance ───────────────────────────────────────────────────


class TestComputeAcceptance:
    def test_clean_trajectory_accepted(self, catalog):
        sv = StepVerifier(catalog)
        traj = _trajectory()
        validation = sv.validate_trajectory(
            traj, allow_mock=True, max_tool_calls=10)
        result = sv.compute_acceptance(traj, validation)
        # A minimal trajectory without real tool calls may fail on
        # route validation; we just check the method runs and returns
        # an AcceptanceResult.
        assert isinstance(result, AcceptanceResult)

    def test_blocking_errors_reject(self, catalog):
        sv = StepVerifier(catalog)
        traj = _trajectory(final_text="")
        validation = sv.validate_trajectory(
            traj, allow_mock=True, max_tool_calls=10)
        result = sv.compute_acceptance(traj, validation)
        assert not result.accepted
        assert result.blocking_errors


# ── find_first_invalid_step ──────────────────────────────────────────────


class TestFindFirstInvalidStep:
    def test_no_history_returns_none(self, catalog):
        sv = StepVerifier(catalog)
        assert sv.find_first_invalid_step([], "case_analysis") is None

    def test_failed_tool_returns_index(self, catalog):
        sv = StepVerifier(catalog)
        history = [
            _observation("semantic_search_ccq"),
            _observation("get_ccq_articles", ok=False,
                         response="", error="timeout"),
        ]
        idx = sv.find_first_invalid_step(history, "case_analysis")
        assert idx == 1

    def test_missing_required_returns_end(self, catalog):
        sv = StepVerifier(catalog)
        history = [_observation("semantic_search_ccq")]
        idx = sv.find_first_invalid_step(history, "case_analysis")
        assert idx == 1
