# -*- coding: utf-8 -*-
"""Tests for the LangGraph agent graph (Phase 2).

Tests routing logic and graph compilation without requiring LLM calls.
"""

import pytest

from agentic_generation.schemas import (
    AcceptanceResult,
    CriticResult,
    Message,
    RepairReport,
    Role,
    ScenarioSpec,
    StateStatus,
)
from lexior.agent_graph.graph import (
    _route_after_critics,
    _route_after_execute,
    _route_after_final,
    _route_after_generate,
    _route_after_plan,
)
from lexior.agent_graph.state import initial_state


def _scenario(**kw):
    defaults = dict(
        scenario_id="test-001",
        scenario_family_id="test",
        request_type="case_analysis",
        user_query="Quels sont les droits du locataire?",
    )
    defaults.update(kw)
    return ScenarioSpec(**defaults)


def _state(**overrides):
    s = initial_state(_scenario(), system_prompt="test")
    s.update(overrides)
    return s


# ── Route after plan ─────────────────────────────────────────────────────


class TestRouteAfterPlan:
    def test_call_tool(self):
        state = _state(current_decision={"decision": "call_tool"})
        assert _route_after_plan(state) == "execute_tool"

    def test_ask_clarification(self):
        state = _state(
            current_decision={"decision": "ask_clarification"},
            clarification_count=0,
        )
        assert _route_after_plan(state) == "handle_clarification"

    def test_clarification_already_done(self):
        state = _state(
            current_decision={"decision": "ask_clarification"},
            clarification_count=1,
        )
        assert _route_after_plan(state) == "reject"

    def test_final_answer(self):
        state = _state(current_decision={"decision": "final_answer"})
        assert _route_after_plan(state) == "generate_answer"

    def test_cannot_conclude(self):
        state = _state(current_decision={"decision": "cannot_conclude"})
        assert _route_after_plan(state) == "generate_answer"

    def test_rejected_status(self):
        state = _state(
            status="rejected",
            current_decision={"decision": "call_tool"},
        )
        assert _route_after_plan(state) == "reject"

    def test_step_limit(self):
        state = _state(
            step=10, max_tool_calls=4,
            current_decision={"decision": "call_tool"},
        )
        assert _route_after_plan(state) == "reject"

    def test_unknown_decision(self):
        state = _state(current_decision={"decision": "unknown"})
        assert _route_after_plan(state) == "reject"

    def test_no_decision(self):
        state = _state(current_decision=None)
        assert _route_after_plan(state) == "reject"


# ── Route after execute ──────────────────────────────────────────────────


class TestRouteAfterExecute:
    def test_normal(self):
        state = _state(status="planning")
        assert _route_after_execute(state) == "plan"

    def test_rejected(self):
        state = _state(status="rejected")
        assert _route_after_execute(state) == "reject"


# ── Route after generate ─────────────────────────────────────────────────


class TestRouteAfterGenerate:
    def test_normal(self):
        state = _state(status="answering")
        assert _route_after_generate(state) == "run_critics"

    def test_rejected(self):
        state = _state(status="rejected")
        assert _route_after_generate(state) == "reject"


# ── Route after critics ──────────────────────────────────────────────────


class TestRouteAfterCritics:
    def test_no_critics(self):
        state = _state()
        assert _route_after_critics(state) == "validate_final"

    def test_both_pass(self):
        state = _state(
            legal_critic_result=CriticResult(
                critic="legal", accepted=True, score=0.9),
            agentic_critic_result=CriticResult(
                critic="agentic", accepted=True, score=0.9),
        )
        assert _route_after_critics(state) == "validate_final"

    def test_legal_fails_triggers_repair(self):
        state = _state(
            legal_critic_result=CriticResult(
                critic="legal", accepted=False, score=0.5),
            agentic_critic_result=CriticResult(
                critic="agentic", accepted=True, score=0.9),
            repair_count=0,
        )
        assert _route_after_critics(state) == "repair"

    def test_repair_exhausted(self):
        state = _state(
            legal_critic_result=CriticResult(
                critic="legal", accepted=False, score=0.5),
            repair_count=1,
        )
        assert _route_after_critics(state) == "validate_final"

    def test_precise_article_type_skips_critics(self):
        scenario = _scenario(request_type="exact_text_retrieval")
        s = initial_state(scenario, system_prompt="test")
        s["legal_critic_result"] = CriticResult(
            critic="legal", accepted=False, score=0.3)
        assert _route_after_critics(s) == "validate_final"


# ── Route after final validation ─────────────────────────────────────────


class TestRouteAfterFinal:
    def test_accepted(self):
        state = _state(
            acceptance=AcceptanceResult(accepted=True))
        assert _route_after_final(state) == "export"

    def test_rejected(self):
        state = _state(
            acceptance=AcceptanceResult(accepted=False))
        assert _route_after_final(state) == "reject"

    def test_no_acceptance(self):
        state = _state()
        assert _route_after_final(state) == "reject"


# ── Initial state ────────────────────────────────────────────────────────


class TestInitialState:
    def test_has_system_and_user_messages(self):
        state = initial_state(_scenario(), system_prompt="sys")
        assert state["messages"][0].role == Role.system
        assert state["messages"][1].role == Role.user

    def test_mode_default(self):
        state = initial_state(_scenario(), system_prompt="sys")
        assert state["mode"] == "dataset"

    def test_chat_mode(self):
        state = initial_state(
            _scenario(), mode="chat", system_prompt="sys")
        assert state["mode"] == "chat"

    def test_defaults(self):
        state = initial_state(_scenario(), system_prompt="sys")
        assert state["step"] == 0
        assert state["status"] == "planning"
        assert state["repair_count"] == 0
        assert state["current_decision"] is None
        assert state["final_answer"] == ""
