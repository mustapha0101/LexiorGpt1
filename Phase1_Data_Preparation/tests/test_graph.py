# -*- coding: utf-8 -*-
"""Tests du routage du graphe central (sans appels LLM).

Ancien contenu : topologie 9 nœuds (plan → execute → ... → export).
Réécrit pour la topologie centrale (initialize → ... →
compute_acceptance) — couverture équivalente, routes actuelles.
"""

import pytest

from agentic_generation.schemas import (
    AcceptanceResult,
    Role,
    ScenarioSpec,
)
from lexior.agent_graph.routing import (
    route_after_acceptance,
    route_after_classification,
    route_after_clarification,
    route_after_execute,
    route_after_failures,
    route_after_generate,
    route_after_plan,
    route_after_validate_plan,
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


# ── Après plan / validate_plan ───────────────────────────────────────────


class TestRouteAfterValidatePlan:
    def test_call_tool(self):
        state = _state(latest_decision={"decision": "call_tool"})
        assert route_after_validate_plan(state) == "execute_tool"

    def test_ask_clarification(self):
        state = _state(
            latest_decision={"decision": "ask_clarification"},
            clarification_count=0,
        )
        assert route_after_validate_plan(state) == "handle_clarification"

    def test_final_answer(self):
        state = _state(latest_decision={"decision": "final_answer"})
        assert route_after_validate_plan(state) == "build_answer_contract"

    def test_cannot_conclude(self):
        state = _state(latest_decision={"decision": "cannot_conclude"})
        assert route_after_validate_plan(state) == "build_answer_contract"

    def test_rejected_status(self):
        state = _state(
            status="rejected",
            latest_decision={"decision": "call_tool"},
        )
        assert route_after_validate_plan(state) == "reject"

    def test_unknown_decision(self):
        state = _state(latest_decision={"decision": "unknown"})
        assert route_after_validate_plan(state) == "reject"

    def test_no_decision(self):
        state = _state(latest_decision=None)
        assert route_after_validate_plan(state) == "reject"

    def test_planner_exception_routes_to_reject(self):
        state = _state(status="rejected")
        assert route_after_plan(state) == "reject"

    def test_plan_goes_to_validate(self):
        state = _state()
        assert route_after_plan(state) == "validate_plan"


# ── Après clarification ──────────────────────────────────────────────────


class TestRouteAfterClarification:
    def test_synthetic_answer_reenters_cycle(self):
        state = _state(clarification_answer="Oui, au Québec.")
        assert route_after_clarification(state) == "resolve_jurisdiction"

    def test_question_only_trajectory_goes_to_answer(self):
        state = _state(stop_reason="clarification_required")
        assert route_after_clarification(state) == "build_answer_contract"


# ── Route outil ──────────────────────────────────────────────────────────


class TestRouteAfterExecute:
    def test_normal(self):
        state = _state(status="planning")
        assert route_after_execute(state) == "verify_tool_result"

    def test_rejected(self):
        state = _state(status="rejected")
        assert route_after_execute(state) == "reject"


class TestRouteAfterClassification:
    def test_usable_records_evidence(self):
        state = _state(last_tool_result_status="usable")
        assert route_after_classification(state) == "update_research_state"

    def test_irrelevant_reformulates(self):
        state = _state(last_tool_result_status="irrelevant",
                       reformulation_count=0, max_reformulations=1)
        assert route_after_classification(state) == "reformulate_search"

    def test_empty_reformulates(self):
        state = _state(last_tool_result_status="empty",
                       reformulation_count=0, max_reformulations=1)
        assert route_after_classification(state) == "reformulate_search"

    def test_reformulation_budget_exhausted(self):
        state = _state(last_tool_result_status="empty",
                       reformulation_count=1, max_reformulations=1)
        assert route_after_classification(state) == "update_research_state"

    def test_wrong_document_type_repairs_trajectory(self):
        state = _state(last_tool_result_status="wrong_document_type",
                       repair_count=0, max_repairs=1)
        assert route_after_classification(state) == "repair_trajectory"

    def test_tool_error_falls_through_to_planner(self):
        state = _state(last_tool_result_status="tool_error")
        assert route_after_classification(state) == "update_research_state"


# ── Route réponse ────────────────────────────────────────────────────────


class TestRouteAfterGenerate:
    def test_normal(self):
        state = _state(status="answering")
        assert route_after_generate(state) == "run_critics"

    def test_rejected(self):
        state = _state(status="rejected")
        assert route_after_generate(state) == "reject"


class TestRouteAfterFailures:
    def test_no_failure_validates(self):
        state = _state(repair_from_node="validate_final")
        assert route_after_failures(state) == "validate_final"

    def test_writing_failure_repairs_answer(self):
        state = _state(repair_from_node="repair_answer")
        assert route_after_failures(state) == "repair_answer"

    def test_retrieval_failure_repairs_trajectory(self):
        state = _state(repair_from_node="repair_trajectory")
        assert route_after_failures(state) == "repair_trajectory"

    def test_jurisdiction_failure_resolves_jurisdiction(self):
        state = _state(repair_from_node="resolve_jurisdiction")
        assert route_after_failures(state) == "resolve_jurisdiction"

    def test_clarification_failure_asks(self):
        state = _state(repair_from_node="handle_clarification")
        assert route_after_failures(state) == "handle_clarification"


# ── Route finale ─────────────────────────────────────────────────────────


class TestRouteAfterAcceptance:
    def test_accepted_dataset_exports(self):
        state = _state(acceptance_result=AcceptanceResult(accepted=True))
        assert route_after_acceptance(state) == "export_dataset"

    def test_accepted_live_returns_answer(self):
        state = _state(mode="live",
                       acceptance_result=AcceptanceResult(accepted=True))
        assert route_after_acceptance(state) == "return_live_answer"

    def test_final_rejection(self):
        state = _state(
            acceptance_result=AcceptanceResult(accepted=False),
            first_invalid_step=None,
        )
        assert route_after_acceptance(state) == "reject"

    def test_repairable_rejection_repairs_trajectory(self):
        state = _state(
            acceptance_result=AcceptanceResult(accepted=False),
            first_invalid_step=1,
            repair_count=0,
            max_repairs=1,
        )
        assert route_after_acceptance(state) == "repair_trajectory"

    def test_repair_budget_exhausted_rejects(self):
        state = _state(
            acceptance_result=AcceptanceResult(accepted=False),
            first_invalid_step=1,
            repair_count=1,
            max_repairs=1,
        )
        assert route_after_acceptance(state) == "reject"

    def test_no_acceptance_rejects(self):
        state = _state()
        assert route_after_acceptance(state) == "reject"


# ── État initial ─────────────────────────────────────────────────────────


class TestInitialState:
    def test_has_system_and_user_messages(self):
        state = initial_state(_scenario(), system_prompt="sys")
        assert state["messages"][0].role == Role.system
        assert state["messages"][1].role == Role.user

    def test_mode_default(self):
        state = initial_state(_scenario(), system_prompt="sys")
        assert state["mode"] == "dataset"

    def test_live_mode(self):
        state = initial_state(
            _scenario(), mode="live", system_prompt="sys")
        assert state["mode"] == "live"

    def test_defaults(self):
        state = initial_state(_scenario(), system_prompt="sys")
        assert state["step"] == 0
        assert state["status"] == "planning"
        assert state["repair_count"] == 0
        assert state["latest_decision"] is None
        assert state["final_answer"] == ""
        assert state["jurisdiction_locked"] is False
        assert state["answer_contract"] is None
