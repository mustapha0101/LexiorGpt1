# -*- coding: utf-8 -*-
"""LexiorState — single authoritative graph state for the Lexior agent.

Consolidates execution state (ResearchState), scenario metadata (ScenarioSpec),
and quality/acceptance outcomes (QualityReport) into one TypedDict.

Eliminates duplicated truth in QualityReport:
  - ``repair`` (RepairReport) is the single source for repair status.
    Replaces QualityReport.repaired + QualityReport.repair_status.
  - ``acceptance`` (AcceptanceResult) is the single source for accept/reject.
    Replaces QualityReport.accepted_for_intermediate.

Phase 2 adds LangGraph reducer annotations to list fields.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from agentic_generation.schemas import (
    AcceptanceResult,
    CaseRelevanceResult,
    CriticResult,
    GenerationMetadata,
    GroundingEntry,
    Message,
    QualityReport,
    RejectionDetail,
    RepairReport,
    ResearchState,
    Role,
    ScenarioSpec,
    SearchEvaluation,
    StateStatus,
    ToolObservation,
    TrainingTrajectory,
)


class LexiorState(TypedDict, total=False):
    """Graph state for the Lexior multi-agent legal assistant.

    ``total=False`` — nodes return partial updates; absent fields keep
    their current value.

    Reducer annotations (Phase 2, applied via ``Annotated``)::

        messages       → add_messages   (LangGraph message reducer)
        tool_history   → operator.add   (append-only)
        sources        → operator.add   (append-only)

    All other fields use last-write-wins (LangGraph default).
    """

    # ── Control ──────────────────────────────────────────────────────────
    mode: str  # "dataset" | "chat"

    # ── Scenario (immutable after graph entry) ───────────────────────────
    scenario: ScenarioSpec

    # ── Execution (mutated by graph nodes) ───────────────────────────────
    messages: list[Message]
    tool_history: list[ToolObservation]
    search_evaluations: list[SearchEvaluation]
    sources: list[str]
    step: int
    max_tool_calls: int
    status: str  # StateStatus value
    stop_reason: str
    missing_critical_facts: list[str]
    official_rule_retrieved: bool
    official_rule_sources: list[str]
    usable_case_sources: list[CaseRelevanceResult]
    case_law_search_status: str
    reformulation_count: int
    resolved_jurisdiction: str
    clarification_count: int
    exempt_tools: list[str]

    # ── Quality (single source of truth — no duplication) ────────────────
    deterministic_validation: bool
    legal_critic_result: Optional[CriticResult]
    agentic_critic_result: Optional[CriticResult]
    repair: RepairReport
    acceptance: AcceptanceResult
    first_invalid_step: Optional[int]
    rejection_detail: Optional[RejectionDetail]

    # ── Graph control (Phase 2) ──────────────────────────────────────────
    current_decision: Optional[dict]
    repair_count: int

    # ── Output ───────────────────────────────────────────────────────────
    grounding: list[GroundingEntry]
    generation_metadata: GenerationMetadata
    final_answer: str
    final_thinking: str


# ── Factory ──────────────────────────────────────────────────────────────


def initial_state(
    scenario: ScenarioSpec,
    *,
    mode: str = "dataset",
    max_tool_calls: int = 4,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Create the initial state dict for a graph run."""
    messages: list[Message] = [
        Message(role=Role.system, content=system_prompt),
    ]
    if scenario.user_query:
        messages.append(Message(role=Role.user, content=scenario.user_query))
    return {
        "mode": mode,
        "scenario": scenario,
        "messages": messages,
        "tool_history": [],
        "search_evaluations": [],
        "sources": [],
        "step": 0,
        "max_tool_calls": max_tool_calls,
        "status": StateStatus.planning.value,
        "stop_reason": "",
        "missing_critical_facts": list(scenario.facts_missing),
        "official_rule_retrieved": False,
        "official_rule_sources": [],
        "usable_case_sources": [],
        "case_law_search_status": "not_required",
        "reformulation_count": 0,
        "resolved_jurisdiction": "",
        "clarification_count": 0,
        "exempt_tools": [],
        "deterministic_validation": False,
        "legal_critic_result": None,
        "agentic_critic_result": None,
        "repair": RepairReport(),
        "acceptance": AcceptanceResult(),
        "current_decision": None,
        "repair_count": 0,
        "first_invalid_step": None,
        "rejection_detail": None,
        "grounding": [],
        "generation_metadata": GenerationMetadata(),
        "final_answer": "",
        "final_thinking": "",
    }


# ── Converters (backward compatibility during migration) ─────────────────


def to_research_state(state: LexiorState) -> ResearchState:
    """Convert a LexiorState dict to a legacy ResearchState."""
    scenario = state["scenario"]
    return ResearchState(
        scenario=scenario,
        messages=state.get("messages", []),
        tool_history=state.get("tool_history", []),
        search_evaluations=state.get("search_evaluations", []),
        sources=state.get("sources", []),
        step=state.get("step", 0),
        max_tool_calls=state.get("max_tool_calls", 4),
        jurisdiction_status=state.get("resolved_jurisdiction") or "unknown",
        missing_critical_facts=state.get("missing_critical_facts", []),
        status=StateStatus(state.get("status", "planning")),
        stop_reason=state.get("stop_reason") or None,
        official_rule_retrieved=state.get("official_rule_retrieved", False),
        official_rule_sources=state.get("official_rule_sources", []),
        usable_case_sources=state.get("usable_case_sources", []),
        case_law_search_status=state.get(
            "case_law_search_status", "not_required"),
        reformulation_count=state.get("reformulation_count", 0),
    )


def to_quality_report(state: LexiorState) -> QualityReport:
    """Convert quality fields to a legacy QualityReport.

    Derives the deprecated ``repaired``, ``repair_status``, and
    ``accepted_for_intermediate`` fields from the single sources
    ``repair`` and ``acceptance``.
    """
    repair = state.get("repair", RepairReport())
    acceptance = state.get("acceptance", AcceptanceResult())
    legal = state.get("legal_critic_result")
    agentic = state.get("agentic_critic_result")
    return QualityReport(
        deterministic_validation=state.get("deterministic_validation", False),
        legal_critic_score=legal.score if legal else None,
        agentic_critic_score=agentic.score if agentic else None,
        repair=repair,
        acceptance=acceptance,
        accepted_for_intermediate=acceptance.accepted,
        repaired=repair.attempted and repair.status == "successful",
        repair_status=repair.status,
        rejection_detail=state.get("rejection_detail"),
    )


def to_trajectory(state: LexiorState) -> TrainingTrajectory:
    """Convert a LexiorState dict to a legacy TrainingTrajectory."""
    scenario = state["scenario"]
    return TrainingTrajectory(
        scenario_id=scenario.scenario_id,
        scenario_family_id=scenario.scenario_family_id,
        language=scenario.language,
        request_type=scenario.request_type,
        legal_domain=scenario.legal_domain,
        expected_jurisdiction=scenario.expected_jurisdiction,
        resolved_jurisdiction=state.get("resolved_jurisdiction", ""),
        messages=state.get("messages", []),
        tool_trace=state.get("tool_history", []),
        grounding=state.get("grounding", []),
        generation_metadata=state.get(
            "generation_metadata", GenerationMetadata()),
        quality=to_quality_report(state),
    )
