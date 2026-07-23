# -*- coding: utf-8 -*-
"""LexiorState — l'état unique et autoritaire du système central.

Une seule source de vérité pour les deux modes (dataset et live).
Aucune valeur concurrente dans des variables locales ou des objets
séparés : les nœuds lisent l'état, appellent un service, écrivent une
mise à jour partielle.

Conventions :
  - ``total=False`` — chaque nœud retourne un dict partiel;
  - listes reconstruites par les nœuds (canaux last-write-wins,
    volontairement : chaque tour live ré-injecte l'état complet);
  - les convertisseurs ``to_research_state`` / ``to_trajectory``
    projettent l'état vers les schémas historiques consommés par les
    services (une projection, pas une seconde vérité).
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict

from lexior.agentic.schemas import (
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
    """État du graphe central Lexior (les deux modes)."""

    # ── Contrôle ─────────────────────────────────────────────────────────
    mode: str                      # "dataset" | "live"
    thread_id: str
    status: str                    # StateStatus + "clarification"
    stop_reason: str

    # ── Entrée ───────────────────────────────────────────────────────────
    scenario: ScenarioSpec         # dataset : réel; live : synthétisé
    messages: list[Message]
    latest_user_message: str

    # ── Classification de la demande / suivi conversationnel ─────────────
    request_type: str
    active_issue: str
    current_user_goal: str
    latest_user_intent: str
    refers_to_previous_answer: bool
    already_answered: str
    still_needed: str
    requested_output_type: str     # "answer" | "article" | "source_url" | ...
    active_case_or_document: str

    # ── Juridiction (une seule vérité, verrouillable) ────────────────────
    work_location: str
    employer: str
    employer_legal_entity: str
    candidate_jurisdiction: str
    resolved_jurisdiction: str
    jurisdiction_status: str
    jurisdiction_verified: bool
    jurisdiction_basis: str
    jurisdiction_locked: bool

    # ── Faits ────────────────────────────────────────────────────────────
    facts: dict[str, Any]
    missing_facts_before_search: list[str]
    missing_facts_before_application: list[str]
    missing_critical_facts: list[str]

    # ── Planification et exécution d'outils ──────────────────────────────
    latest_decision: Optional[dict]
    planner_feedback: str          # correctif transmis au prochain plan
    step: int
    max_tool_calls: int
    tool_history: list[ToolObservation]
    sources: list[str]
    usable_evidence: list[int]     # index dans tool_history (legacy compat)
    search_evaluations: list[SearchEvaluation]

    last_tool_call: Optional[dict]
    last_tool_result_status: str   # SearchResultStatus
    last_tool_assessment: Optional[dict]
    reformulation_count: int
    max_reformulations: int

    official_rule_retrieved: bool
    official_rule_sources: list[str]
    usable_case_sources: list[CaseRelevanceResult]
    case_law_search_status: str
    exempt_tools: list[str]

    # ── Evidence collections (three-tier) ────────────────────────────────
    candidate_sources: list[dict]    # EvidenceEntry dicts
    citable_sources: list[dict]      # EvidenceEntry dicts
    usable_evidence_entries: list[dict]  # EvidenceEntry dicts
    alternative_sources: list[dict]  # EvidenceEntry dicts
    invalidated_sources: list[dict]  # EvidenceEntry dicts
    coverage_gaps: list[dict]        # CoverageGap dicts
    acceptance_blockers: list[str]   # AcceptanceBlocker values

    # ── Jurisdiction dimensions (separate from the single resolved value)
    requested_source_jurisdiction: str
    requested_court_scope: str
    returned_source_jurisdiction: str
    returned_court_scope: str
    substantive_law: str

    # ── Clarification ────────────────────────────────────────────────────
    pending_clarification: str
    clarification_answer: str
    clarification_count: int

    # ── Rédaction ────────────────────────────────────────────────────────
    answer_contract: Optional[dict]
    final_answer: str
    final_reasoning_summary: str

    # ── Qualité ──────────────────────────────────────────────────────────
    critic_results: dict[str, Optional[CriticResult]]  # {"legal","agentic"}
    failure_reports: list[dict]
    validation_issues: list[str]
    deterministic_blockers: list[str]
    deterministic_validation: bool
    validation_result: Optional[Any]
    acceptance_result: AcceptanceResult
    first_invalid_step: Optional[int]
    rejection_detail: Optional[RejectionDetail]

    # ── Réparation ───────────────────────────────────────────────────────
    repair: RepairReport
    repair_from_node: str
    repair_count: int
    max_repairs: int
    repair_history: list[dict]

    # ── Sortie ───────────────────────────────────────────────────────────
    grounding: list[GroundingEntry]
    generation_metadata: GenerationMetadata
    trajectory: Optional[dict]
    export_result: Optional[dict]


# ── Fabrique ─────────────────────────────────────────────────────────────


def initial_state(
    scenario: ScenarioSpec,
    *,
    mode: str = "dataset",
    max_tool_calls: int = 4,
    system_prompt: str = "",
    thread_id: str = "",
    max_reformulations: int = 1,
    max_repairs: int = 1,
) -> dict[str, Any]:
    """État initial complet d'un run (toutes les clés, valeurs neutres).

    Fournir TOUTES les clés est un invariant : en mode live, un thread
    persistant est ré-invoqué à chaque tour et l'état complet écrase
    proprement le tour précédent (canaux last-write-wins).
    """
    messages: list[Message] = [
        Message(role=Role.system, content=system_prompt),
    ]
    if scenario.user_query:
        messages.append(Message(role=Role.user, content=scenario.user_query))
    return {
        "mode": mode,
        "thread_id": thread_id,
        "status": StateStatus.planning.value,
        "stop_reason": "",
        "scenario": scenario,
        "messages": messages,
        "latest_user_message": scenario.user_query or "",
        "request_type": scenario.request_type,
        "active_issue": "",
        "current_user_goal": "",
        "latest_user_intent": "",
        "refers_to_previous_answer": False,
        "already_answered": "",
        "still_needed": "",
        "requested_output_type": "answer",
        "active_case_or_document": "",
        "work_location": "",
        "employer": "",
        "employer_legal_entity": "",
        "candidate_jurisdiction": "",
        "resolved_jurisdiction": "",
        "jurisdiction_status": "unknown",
        "jurisdiction_verified": False,
        "jurisdiction_basis": "",
        "jurisdiction_locked": False,
        "facts": dict(scenario.facts_provided),
        "missing_facts_before_search": list(
            scenario.facts_required_before_search),
        "missing_facts_before_application": list(
            scenario.facts_required_before_application),
        "missing_critical_facts": list(scenario.facts_missing),
        "latest_decision": None,
        "planner_feedback": "",
        "step": 0,
        "max_tool_calls": max_tool_calls,
        "tool_history": [],
        "sources": [],
        "usable_evidence": [],
        "search_evaluations": [],
        "last_tool_call": None,
        "last_tool_result_status": "",
        "last_tool_assessment": None,
        "reformulation_count": 0,
        "max_reformulations": max_reformulations,
        "official_rule_retrieved": False,
        "official_rule_sources": [],
        "usable_case_sources": [],
        "case_law_search_status": "not_required",
        "exempt_tools": [],
        "pending_clarification": "",
        "clarification_answer": "",
        "clarification_count": 0,
        "answer_contract": None,
        "final_answer": "",
        "final_reasoning_summary": "",
        "critic_results": {},
        "failure_reports": [],
        "validation_issues": [],
        "deterministic_blockers": [],
        "deterministic_validation": False,
        "validation_result": None,
        "acceptance_result": AcceptanceResult(),
        "first_invalid_step": None,
        "rejection_detail": None,
        "repair": RepairReport(),
        "repair_from_node": "",
        "repair_count": 0,
        "max_repairs": max_repairs,
        "repair_history": [],
        "candidate_sources": [],
        "citable_sources": [],
        "usable_evidence_entries": [],
        "alternative_sources": [],
        "invalidated_sources": [],
        "coverage_gaps": [],
        "acceptance_blockers": [],
        "requested_source_jurisdiction": "",
        "requested_court_scope": "",
        "returned_source_jurisdiction": "",
        "returned_court_scope": "",
        "substantive_law": "",
        "grounding": [],
        "generation_metadata": GenerationMetadata(),
        "trajectory": None,
        "export_result": None,
    }


# ── Projections vers les schémas historiques ─────────────────────────────


def to_research_state(state: LexiorState) -> ResearchState:
    """Projette l'état du graphe vers le ``ResearchState`` des services."""
    scenario = state["scenario"]
    status = state.get("status", "planning")
    if status not in {s.value for s in StateStatus}:
        status = StateStatus.planning.value
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
        status=StateStatus(status),
        stop_reason=state.get("stop_reason") or None,
        official_rule_retrieved=state.get("official_rule_retrieved", False),
        official_rule_sources=state.get("official_rule_sources", []),
        usable_case_sources=state.get("usable_case_sources", []),
        case_law_search_status=state.get(
            "case_law_search_status", "not_required"),
        reformulation_count=state.get("reformulation_count", 0),
    )


def to_quality_report(state: LexiorState) -> QualityReport:
    """Projette les champs qualité vers le ``QualityReport`` historique."""
    repair = state.get("repair", RepairReport())
    acceptance = state.get("acceptance_result", AcceptanceResult())
    critics = state.get("critic_results", {}) or {}
    legal = critics.get("legal")
    agentic = critics.get("agentic")
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
    """Projette l'état du graphe vers la trajectoire d'entraînement."""
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
