# -*- coding: utf-8 -*-

"""
Objets structurés (Pydantic) du pipeline agentique v2.0.

Schema agentic-2.0 : enregistrements intermédiaires riches et auditables.
Ne produit PAS de format d'entraînement final (SFT/DPO/ChatML).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from . import SCHEMA_VERSION, DATASET_TYPE


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Messages (protocole Lexior / ChatML strict)
# ---------------------------------------------------------------------------

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None
    thinking: Optional[str] = None

    @field_validator("content")
    @classmethod
    def _non_null(cls, v: str) -> str:
        if v is None:
            raise ValueError("content ne peut pas être null")
        return v


# ---------------------------------------------------------------------------
# Enums v2.0
# ---------------------------------------------------------------------------

class RequestType(str, Enum):
    case_analysis = "case_analysis"
    procedure_guidance = "procedure_guidance"
    topic_research = "topic_research"
    exact_text_retrieval = "exact_text_retrieval"
    article_explanation = "article_explanation"
    case_law_research = "case_law_research"
    law_or_regulation_identification = "law_or_regulation_identification"
    legislative_status_verification = "legislative_status_verification"
    document_analysis = "document_analysis"
    comparative_law = "comparative_law"
    dataset_coverage = "dataset_coverage"
    non_legal = "non_legal"


class ClarificationStage(str, Enum):
    none = "none"
    before_search = "before_search"
    after_initial_research = "after_initial_research"


class JurisdictionStatus(str, Enum):
    supported_quebec = "supported_quebec"
    supported_federal = "supported_federal"
    supported_other_canadian = "supported_other_canadian"
    unsupported_provincial = "unsupported_provincial"
    unsupported_foreign = "unsupported_foreign"
    municipal_coverage_uncertain = "municipal_coverage_uncertain"
    undetermined = "undetermined"


class SourceIntent(str, Enum):
    legislation = "legislation"
    regulation = "regulation"
    jurisprudence = "jurisprudence"
    procedural_rule = "procedural_rule"
    official_form = "official_form"
    municipal_bylaw = "municipal_bylaw"
    legislative_metadata = "legislative_metadata"
    provided_document = "provided_document"
    none = "none"


class SourceIdentity(str, Enum):
    exactly_known = "exactly_known"
    partially_known = "partially_known"
    unknown_but_discoverable = "unknown_but_discoverable"
    not_applicable = "not_applicable"


class FailureMode(str, Enum):
    tool_error = "tool_error"
    empty_result = "empty_result"
    irrelevant_results = "irrelevant_results"
    wrong_document_type = "wrong_document_type"
    truncated_source = "truncated_source"
    malformed_result = "malformed_result"
    stale_source = "stale_source"
    missing_official_text = "missing_official_text"
    coverage_gap = "coverage_gap"
    contradictory_sources = "contradictory_sources"
    none = "none"


class SearchResultStatus(str, Enum):
    exact_match = "exact_match"
    usable = "usable"
    alternative_only = "alternative_only"
    wrong_jurisdiction = "wrong_jurisdiction"
    wrong_court_scope = "wrong_court_scope"
    irrelevant = "irrelevant"
    empty = "empty"
    wrong_document_type = "wrong_document_type"
    truncated = "truncated"
    malformed = "malformed"
    tool_error = "tool_error"
    stale = "stale"
    coverage_unavailable = "coverage_unavailable"


class ClaimType(str, Enum):
    rule = "rule"
    procedure = "procedure"
    deadline = "deadline"
    application = "application"
    limitation = "limitation"


# ---------------------------------------------------------------------------
# Scénarios
# ---------------------------------------------------------------------------

class ExpectedRouteStep(BaseModel):
    tool: str
    optional: bool = False
    condition: str = ""
    note: Optional[str] = None


class ExpectedRoute(BaseModel):
    steps: list[ExpectedRouteStep] = Field(default_factory=list)
    requires_clarification: bool = False
    no_tool: bool = False

    def required_tools(self) -> list[str]:
        return [s.tool for s in self.steps if not s.optional]

    def allowed_tools(self) -> list[str]:
        return [s.tool for s in self.steps]


class ConditionalTool(BaseModel):
    tool: str
    condition: str = ""


class RoutePolicy(BaseModel):
    required_capabilities: list[str] = Field(default_factory=list)
    conditional_tools: list[ConditionalTool] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    preferred_initial_tools: list[str] = Field(default_factory=list)
    requires_clarification: bool = False
    no_tool: bool = False
    expected_route: ExpectedRoute = Field(default_factory=ExpectedRoute)

    def allows_tool(self, tool: str) -> bool:
        return tool not in self.forbidden_tools


class ScenarioSpec(BaseModel):
    scenario_id: str
    scenario_family_id: str
    request_type: str
    language: str = "fr"
    user_query: str

    legal_domain: str = ""
    legal_system: str = ""
    jurisdiction: str = ""
    jurisdiction_status: str = "undetermined"
    expected_jurisdiction: str = ""

    source_intent: list[str] = Field(default_factory=list)
    source_identity: str = "not_applicable"

    clarification_stage: str = "none"

    facts_provided: dict[str, Any] = Field(default_factory=dict)
    facts_required_before_search: list[str] = Field(default_factory=list)
    facts_required_before_application: list[str] = Field(default_factory=list)
    facts_useful: list[str] = Field(default_factory=list)
    retrieval_targets: list[str] = Field(default_factory=list)
    # Deprecated: kept for migration compatibility
    facts_missing: list[str] = Field(default_factory=list)

    expected_source_types: list[str] = Field(default_factory=list)
    expected_route: ExpectedRoute = Field(default_factory=ExpectedRoute)

    synthetic_clarification_answer: Optional[str] = None
    # Deprecated alias
    clarification_answer: Optional[str] = None

    planned_failure_mode: Optional[str] = None
    # Deprecated alias
    failure_mode: Optional[str] = None

    @property
    def effective_clarification_answer(self) -> Optional[str]:
        return self.synthetic_clarification_answer or self.clarification_answer

    @property
    def effective_failure_mode(self) -> Optional[str]:
        return self.planned_failure_mode or self.failure_mode


# ---------------------------------------------------------------------------
# Décisions du Planner / Trajectory Agent
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    ask_clarification = "ask_clarification"
    call_tool = "call_tool"
    final_answer = "final_answer"
    cannot_conclude = "cannot_conclude"


class DecisionTrace(BaseModel):
    request_type: str = ""
    jurisdiction: str = ""
    need: str = ""
    next_action: str = ""


class PlannerDecision(BaseModel):
    request_type: str = ""
    jurisdiction: str = ""
    missing_critical_facts: list[str] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)
    decision: Decision
    next_tool: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision_trace: DecisionTrace = Field(default_factory=DecisionTrace)
    clarification_question: Optional[str] = None
    thinking_text: str = ""


# ---------------------------------------------------------------------------
# Appels et observations d'outils
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        payload = {"name": self.name, "arguments": self.arguments}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                          sort_keys=True)


class ToolObservation(BaseModel):
    tool_name: str
    server: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: str = Field(default_factory=utcnow_iso)
    raw_response: Any = None
    normalized_response: str = ""
    content_hash: str = ""
    source_urls: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    truncated: bool = False
    error: Optional[str] = None
    ok: bool = True
    mock: bool = False
    latency_ms: Optional[float] = None

    def finalize_hash(self) -> "ToolObservation":
        if not self.content_hash:
            self.content_hash = sha256_text(self.normalized_response or "")
        return self


# ---------------------------------------------------------------------------
# Classification des résultats de recherche (spec section 9)
# ---------------------------------------------------------------------------

class SearchEvaluation(BaseModel):
    tool_call_index: int = 0
    tool_name: str = ""
    result_status: str = "usable"
    result_reason: str = ""
    accepted_documents: list[dict[str, Any]] = Field(default_factory=list)
    rejected_documents: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Ancrage des affirmations (spec section 13)
# ---------------------------------------------------------------------------

class FinalClaim(BaseModel):
    claim: str
    claim_type: str = "rule"
    supporting_tool_call_ids: list[int] = Field(default_factory=list)
    supporting_source_spans: list[str] = Field(default_factory=list)
    supported: bool = True


# ---------------------------------------------------------------------------
# État de recherche (machine à états de l'orchestrateur)
# ---------------------------------------------------------------------------

class StateStatus(str, Enum):
    planning = "planning"
    waiting_tool = "waiting_tool"
    answering = "answering"
    accepted = "accepted"
    rejected = "rejected"


class CaseLawSearchStatus(str, Enum):
    usable = "usable"
    empty = "empty"
    irrelevant = "irrelevant"
    failed = "failed"
    not_required = "not_required"


class CaseRelevanceResult(BaseModel):
    source_type: str = "unknown"
    correct_jurisdiction: bool = False
    mentions_target_provision: bool = False
    matches_legal_issue: bool = False
    matches_material_facts: bool = False
    usable: bool = False
    reason: str = ""
    case_name: str = ""
    citation: str = ""
    court: str = ""
    date: str = ""
    target_provisions: list[str] = Field(default_factory=list)
    legal_issue: str = ""
    material_facts: str = ""
    holding_or_principle: str = ""
    source_url: str = ""
    relevance_score: float = 0.0


class ResearchState(BaseModel):
    scenario: ScenarioSpec
    messages: list[Message] = Field(default_factory=list)
    tool_history: list[ToolObservation] = Field(default_factory=list)
    search_evaluations: list[SearchEvaluation] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    step: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(default=4, ge=0)
    jurisdiction_status: str = "unknown"
    missing_critical_facts: list[str] = Field(default_factory=list)
    status: StateStatus = StateStatus.planning
    stop_reason: Optional[str] = None
    official_rule_retrieved: bool = False
    official_rule_sources: list[str] = Field(default_factory=list)
    usable_case_sources: list[CaseRelevanceResult] = Field(default_factory=list)
    case_law_search_status: str = "not_required"
    reformulation_count: int = 0

    def tool_calls_made(self) -> int:
        return len(self.tool_history)


# ---------------------------------------------------------------------------
# Critiques — scores multi-dimensionnels (spec section 21)
# ---------------------------------------------------------------------------

class CriticProfile(str, Enum):
    exact_text = "exact_text"
    legal_explanation = "legal_explanation"
    factual_legal_analysis = "factual_legal_analysis"
    case_law_request = "case_law_request"
    regulation_lookup = "regulation_lookup"
    procedure = "procedure"
    clarification = "clarification"
    non_legal = "non_legal"
    failure_mode = "failure_mode"
    federal = "federal"
    comparative = "comparative"
    general = "general"


CRITIC_LABELS = [
    "unnecessary_clarification",
    "missing_clarification",
    "retrieval_target_mislabeled_as_fact",
    "wrong_legal_domain",
    "wrong_jurisdiction",
    "wrong_tool",
    "mechanical_route_following",
    "bad_query",
    "duplicate_reformulation",
    "wrong_document_type_accepted",
    "unsupported_claim",
    "unsupported_deadline",
    "unretrieved_article_used",
    "fabricated_case_law_pattern",
    "stale_source_presented_as_current",
    "coverage_limitation_ignored",
    "thinking_too_long",
    "register_informal",
    "final_answer_does_not_answer",
]


class MultiDimensionalScore(BaseModel):
    request_classification_score: float = Field(default=0.0, ge=0.0, le=1.0)
    jurisdiction_score: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_selection_score: float = Field(default=0.0, ge=0.0, le=1.0)
    search_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    result_validation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding_score: float = Field(default=0.0, ge=0.0, le=1.0)
    legal_accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0)
    answer_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    labels: list[str] = Field(default_factory=list)

    @property
    def aggregate_score(self) -> float:
        scores = [
            self.request_classification_score, self.jurisdiction_score,
            self.clarification_score, self.tool_selection_score,
            self.search_quality_score, self.result_validation_score,
            self.grounding_score, self.legal_accuracy_score,
            self.answer_quality_score,
        ]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def accepted(self) -> bool:
        return self.aggregate_score >= 0.70 and not any(
            label in self.labels for label in (
                "unsupported_claim", "fabricated_case_law_pattern",
                "unsupported_deadline", "wrong_jurisdiction",
            )
        )


class CriticResult(BaseModel):
    critic: str
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)
    soft_issues: list[str] = Field(default_factory=list)
    critic_profile: Optional[str] = None
    dimensional_scores: Optional[MultiDimensionalScore] = None


# ---------------------------------------------------------------------------
# Qualité et métadonnées de génération
# ---------------------------------------------------------------------------

class GenerationMetadata(BaseModel):
    teacher_model: str = ""
    teacher_base_url_hash: str = ""
    critic_model: str = ""
    seed: int = 3407
    generated_at: str = Field(default_factory=utcnow_iso)
    prompt_version: str = ""
    tool_catalog_hash: str = ""


class RepairReport(BaseModel):
    attempted: bool = False
    status: str = "not_needed"  # not_needed | successful | failed | not_repairable
    reason: str = ""
    first_invalid_step: Optional[int] = None
    changes: list[str] = Field(default_factory=list)


class AcceptanceResult(BaseModel):
    accepted: bool = False
    reasons: list[str] = Field(default_factory=list)
    blocking_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failed_checks: dict[str, Any] = Field(default_factory=dict)


class RejectionDetail(BaseModel):
    scenario_id: str = ""
    first_invalid_step: Optional[int] = None
    blocking_reason: str = ""
    repair_attempted: bool = False
    repair_successful: bool = False


class QualityReport(BaseModel):
    deterministic_validation: bool = False
    legal_critic_score: Optional[float] = None
    agentic_critic_score: Optional[float] = None
    legal_critic: Optional[dict[str, Any]] = None
    agentic_critic: Optional[dict[str, Any]] = None
    grounding_critic: Optional[dict[str, Any]] = None
    clarification_critic: Optional[dict[str, Any]] = None
    accepted_for_intermediate: bool = False
    repair: RepairReport = Field(default_factory=RepairReport)
    acceptance: Optional[AcceptanceResult] = None
    rejection_detail: Optional[RejectionDetail] = None
    # deprecated: use repair.status and repair.attempted instead
    repair_status: str = "none"
    repaired: bool = False


class GroundingEntry(BaseModel):
    tool_name: str
    content_hash: str = ""
    source_urls: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Enregistrement intermédiaire v2.0 (ligne JSONL maître)
# ---------------------------------------------------------------------------

class IntermediateRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    dataset_type: str = DATASET_TYPE

    scenario: ScenarioSpec

    messages: list[Message] = Field(default_factory=list)
    tool_trace: list[ToolObservation] = Field(default_factory=list)
    search_evaluations: list[SearchEvaluation] = Field(default_factory=list)
    grounding: list[GroundingEntry] = Field(default_factory=list)
    final_claims: list[FinalClaim] = Field(default_factory=list)

    quality: QualityReport = Field(default_factory=QualityReport)
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)

    def final_answer(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == Role.assistant:
                return msg.content
        return ""

    def group_key(self) -> str:
        cites = sorted({c for g in self.grounding for c in g.citations})[:3]
        return f"fam:{self.scenario.scenario_family_id}|src:{'|'.join(cites)}"

    def to_jsonl(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# TrainingTrajectory — DEPRECATED, kept for migration and training_formatter
# ---------------------------------------------------------------------------

class TrainingTrajectory(BaseModel):
    schema_version: str = SCHEMA_VERSION
    dataset_type: str = DATASET_TYPE
    scenario_id: str
    scenario_family_id: str
    language: str = "fr"
    request_type: str = ""
    legal_domain: str = ""
    expected_jurisdiction: str = ""
    resolved_jurisdiction: str = ""
    messages: list[Message] = Field(default_factory=list)
    tool_trace: list[ToolObservation] = Field(default_factory=list)
    grounding: list[GroundingEntry] = Field(default_factory=list)
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
    quality: QualityReport = Field(default_factory=QualityReport)

    def group_key(self) -> str:
        cites = sorted({c for g in self.grounding for c in g.citations})[:3]
        return f"fam:{self.scenario_family_id}|src:{'|'.join(cites)}"

    def final_answer(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == Role.assistant:
                return msg.content
        return ""

    def to_jsonl(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Rejets et manifeste
# ---------------------------------------------------------------------------

class RejectionRecord(BaseModel):
    scenario_id: str
    request_type: str = ""
    stage: str = ""
    reasons: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=utcnow_iso)
    trajectory: Optional[dict[str, Any]] = None
    failure_mode: Optional[str] = None
    critic_labels: list[str] = Field(default_factory=list)


class PreferencePair(BaseModel):
    scenario_id: str
    chosen_trajectory: dict[str, Any] = Field(default_factory=dict)
    rejected_trajectory: dict[str, Any] = Field(default_factory=dict)
    preference_reasons: list[str] = Field(default_factory=list)


class GenerationManifest(BaseModel):
    run_id: str
    created_at: str = Field(default_factory=utcnow_iso)
    schema_version: str = SCHEMA_VERSION
    seed: int = 3407
    prompt_version: str = ""
    tool_catalog_hash: str = ""
    teacher_model: str = ""
    teacher_base_url_hash: str = ""
    critic_model: str = ""
    target_accepted: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    accepted_by_category: dict[str, int] = Field(default_factory=dict)
    category_targets: dict[str, int] = Field(default_factory=dict)
    costs: dict[str, Any] = Field(default_factory=dict)
    taxonomy_proportions: dict[str, float] = Field(default_factory=dict)
    mix: dict[str, Any] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    distribution_achieved: dict[str, Any] = Field(default_factory=dict)
