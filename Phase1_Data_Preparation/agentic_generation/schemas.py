# -*- coding: utf-8 -*-

"""
Objets structurés (Pydantic) du pipeline agentique.

Tout ce qui circule entre les agents, l'exécuteur MCP, les critiques et le
stockage passe par ces modèles : une trajectoire qui ne valide pas ici n'existe
pas pour le reste du pipeline.
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
    # Nom de l'outil, uniquement pour role=tool.
    name: Optional[str] = None
    # Raisonnement interne de l'assistant (séparé du contenu visible).
    # Le formatter fusionne ce champ dans <thinking>...</thinking> + content.
    thinking: Optional[str] = None

    @field_validator("content")
    @classmethod
    def _non_null(cls, v: str) -> str:
        if v is None:
            raise ValueError("content ne peut pas être null")
        return v


# ---------------------------------------------------------------------------
# Scénarios
# ---------------------------------------------------------------------------

class ExpectedRouteStep(BaseModel):
    tool: str
    # Étape conditionnelle : sa présence n'est pas exigée, mais si elle
    # apparaît elle doit être à cette position relative.
    optional: bool = False
    note: Optional[str] = None


class ExpectedRoute(BaseModel):
    steps: list[ExpectedRouteStep] = Field(default_factory=list)
    # La bonne politique commence par une demande de clarification.
    requires_clarification: bool = False
    # Aucun outil ne doit être appelé (salutation, question non juridique).
    no_tool: bool = False

    def required_tools(self) -> list[str]:
        return [s.tool for s in self.steps if not s.optional]

    def allowed_tools(self) -> list[str]:
        return [s.tool for s in self.steps]


class ScenarioSpec(BaseModel):
    scenario_id: str
    scenario_family_id: str
    request_type: str
    language: str = "fr"
    user_query: str
    legal_domain: str = ""
    # Métadonnées CACHÉES : jamais montrées au Planner avant sa décision.
    expected_jurisdiction: str = ""
    facts_provided: dict[str, Any] = Field(default_factory=dict)
    facts_missing: list[str] = Field(default_factory=list)
    expected_source_types: list[str] = Field(default_factory=list)
    expected_route: ExpectedRoute = Field(default_factory=ExpectedRoute)
    # Réponse simulée de l'utilisateur si l'assistant demande une
    # clarification (catégorie clarification_puis_recherche). None = la
    # trajectoire se termine sur la demande de clarification.
    clarification_answer: Optional[str] = None
    # Mode de panne simulé : panne_mcp | resultat_vide | source_trop_longue.
    failure_mode: Optional[str] = None


# ---------------------------------------------------------------------------
# Décisions du Planner / Trajectory Agent
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    ask_clarification = "ask_clarification"
    call_tool = "call_tool"
    final_answer = "final_answer"
    cannot_conclude = "cannot_conclude"


class DecisionTrace(BaseModel):
    """Trace de décision courte et stable — c'est elle qui devient le bloc
    <thinking> : type de demande, juridiction, besoin, prochaine action."""
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
    # Raisonnement en langue naturelle du Teacher sur le choix d'outil.
    # C'est ce texte qui devient le <thinking> dans les données d'entraînement.
    thinking_text: str = ""


# ---------------------------------------------------------------------------
# Appels et observations d'outils
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        """Rendu canonique injecté dans le tour assistant."""
        payload = {"name": self.name, "arguments": self.arguments}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                          sort_keys=True)


class ToolObservation(BaseModel):
    tool_name: str
    server: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: str = Field(default_factory=utcnow_iso)
    # Réponse brute, conservée HORS des prompts (peut être longue).
    raw_response: Any = None
    # Réponse normalisée : ce qui est réellement injecté dans le message tool.
    normalized_response: str = ""
    content_hash: str = ""
    source_urls: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    truncated: bool = False
    error: Optional[str] = None
    ok: bool = True
    # True UNIQUEMENT pour une fixture de test / dry-run, jamais en génération
    # réelle. Le validateur rejette toute trajectoire "réelle" contenant un mock.
    mock: bool = False
    latency_ms: Optional[float] = None

    def finalize_hash(self) -> "ToolObservation":
        if not self.content_hash:
            self.content_hash = sha256_text(self.normalized_response or "")
        return self


# ---------------------------------------------------------------------------
# État de recherche (machine à états de l'orchestrateur)
# ---------------------------------------------------------------------------

class StateStatus(str, Enum):
    planning = "planning"
    waiting_tool = "waiting_tool"
    answering = "answering"
    accepted = "accepted"
    rejected = "rejected"


class ResearchState(BaseModel):
    scenario: ScenarioSpec
    messages: list[Message] = Field(default_factory=list)
    tool_history: list[ToolObservation] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    step: int = Field(default=0, ge=0)
    max_tool_calls: int = Field(default=4, ge=0)
    jurisdiction_status: str = "unknown"
    missing_critical_facts: list[str] = Field(default_factory=list)
    status: StateStatus = StateStatus.planning
    stop_reason: Optional[str] = None

    def tool_calls_made(self) -> int:
        return len(self.tool_history)


# ---------------------------------------------------------------------------
# Critiques
# ---------------------------------------------------------------------------

class CriticResult(BaseModel):
    critic: str  # "legal" | "agentic"
    # Obligatoires : un JSON mal formé ne doit jamais devenir silencieusement
    # un rejet à score nul par application des valeurs par défaut.
    accepted: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Trajectoire d'entraînement (ligne JSONL maître)
# ---------------------------------------------------------------------------

class GenerationMetadata(BaseModel):
    teacher_model: str = ""
    # Jamais le base_url complet : il peut contenir un identifiant de pod.
    teacher_base_url_hash: str = ""
    critic_model: str = ""
    seed: int = 3407
    generated_at: str = Field(default_factory=utcnow_iso)
    prompt_version: str = ""
    tool_catalog_hash: str = ""


class QualityReport(BaseModel):
    legal_critic_score: Optional[float] = None
    agentic_critic_score: Optional[float] = None
    deterministic_validation: bool = False
    repaired: bool = False


class GroundingEntry(BaseModel):
    tool_name: str
    content_hash: str = ""
    source_urls: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


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
        """Clé de regroupement anti-fuite pour le split train/test.

        Une même famille de scénarios — et les trajectoires citant les mêmes
        sources principales — ne doit jamais être des deux côtés du split.
        """
        cites = sorted({c for g in self.grounding for c in g.citations})[:3]
        return f"fam:{self.scenario_family_id}|src:{'|'.join(cites)}"

    def final_answer(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == Role.assistant:
                return msg.content
        return ""

    def to_jsonl(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


class RejectionRecord(BaseModel):
    scenario_id: str
    request_type: str = ""
    stage: str = ""  # planner | executor | critic | validator | orchestrator
    reasons: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=utcnow_iso)
    trajectory: Optional[dict[str, Any]] = None


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
    # Copie de la configuration SANS secret (ni clé, ni base_url complet).
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
