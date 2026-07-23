# -*- coding: utf-8 -*-
"""Service de réparation — classification des échecs et retour ciblé.

Deux réparations distinctes (jamais un « repair » générique) :

    repair_answer     — la rédaction est fautive, les preuves sont bonnes.
    repair_trajectory — la recherche est fautive; on répare le PROCESSUS
                        en retournant au premier nœud invalide.

``classify_failures`` transforme les sorties des critiques et des
validateurs en catégories d'échec typées, chacune associée au nœud du
graphe qui doit rejouer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from lexior.agentic.schemas import ResearchState
from lexior.agentic.trajectory_agent import TrajectoryAgent

from .critics import CriticsOutcome
from .modes import LIVE, normalize_mode

# ── Catégories d'échec → nœud cible ──────────────────────────────────────

FAILURE_TARGETS = {
    "writing": "repair_answer",
    "grounding": "repair_answer",
    "retrieval": "repair_trajectory",
    "wrong_document_type": "repair_trajectory",
    "tool_execution": "repair_trajectory",
    "planning": "repair_trajectory",
    "legal_status": "repair_trajectory",
    "jurisdiction": "resolve_jurisdiction",
    "clarification": "handle_clarification",
    "conversation_follow_up": "build_answer_contract",
}

# Priorité de traitement quand plusieurs catégories coexistent :
# corriger le processus avant la prose.
_PRIORITY = [
    "clarification",
    "jurisdiction",
    "tool_execution",
    "wrong_document_type",
    "retrieval",
    "legal_status",
    "planning",
    "conversation_follow_up",
    "grounding",
    "writing",
]

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("jurisdiction", re.compile(r"juridiction|province|hors qu[ée]bec", re.I)),
    ("clarification", re.compile(
        r"clarification|faits? essentiels?|pr[ée]cision manquante", re.I)),
    ("wrong_document_type", re.compile(
        r"type de document|wrong_document_type|l[ée]gislation au lieu|"
        r"d[ée]cision au lieu", re.I)),
    ("retrieval", re.compile(
        r"document attendu|non pertinent|irrelevant|aucun r[ée]sultat|"
        r"recherche (?:vide|infructueuse)", re.I)),
    ("tool_execution", re.compile(
        r"panne|erreur (?:MCP|d'outil)|tool_error|malform", re.I)),
    ("legal_status", re.compile(r"abrog|stale|obsol[èe]te|p[ée]rim", re.I)),
    ("planning", re.compile(
        r"route|s[ée]quence|outil requis|outil manquant|max_tool_calls", re.I)),
    ("conversation_follow_up", re.compile(
        r"question de suivi|derni[èe]re question|r[ée]p[èe]te la r[ée]ponse",
        re.I)),
    ("grounding", re.compile(
        r"URL absente|citation absente|article .* absent|"
        r"certitude non justifi[ée]e|source r[ée]cup[ée]r[ée]e|invent[ée]",
        re.I)),
]


def classify_issue(issue: str) -> str:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(issue or ""):
            return category
    return "writing"


@dataclass
class FailureReport:
    """Un échec classé, avec le nœud du graphe qui doit rejouer."""

    category: str
    target_node: str
    instructions: list[str] = field(default_factory=list)
    source: str = ""  # legal_critic | agentic_critic | validator | verifier

    @staticmethod
    def from_issue(issue: str, source: str) -> "FailureReport":
        category = classify_issue(issue)
        return FailureReport(
            category=category,
            target_node=FAILURE_TARGETS[category],
            instructions=[issue],
            source=source,
        )


class RepairService:
    def __init__(self, answer_service, legal_min_score: float = 0.7,
                 agentic_min_score: float = 0.7, max_repairs: int = 1):
        self.answer_service = answer_service
        self.legal_min_score = legal_min_score
        self.agentic_min_score = agentic_min_score
        self.max_repairs = max_repairs

    # ── Classification ───────────────────────────────────────────────────

    def classify_failures(self, critics: Optional[CriticsOutcome],
                          validation_issues: Optional[list[str]] = None,
                          ) -> list[FailureReport]:
        reports: list[FailureReport] = []
        if critics:
            for result, source in ((critics.legal, "legal_critic"),
                                   (critics.agentic, "agentic_critic")):
                if result is None:
                    continue
                min_score = (self.legal_min_score if source == "legal_critic"
                             else self.agentic_min_score)
                if result.accepted and result.score >= min_score:
                    continue
                issues = (result.repair_instructions or result.issues
                          or ["réponse insuffisante"])
                for issue in issues:
                    reports.append(FailureReport.from_issue(issue, source))
        for issue in validation_issues or []:
            reports.append(FailureReport.from_issue(issue, "validator"))
        return self._merge(reports)

    @staticmethod
    def _merge(reports: list[FailureReport]) -> list[FailureReport]:
        by_category: dict[str, FailureReport] = {}
        for report in reports:
            existing = by_category.get(report.category)
            if existing:
                existing.instructions.extend(report.instructions)
            else:
                by_category[report.category] = report
        return sorted(
            by_category.values(),
            key=lambda r: _PRIORITY.index(r.category),
        )

    @staticmethod
    def primary(reports: list[FailureReport]) -> Optional[FailureReport]:
        return reports[0] if reports else None

    # ── Réparation de la rédaction ───────────────────────────────────────

    def repair_answer(self, state: ResearchState, mode: str, answer: str,
                      thinking: str,
                      instructions: list[str]) -> tuple[str, str]:
        writer: TrajectoryAgent = self.answer_service.writer_for(
            normalize_mode(mode))
        return writer.repair(state, answer, thinking, instructions)
