# -*- coding: utf-8 -*-
"""Structured tool-coverage metadata.

Every tool in the catalog has explicit coverage metadata describing:
    - what document types it can retrieve
    - which jurisdictions and court systems it covers
    - its availability status and mode restrictions

The coverage gate uses this metadata to authorize tool selection and
classify non-equivalent fallback attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lexior.agentic.citations import (
    FEDERAL_COURT_SCOPES as _FEDERAL_COURT_SCOPES,
    QUEBEC_COURT_SCOPES as _QUEBEC_COURT_SCOPES,
)


@dataclass(frozen=True)
class ToolCoverageEntry:
    """Coverage metadata for a single tool."""

    tool_name: str
    document_types: tuple[str, ...] = ()
    legal_jurisdictions: tuple[str, ...] = ()
    court_scopes: tuple[str, ...] = ()
    substantive_law_scopes: tuple[str, ...] = ()
    enabled_modes: tuple[str, ...] = ("dataset", "live")
    availability_status: str = "available"
    availability_reason: str = ""
    retrieval_only: bool = False
    citable: bool = True

    def covers_court_scope(self, scope: str) -> bool:
        if not self.court_scopes:
            return False
        return scope in self.court_scopes

    def covers_jurisdiction(self, jurisdiction: str) -> bool:
        if not self.legal_jurisdictions:
            return False
        jur = jurisdiction.lower().strip()
        return any(j.lower() == jur for j in self.legal_jurisdictions)

    def is_available(self, mode: str = "live") -> bool:
        if mode not in self.enabled_modes:
            return False
        # In dataset mode, responses are scripted — server availability
        # is irrelevant.
        if mode == "dataset":
            return True
        return self.availability_status == "available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "document_types": list(self.document_types),
            "legal_jurisdictions": list(self.legal_jurisdictions),
            "court_scopes": list(self.court_scopes),
            "substantive_law_scopes": list(self.substantive_law_scopes),
            "enabled_modes": list(self.enabled_modes),
            "availability_status": self.availability_status,
            "availability_reason": self.availability_reason,
            "retrieval_only": self.retrieval_only,
            "citable": self.citable,
        }


# Réexportés depuis la source unique : voir agentic/citations.py.
QUEBEC_COURT_SCOPES = _QUEBEC_COURT_SCOPES
FEDERAL_COURT_SCOPES = _FEDERAL_COURT_SCOPES


# ── Registry ────────────────────────────────────────────────────────────

TOOL_COVERAGE: dict[str, ToolCoverageEntry] = {

    # ── Quebec MCP: statutes ────────────────────────────────────────────

    "get_ccq_articles": ToolCoverageEntry(
        tool_name="get_ccq_articles",
        document_types=("statute_article",),
        legal_jurisdictions=("Québec",),
        court_scopes=(),
        substantive_law_scopes=("Code civil du Québec",),
    ),
    "get_cpc_articles": ToolCoverageEntry(
        tool_name="get_cpc_articles",
        document_types=("statute_article",),
        legal_jurisdictions=("Québec",),
        court_scopes=(),
        substantive_law_scopes=("Code de procédure civile",),
    ),
    "search_ccq_keywords": ToolCoverageEntry(
        tool_name="search_ccq_keywords",
        document_types=("statute_article",),
        legal_jurisdictions=("Québec",),
        substantive_law_scopes=("Code civil du Québec",),
        retrieval_only=True,
        citable=False,
    ),
    "search_cpc_keywords": ToolCoverageEntry(
        tool_name="search_cpc_keywords",
        document_types=("statute_article",),
        legal_jurisdictions=("Québec",),
        substantive_law_scopes=("Code de procédure civile",),
        retrieval_only=True,
        citable=False,
    ),
    "semantic_search_ccq": ToolCoverageEntry(
        tool_name="semantic_search_ccq",
        document_types=("statute_article",),
        legal_jurisdictions=("Québec",),
        substantive_law_scopes=("Code civil du Québec",),
        retrieval_only=True,
        citable=False,
    ),
    "semantic_search_cpc": ToolCoverageEntry(
        tool_name="semantic_search_cpc",
        document_types=("statute_article",),
        legal_jurisdictions=("Québec",),
        substantive_law_scopes=("Code de procédure civile",),
        retrieval_only=True,
        citable=False,
    ),

    # ── Quebec MCP: regulations ─────────────────────────────────────────

    "search_quebec_regulations": ToolCoverageEntry(
        tool_name="search_quebec_regulations",
        document_types=("regulation",),
        legal_jurisdictions=("Québec",),
    ),
    "get_quebec_regulation": ToolCoverageEntry(
        tool_name="get_quebec_regulation",
        document_types=("regulation",),
        legal_jurisdictions=("Québec",),
    ),
    "get_quebec_legal_info": ToolCoverageEntry(
        tool_name="get_quebec_legal_info",
        document_types=("statute_metadata", "regulation_metadata"),
        legal_jurisdictions=("Québec",),
    ),

    # ── Quebec MCP: jurisprudence ───────────────────────────────────────

    "search_quebec_jurisprudence": ToolCoverageEntry(
        tool_name="search_quebec_jurisprudence",
        document_types=("court_decision",),
        legal_jurisdictions=("Québec",),
        court_scopes=QUEBEC_COURT_SCOPES,
        enabled_modes=("dataset", "live"),
        availability_status="available",
        availability_reason=(
            "Réactivé le 2026-07-24. Le taux d'échec observé auparavant "
            "était côté client : QCTAL manquait des expressions de citation "
            "de result_classifier, validators et case_law_gate, donc toute "
            "décision du Tribunal administratif du logement — la juridiction "
            "du contentieux locatif — était classée wrong_document_type. "
            "Liste unifiée dans agentic/citations.py. La formulation de la "
            "requête compte : nommer le tribunal et le mot « décision » "
            "ramène 4 décisions sur 8 résultats là où une formulation "
            "doctrinale en ramenait 0 sur 8 (voir prompts.py). Les résumés "
            "restent rédigés par un modèle côté serveur : la décision et son "
            "lien sont citables, le texte du résumé ne l'est pas "
            "(contains_generated_summary)."),
    ),

    # ── a2aj: federal ───────────────────────────────────────────────────

    "search_legal_documents": ToolCoverageEntry(
        tool_name="search_legal_documents",
        document_types=("court_decision", "statute", "regulation"),
        legal_jurisdictions=("Federal", "Canada"),
        court_scopes=FEDERAL_COURT_SCOPES,
        substantive_law_scopes=(
            "Canada Labour Code", "Criminal Code",
            "Canadian Human Rights Act", "Income Tax Act",
        ),
    ),
    "fetch_document": ToolCoverageEntry(
        tool_name="fetch_document",
        document_types=("court_decision", "statute", "regulation"),
        legal_jurisdictions=("Federal", "Canada"),
        court_scopes=FEDERAL_COURT_SCOPES,
    ),
    "coverage": ToolCoverageEntry(
        tool_name="coverage",
        document_types=("metadata",),
        legal_jurisdictions=("Federal", "Canada"),
        retrieval_only=True,
        citable=False,
    ),
}


def get_coverage(tool_name: str) -> ToolCoverageEntry | None:
    return TOOL_COVERAGE.get(tool_name)


def tools_covering_court_scope(scope: str) -> list[ToolCoverageEntry]:
    return [
        entry for entry in TOOL_COVERAGE.values()
        if entry.covers_court_scope(scope) and entry.is_available()
    ]


def has_equivalent_coverage(
    document_type: str,
    court_scope: str = "",
    jurisdiction: str = "",
    mode: str = "live",
) -> tuple[bool, list[ToolCoverageEntry], str]:
    """Check if any active tool provides equivalent coverage.

    Returns (has_coverage, matching_tools, reason).
    """
    candidates = []
    for entry in TOOL_COVERAGE.values():
        if not entry.is_available(mode):
            continue
        if document_type and document_type not in entry.document_types:
            continue
        if court_scope and not entry.covers_court_scope(court_scope):
            continue
        if jurisdiction and not entry.covers_jurisdiction(jurisdiction):
            continue
        candidates.append(entry)

    if candidates:
        return True, candidates, ""

    reason_parts = []
    if court_scope:
        reason_parts.append(f"court scope {court_scope}")
    if jurisdiction:
        reason_parts.append(f"jurisdiction {jurisdiction}")
    if document_type:
        reason_parts.append(f"document type {document_type}")
    reason = (
        "no active tool covers "
        + " + ".join(reason_parts or ["the requested source"]))
    return False, [], reason
