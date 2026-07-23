# -*- coding: utf-8 -*-
"""Evidence classification — three-tier evidence model.

Every tool result passes through the evidence pipeline:

    candidate → citable → usable

Definitions:
    candidate: returned by semantic or keyword search; used only to
        identify possible documents or article numbers.
    citable: retrieved from an official source; may be cited.
    usable: official, citable, relevant to the precise legal issue,
        compatible with the requested jurisdiction, and capable of
        supporting a claim in the final answer.

An official source does NOT automatically become usable evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Evidence levels ─────────────────────────────────────────────────────

class EvidenceLevel(str, Enum):
    candidate = "candidate"
    citable = "citable"
    usable = "usable"


# ── Extended result statuses ────────────────────────────────────────────

class DetailedResultStatus(str, Enum):
    exact_match = "exact_match"
    usable = "usable"
    alternative_only = "alternative_only"
    wrong_jurisdiction = "wrong_jurisdiction"
    wrong_court_scope = "wrong_court_scope"
    wrong_document_type = "wrong_document_type"
    irrelevant = "irrelevant"
    empty = "empty"
    truncated = "truncated"
    malformed = "malformed"
    tool_error = "tool_error"
    stale = "stale"
    coverage_unavailable = "coverage_unavailable"


USABLE_STATUSES = {
    DetailedResultStatus.exact_match,
    DetailedResultStatus.usable,
}

CITABLE_STATUSES = USABLE_STATUSES | {
    DetailedResultStatus.alternative_only,
    DetailedResultStatus.truncated,
}


# ── Jurisdiction dimensions ─────────────────────────────────────────────

@dataclass
class JurisdictionDimensions:
    """Separate jurisdiction dimensions — never a single ambiguous field."""

    work_location: str = ""
    applicable_legal_jurisdiction: str = ""
    requested_source_jurisdiction: str = ""
    requested_court_scope: str = ""
    returned_source_jurisdiction: str = ""
    returned_court_scope: str = ""
    substantive_law: str = ""

    def court_scope_matches(self) -> bool:
        if not self.requested_court_scope or not self.returned_court_scope:
            return True
        return self.requested_court_scope == self.returned_court_scope

    def source_jurisdiction_matches(self) -> bool:
        if not self.requested_source_jurisdiction:
            return True
        if not self.returned_source_jurisdiction:
            return True
        return (self.requested_source_jurisdiction
                == self.returned_source_jurisdiction)

    def usable_as_exact_answer(self) -> bool:
        return (self.court_scope_matches()
                and self.source_jurisdiction_matches())

    def usable_as_alternative(self) -> bool:
        if self.usable_as_exact_answer():
            return True
        if (self.returned_court_scope in ("SCC", "CSC")
                and self.work_location in (
                    "quebec", "québec", "Québec")):
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_location": self.work_location,
            "applicable_legal_jurisdiction":
                self.applicable_legal_jurisdiction,
            "requested_source_jurisdiction":
                self.requested_source_jurisdiction,
            "requested_court_scope": self.requested_court_scope,
            "returned_source_jurisdiction":
                self.returned_source_jurisdiction,
            "returned_court_scope": self.returned_court_scope,
            "substantive_law": self.substantive_law,
        }


# ── Evidence entry ──────────────────────────────────────────────────────

@dataclass
class EvidenceEntry:
    """A classified piece of evidence from a tool result."""

    tool_history_index: int
    tool_name: str
    evidence_level: EvidenceLevel
    detailed_status: DetailedResultStatus
    jurisdiction_dims: JurisdictionDimensions = field(
        default_factory=JurisdictionDimensions)

    technical_success: bool = True
    official: bool = False
    citable: bool = False
    relevant: bool = False
    usable: bool = False

    reason: str = ""
    expected_document: str = ""
    returned_document: str = ""
    verifier_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_history_index": self.tool_history_index,
            "tool_name": self.tool_name,
            "evidence_level": self.evidence_level.value,
            "detailed_status": self.detailed_status.value,
            "technical_success": self.technical_success,
            "official": self.official,
            "citable": self.citable,
            "relevant": self.relevant,
            "usable": self.usable,
            "reason": self.reason,
            "expected_document": self.expected_document,
            "returned_document": self.returned_document,
            "jurisdiction": self.jurisdiction_dims.to_dict(),
        }


# ── Coverage gap ────────────────────────────────────────────────────────

@dataclass
class CoverageGap:
    """Records a coverage gap — the system cannot satisfy the request."""

    requested_document_type: str = ""
    requested_court_scope: str = ""
    requested_jurisdiction: str = ""
    available_tools: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_document_type": self.requested_document_type,
            "requested_court_scope": self.requested_court_scope,
            "requested_jurisdiction": self.requested_jurisdiction,
            "available_tools": self.available_tools,
            "reason": self.reason,
        }


# ── Acceptance blockers ─────────────────────────────────────────────────

class AcceptanceBlocker(str, Enum):
    coverage_mismatch = "coverage_mismatch"
    requested_court_scope_not_covered = "requested_court_scope_not_covered"
    silent_non_equivalent_fallback = "silent_non_equivalent_fallback"
    wrong_source_jurisdiction = "wrong_source_jurisdiction"
    wrong_court_scope = "wrong_court_scope"
    irrelevant_official_source = "irrelevant_official_source"
    retrieval_only_source_used_as_evidence = (
        "retrieval_only_source_used_as_evidence")
    no_usable_evidence = "no_usable_evidence"
    unsupported_legal_claim = "unsupported_legal_claim"
    requested_output_missing = "requested_output_missing"
    coverage_failure_presented_as_absence_of_law = (
        "coverage_failure_presented_as_absence_of_law")
