# -*- coding: utf-8 -*-
"""Unified search-result classification with 8 statuses.

Consolidates the classification logic from:
  - ``validators.classify_search_result``  (basic tool-level classification)
  - ``case_law_gate.classify_case_result`` (jurisprudence relevance scoring)

into one ``ResultClassifier`` class that returns ``SearchResultStatus`` values.
"""

from __future__ import annotations

import re
from typing import Optional

from lexior.agentic.schemas import (
    CaseRelevanceResult,
    CaseLawSearchStatus,
    SearchResultStatus,
    ToolObservation,
)
from lexior.agentic.case_law_gate import (
    classify_case_result as _classify_case,
    gate_search_results as _gate_search,
    filter_usable_cases,
)


# ── Patterns ─────────────────────────────────────────────────────────────

_CASE_CITATION_RE = re.compile(
    r"\b\d{4}\s+(?:QCCA|QCCS|QCCQ|QCTDP|QCRDL|SCC|CSC|FC|CF|FCA|CAF)\s+\d+\b"
)
_CASE_NAME_RE = re.compile(r"[A-ZÀ-Ÿ][\w'-]+\s+c\.\s+[A-ZÀ-Ÿ][\w'-]+")
_LEGISLATION_KEYWORDS = ("loi sur", "règlement sur", "code civil",
                         "code de procédure")
_TRUNCATION_MARKERS = ("...", "[suite]", "[truncated]", "[tronqué]")

_QC_STUB_RE = re.compile(
    r"^\s*\(\s*(?:abrog|omis|modification)", re.IGNORECASE)
_FED_REPEAL_RE = re.compile(r"\[\s*(?:Abrog|Repealed)", re.IGNORECASE)

MIN_CONTENT_CHARS = 30


class ResultClassifier:
    """Deterministic classifier for MCP tool results.

    Returns one of the eight ``SearchResultStatus`` values:
    usable, irrelevant, empty, wrong_document_type, truncated,
    malformed, tool_error, stale.
    """

    def classify(
        self,
        tool_name: str,
        response: str,
        ok: bool,
        error: Optional[str] = None,
    ) -> SearchResultStatus:
        """Classify a single tool result."""
        if not ok:
            return SearchResultStatus.tool_error

        stripped = (response or "").strip()

        if not stripped or stripped in ("[]", "{}", "null"):
            return SearchResultStatus.empty

        # Tool-specific classification before generic length check,
        # because stubs like "(Abrogé)" are short but meaningfully stale.
        if tool_name in ("get_ccq_articles", "get_cpc_articles"):
            return self._classify_article(stripped)

        if tool_name == "fetch_document":
            return self._classify_federal_doc(stripped)

        if len(stripped) < MIN_CONTENT_CHARS:
            return SearchResultStatus.empty

        if tool_name == "search_quebec_jurisprudence":
            return self._classify_jurisprudence(stripped)

        if tool_name in ("semantic_search_ccq", "semantic_search_cpc"):
            return self._classify_semantic_search(stripped)

        if any(marker in stripped for marker in _TRUNCATION_MARKERS):
            return SearchResultStatus.truncated

        return SearchResultStatus.usable

    def classify_observation(
        self, observation: ToolObservation,
    ) -> SearchResultStatus:
        """Classify a ToolObservation directly."""
        return self.classify(
            observation.tool_name,
            observation.normalized_response,
            observation.ok,
            observation.error,
        )

    def classify_case(
        self,
        result_text: str,
        target_articles: list[str],
        user_situation: str,
        jurisdiction: str = "Québec",
    ) -> CaseRelevanceResult:
        """Classify a single jurisprudence result for relevance.

        Delegates to ``case_law_gate.classify_case_result``.
        """
        return _classify_case(
            result_text, target_articles, user_situation, jurisdiction)

    def gate_search_results(
        self,
        results_text: str,
        target_articles: list[str],
        user_situation: str,
    ) -> tuple[list[CaseRelevanceResult], CaseLawSearchStatus]:
        """Batch-classify jurisprudence search results.

        Delegates to ``case_law_gate.gate_search_results``.
        """
        return _gate_search(results_text, target_articles, user_situation)

    @staticmethod
    def filter_usable(
        results: list[CaseRelevanceResult],
    ) -> list[CaseRelevanceResult]:
        """Return only usable cases sorted by relevance."""
        return filter_usable_cases(results)

    # ── Tool-specific classifiers ────────────────────────────────────────

    @staticmethod
    def _classify_article(text: str) -> SearchResultStatus:
        if _QC_STUB_RE.match(text):
            return SearchResultStatus.stale
        if any(marker in text for marker in _TRUNCATION_MARKERS):
            return SearchResultStatus.truncated
        return SearchResultStatus.usable

    @staticmethod
    def _classify_jurisprudence(text: str) -> SearchResultStatus:
        has_citation = bool(_CASE_CITATION_RE.search(text))
        has_case_name = bool(_CASE_NAME_RE.search(text))
        if not has_citation and not has_case_name:
            if any(kw in text.lower() for kw in _LEGISLATION_KEYWORDS):
                return SearchResultStatus.wrong_document_type
            return SearchResultStatus.irrelevant
        return SearchResultStatus.usable

    @staticmethod
    def _classify_semantic_search(text: str) -> SearchResultStatus:
        if any(marker in text for marker in _TRUNCATION_MARKERS):
            return SearchResultStatus.truncated
        return SearchResultStatus.usable

    @staticmethod
    def _classify_federal_doc(text: str) -> SearchResultStatus:
        if _FED_REPEAL_RE.search(text[:400]):
            lines = [line for line in text.split("\n") if line.strip()]
            repealed = sum(
                1 for line in lines
                if re.match(r"^\s*\[[^\]]{0,160}\]\s*$", line)
                and re.match(r"(?:abrog|repealed)\b",
                             line.strip().strip("[]").strip().lower())
            )
            if repealed and repealed >= len(lines):
                return SearchResultStatus.stale
        if any(marker in text for marker in _TRUNCATION_MARKERS):
            return SearchResultStatus.truncated
        return SearchResultStatus.usable
