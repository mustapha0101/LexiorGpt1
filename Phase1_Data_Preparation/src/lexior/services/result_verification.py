# -*- coding: utf-8 -*-
"""Service de vérification des résultats d'outils — classification à trois
niveaux de preuve.

Chaque résultat est classé en trois niveaux :

    candidate : identifie un document sans autorité probante.
    citable   : provient d'une source officielle et peut être cité.
    usable    : officiel, pertinent à la question juridique précise,
                compatible avec la juridiction demandée et capable de
                soutenir une affirmation.

Un résultat officiel ne devient PAS automatiquement une preuve utilisable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from lexior.agentic.citations import FEDERAL_CITATION_RE, QUEBEC_CITATION_RE
from lexior.agentic.planner_agent import FEDERAL_STATUTES
from lexior.agentic.response_verifier import (
    contains_generated_summary,
    verify_observation,
)
from lexior.agentic.schemas import (
    SearchEvaluation,
    SearchResultStatus,
    ToolObservation,
)
from lexior.agent_graph.result_classifier import ResultClassifier

from .evidence import (
    CITABLE_STATUSES,
    USABLE_STATUSES,
    CoverageGap,
    DetailedResultStatus,
    EvidenceEntry,
    EvidenceLevel,
    JurisdictionDimensions,
)
from .tool_coverage import (
    QUEBEC_COURT_SCOPES,
    TOOL_COVERAGE,
    get_coverage,
    has_equivalent_coverage,
)


_UNUSABLE = {
    SearchResultStatus.irrelevant,
    SearchResultStatus.empty,
    SearchResultStatus.wrong_document_type,
    SearchResultStatus.wrong_jurisdiction,
    SearchResultStatus.wrong_court_scope,
    SearchResultStatus.malformed,
    SearchResultStatus.tool_error,
    SearchResultStatus.stale,
    SearchResultStatus.coverage_unavailable,
}

_DOCUMENT_SEARCH_TOOLS = {"search_legal_documents", "fetch_document"}

_RETRIEVAL_ONLY_TOOLS = {
    "semantic_search_ccq", "semantic_search_cpc",
    "search_ccq_keywords", "search_cpc_keywords",
    "coverage",
}

_OFFICIAL_TOOLS = {
    "get_ccq_articles", "get_cpc_articles",
    "get_quebec_regulation", "get_quebec_legal_info",
    "fetch_document",
    "search_legal_documents",
    "search_quebec_regulations",
    "search_quebec_jurisprudence",
}

_TITLE_RE = re.compile(
    r"\b((?:Loi|Code|Charte|Règlement)\s+[^\n\r.;|]{3,80})", re.UNICODE)

_QC_COURT_RE = QUEBEC_CITATION_RE
_FED_COURT_RE = FEDERAL_CITATION_RE


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", stripped.lower())


@dataclass
class ToolResultAssessment:
    """Verdict complet sur un résultat d'outil."""

    tool_name: str
    tool_call_succeeded: bool
    search_status: str = SearchResultStatus.usable.value
    expected_document: str = ""
    returned_document: str = ""
    usable_as_evidence: bool = True
    reason: str = ""
    verifier_issues: list[str] = field(default_factory=list)

    evidence_level: str = EvidenceLevel.usable.value
    detailed_status: str = DetailedResultStatus.usable.value
    official: bool = False
    citable: bool = True
    relevant: bool = True
    jurisdiction_dims: Optional[dict] = None

    def to_search_evaluation(self, index: int) -> SearchEvaluation:
        return SearchEvaluation(
            tool_call_index=index,
            tool_name=self.tool_name,
            result_status=self.search_status,
            result_reason=self.reason,
        )

    def to_evidence_entry(self, index: int) -> EvidenceEntry:
        return EvidenceEntry(
            tool_history_index=index,
            tool_name=self.tool_name,
            evidence_level=EvidenceLevel(self.evidence_level),
            detailed_status=DetailedResultStatus(self.detailed_status),
            jurisdiction_dims=JurisdictionDimensions(
                **(self.jurisdiction_dims or {})),
            technical_success=self.tool_call_succeeded,
            official=self.official,
            citable=self.citable,
            relevant=self.relevant,
            usable=self.usable_as_evidence,
            reason=self.reason,
            expected_document=self.expected_document,
            returned_document=self.returned_document,
            verifier_issues=list(self.verifier_issues),
        )


class ResultVerificationService:
    """Vérification + classification à trois niveaux, partagées par les
    deux modes."""

    def __init__(self):
        self.classifier = ResultClassifier()

    # ── 1. Vérification MCP (stubs, abrogations, vides) ──────────────────

    def verify(
        self, observation: ToolObservation,
    ) -> tuple[ToolObservation, list[str]]:
        return verify_observation(observation)

    # ── 2-4. Classification complète ─────────────────────────────────────

    def assess(
        self,
        observation: ToolObservation,
        verifier_issues: Optional[list[str]] = None,
        resolved_jurisdiction: str = "",
        requested_court_scope: str = "",
        requested_document_type: str = "",
        user_query: str = "",
    ) -> ToolResultAssessment:
        # La question de l'usager fait partie du verdict : un résultat bien
        # formé mais étranger au sujet est `irrelevant`, ce qui déclenche la
        # reformulation via route_after_classification.
        status = self.classifier.classify_observation(
            observation, user_query=user_query)
        is_retrieval_only = observation.tool_name in _RETRIEVAL_ONLY_TOOLS
        is_official = observation.tool_name in _OFFICIAL_TOOLS
        # Une synthèse rédigée par un modèle n'est pas une source, quelle
        # que soit l'autorité du serveur qui la renvoie.
        is_generated = contains_generated_summary(
            observation.normalized_response)

        assessment = ToolResultAssessment(
            tool_name=observation.tool_name,
            tool_call_succeeded=observation.ok,
            search_status=status.value,
            usable_as_evidence=(status not in _UNUSABLE
                                and not is_retrieval_only),
            reason=(observation.error or ""),
            verifier_issues=list(verifier_issues or []),
            official=is_official and observation.ok,
            citable=(is_official and observation.ok
                     and not is_retrieval_only and not is_generated),
            relevant=True,
        )

        if is_generated:
            # Candidate au mieux : identifie une piste, ne prouve rien.
            assessment.evidence_level = EvidenceLevel.candidate.value
            assessment.usable_as_evidence = False
            assessment.detailed_status = (
                DetailedResultStatus.usable.value
                if status not in _UNUSABLE
                else _to_detailed(status).value)
            assessment.relevant = status not in _UNUSABLE
            assessment.reason = (
                assessment.reason
                or "contenu rédigé par un modèle : non citable comme preuve")
            return assessment

        if is_retrieval_only:
            assessment.evidence_level = EvidenceLevel.candidate.value
            assessment.detailed_status = (
                DetailedResultStatus.usable.value
                if status not in _UNUSABLE
                else _to_detailed(status).value)
            assessment.usable_as_evidence = False
            assessment.citable = False
            assessment.relevant = status not in _UNUSABLE
            assessment.reason = (assessment.reason
                                 or "retrieval-only: candidate evidence")
            return assessment

        if status in _UNUSABLE:
            assessment.evidence_level = EvidenceLevel.candidate.value
            assessment.detailed_status = _to_detailed(status).value
            assessment.usable_as_evidence = False
            assessment.relevant = False
            if (status == SearchResultStatus.irrelevant
                    and not assessment.reason and user_query):
                assessment.reason = (
                    "résultat sans rapport thématique avec la question posée")
            return assessment

        # Document match check for search tools.
        if (observation.ok
                and status == SearchResultStatus.usable
                and observation.tool_name in _DOCUMENT_SEARCH_TOOLS):
            expected = self.expected_document(observation.arguments)
            if expected:
                assessment.expected_document = expected
                returned = self.returned_document(
                    observation.normalized_response)
                assessment.returned_document = returned
                if not self._document_matches(
                        expected, observation.normalized_response):
                    assessment.search_status = (
                        SearchResultStatus.irrelevant.value)
                    assessment.detailed_status = (
                        DetailedResultStatus.irrelevant.value)
                    assessment.usable_as_evidence = False
                    assessment.relevant = False
                    assessment.evidence_level = (
                        EvidenceLevel.citable.value
                        if is_official else EvidenceLevel.candidate.value)
                    assessment.reason = (
                        f"document attendu « {expected} » absent du "
                        f"résultat"
                        + (f" (retourné : {returned})" if returned else ""))
                    return assessment

        # Court scope / jurisdiction matching.
        jurisdiction_dims = self._build_jurisdiction_dims(
            observation, resolved_jurisdiction, requested_court_scope)
        assessment.jurisdiction_dims = jurisdiction_dims.to_dict()

        if (requested_court_scope
                and not jurisdiction_dims.court_scope_matches()):
            if jurisdiction_dims.usable_as_alternative():
                assessment.detailed_status = (
                    DetailedResultStatus.alternative_only.value)
                assessment.search_status = (
                    SearchResultStatus.alternative_only.value)
                assessment.evidence_level = EvidenceLevel.citable.value
                assessment.usable_as_evidence = False
                assessment.reason = (
                    f"court scope {jurisdiction_dims.returned_court_scope} "
                    f"does not match requested "
                    f"{jurisdiction_dims.requested_court_scope}")
            else:
                assessment.detailed_status = (
                    DetailedResultStatus.wrong_court_scope.value)
                assessment.search_status = (
                    SearchResultStatus.wrong_court_scope.value)
                assessment.evidence_level = EvidenceLevel.candidate.value
                assessment.usable_as_evidence = False
                assessment.relevant = False
                assessment.reason = (
                    f"wrong court scope: "
                    f"{jurisdiction_dims.returned_court_scope}")
            return assessment

        # All checks passed — usable.
        if is_official:
            assessment.evidence_level = EvidenceLevel.usable.value
            assessment.detailed_status = DetailedResultStatus.usable.value
        else:
            assessment.evidence_level = EvidenceLevel.citable.value
            assessment.detailed_status = DetailedResultStatus.usable.value

        return assessment

    # ── Coverage gate ────────────────────────────────────────────────────

    def check_coverage(
        self,
        document_type: str = "",
        court_scope: str = "",
        jurisdiction: str = "",
        mode: str = "live",
    ) -> tuple[bool, list[str], CoverageGap | None]:
        """Check if a tool exists that covers the requested source.

        Returns (covered, tool_names, gap_or_none).
        """
        has_cov, tools, reason = has_equivalent_coverage(
            document_type, court_scope, jurisdiction, mode)
        if has_cov:
            return True, [t.tool_name for t in tools], None
        gap = CoverageGap(
            requested_document_type=document_type,
            requested_court_scope=court_scope,
            requested_jurisdiction=jurisdiction,
            available_tools=[],
            reason=reason,
        )
        return False, [], gap

    # ── Détection attendu / retourné ─────────────────────────────────────

    @staticmethod
    def expected_document(arguments: dict) -> str:
        query = " ".join(
            str(v) for k, v in (arguments or {}).items()
            if isinstance(v, str) and k in (
                "query", "keyword", "keywords", "title", "document_title")
        )
        folded = _fold(query)
        if not folded:
            return ""
        for _keywords, canonical, aliases in FEDERAL_STATUTES:
            if any(alias in folded for alias in aliases):
                return canonical
        return ""

    @staticmethod
    def returned_document(response: str) -> str:
        folded = _fold(response)
        for _keywords, canonical, aliases in FEDERAL_STATUTES:
            if any(alias in folded for alias in aliases):
                return canonical
        match = _TITLE_RE.search(response or "")
        return match.group(1).strip() if match else ""

    @staticmethod
    def _document_matches(expected: str, response: str) -> bool:
        folded = _fold(response)
        for _keywords, canonical, aliases in FEDERAL_STATUTES:
            if canonical == expected:
                return any(alias in folded for alias in aliases)
        return _fold(expected) in folded

    @staticmethod
    def _build_jurisdiction_dims(
        observation: ToolObservation,
        resolved_jurisdiction: str,
        requested_court_scope: str,
    ) -> JurisdictionDimensions:
        dims = JurisdictionDimensions(
            work_location=resolved_jurisdiction,
            requested_court_scope=requested_court_scope,
        )
        response = observation.normalized_response or ""
        qc_match = _QC_COURT_RE.search(response)
        fed_match = _FED_COURT_RE.search(response)
        if qc_match:
            dims.returned_court_scope = qc_match.group(2)
            dims.returned_source_jurisdiction = "Québec"
        elif fed_match:
            dims.returned_court_scope = fed_match.group(2)
            dims.returned_source_jurisdiction = "Federal"
        coverage = get_coverage(observation.tool_name)
        if coverage:
            if coverage.legal_jurisdictions:
                dims.requested_source_jurisdiction = (
                    coverage.legal_jurisdictions[0])
            if coverage.substantive_law_scopes:
                dims.substantive_law = coverage.substantive_law_scopes[0]
        return dims

    @staticmethod
    def is_retrieval_only(tool_name: str) -> bool:
        return tool_name in _RETRIEVAL_ONLY_TOOLS


def _to_detailed(status: SearchResultStatus) -> DetailedResultStatus:
    try:
        return DetailedResultStatus(status.value)
    except ValueError:
        return DetailedResultStatus.irrelevant
