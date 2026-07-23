# -*- coding: utf-8 -*-
"""StepVerifier — centralized deterministic validation for the Lexior graph.

Consolidates and corrects validation logic from:
  - ``validators.py``         (route, trajectory, planner decision, query quality)
  - ``acceptance.py``         (jurisdiction, clarification, grounding, acceptance)
  - ``response_verifier.py``  (MCP response verification)

into one class with 11 public validate methods plus utilities.

Old modules remain for backward compatibility; this class is the single
entry point for the LangGraph graph (Phase 2).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from lexior.agentic.schemas import (
    AcceptanceResult,
    CriticResult,
    Decision,
    PlannerDecision,
    ResearchState,
    ToolObservation,
    TrainingTrajectory,
)
from lexior.agentic.acceptance import (
    compute_acceptance_status,
    _check_jurisdiction_consistency,
    _check_clarification_consistency,
    _check_article_grounding,
)
from lexior.agentic.response_verifier import (
    verify_observation as _verify_observation,
)
from lexior.agentic.tool_catalog import ToolCatalog
from lexior.agentic.validators import (
    ARTICLE_CITATION_RE,
    CERTAINTY_RE,
    CITATION_MARK_RE,
    PRECISE_ARTICLE_TOOLS,
    URL_RE,
    ValidationResult,
    validate_jurisprudence_query as _validate_jurisprudence_query,
    validate_next_action,
    validate_planner_decision as _validate_planner_decision,
    validate_tool_route as _validate_tool_route,
    validate_tool_sequence_logic as _validate_tool_sequence_logic,
    validate_trajectory as _validate_trajectory,
    _detect_language_mismatch,
)


# ── Proposal verdict ─────────────────────────────────────────────────────


class ProposalVerdict(str, Enum):
    permit = "permit"
    modify = "modify"
    reject = "reject"


@dataclass
class VerifiedProposal:
    """Result of verifying a planner proposal."""
    verdict: ProposalVerdict
    decision: PlannerDecision
    reason: str = ""
    errors: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────

_SEARCH_TO_FETCH = [
    ("semantic_search_ccq", "get_ccq_articles"),
    ("semantic_search_cpc", "get_cpc_articles"),
]


def _find_official_text(
    tool_trace: list[ToolObservation],
    tool_names: tuple[str, ...] | set[str],
) -> str:
    for obs in reversed(tool_trace):
        if obs.tool_name in tool_names and obs.ok and not obs.truncated:
            text = (obs.normalized_response or "").strip()
            if text:
                return text
    return ""


# ── StepVerifier ─────────────────────────────────────────────────────────


class StepVerifier:
    """Centralized deterministic validation for the Lexior agent graph.

    Provides 11 validate methods plus utilities.  Each method delegates to
    the existing corrected logic in validators.py / acceptance.py /
    response_verifier.py, providing a single entry point.

    In Phase 2, the LangGraph nodes call these methods; old modules
    are deprecated.
    """

    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog

    # ── 1. Gate: planner proposal verification ───────────────────────────

    def verify_proposal(
        self,
        decision: PlannerDecision,
        request_type: str,
        tool_history: list[ToolObservation],
        max_tool_calls: int = 4,
    ) -> VerifiedProposal:
        """Verify a planner proposal.  Permit, modify, or reject.

        - Validates the decision structure (tool exists, arguments valid).
        - Checks the proposed tool against route policy.
        - Enforces max_tool_calls.
        """
        errors = _validate_planner_decision(decision, self.catalog)
        if errors:
            return VerifiedProposal(
                ProposalVerdict.reject, decision, errors=errors,
                reason="décision invalide",
            )

        if decision.decision == Decision.call_tool and decision.next_tool:
            route_errors = validate_next_action(
                request_type, decision.next_tool)
            if route_errors:
                return VerifiedProposal(
                    ProposalVerdict.reject, decision, errors=route_errors,
                    reason="outil interdit par la politique de routage",
                )

            if len(tool_history) >= max_tool_calls:
                return VerifiedProposal(
                    ProposalVerdict.reject, decision,
                    errors=["max_tool_calls atteint"],
                    reason="max_tool_calls atteint",
                )

        return VerifiedProposal(ProposalVerdict.permit, decision)

    # ── 2. Route validation ──────────────────────────────────────────────

    def validate_tool_route(
        self,
        request_type: str,
        sequence: list[str],
        exempt_tools: Optional[list[str]] = None,
    ) -> list[str]:
        """Validate the complete tool sequence against the expected route."""
        return _validate_tool_route(request_type, sequence, exempt_tools)

    # ── 3. Sequence logic ────────────────────────────────────────────────

    def validate_tool_sequence(
        self,
        request_type: str,
        sequence: list[str],
    ) -> list[str]:
        """Check logical ordering of tool calls."""
        return _validate_tool_sequence_logic(request_type, sequence)

    # ── 4. Jurisprudence query quality ───────────────────────────────────

    def validate_jurisprudence_query(self, query: str) -> list[str]:
        """Check that a jurisprudence search query is well-structured."""
        return _validate_jurisprudence_query(query)

    # ── 5. MCP observation verification ──────────────────────────────────

    def validate_observation(
        self, observation: ToolObservation,
    ) -> tuple[ToolObservation, list[str]]:
        """Verify an MCP response (stubs, repeals, empty results)."""
        return _verify_observation(observation)

    # ── 6. Jurisdiction consistency ──────────────────────────────────────

    def validate_jurisdiction(
        self,
        trajectory: TrainingTrajectory,
        state: Optional[ResearchState] = None,
    ) -> list[str]:
        """Check jurisdiction consistency between query and declared status."""
        return _check_jurisdiction_consistency(trajectory, state)

    # ── 7. Clarification consistency ─────────────────────────────────────

    def validate_clarification(
        self,
        trajectory: TrainingTrajectory,
        state: Optional[ResearchState] = None,
    ) -> list[str]:
        """Check clarification metadata is self-consistent."""
        return _check_clarification_consistency(trajectory, state)

    # ── 8. Article grounding ─────────────────────────────────────────────

    def validate_article_grounding(
        self, trajectory: TrainingTrajectory,
    ) -> list[str]:
        """Check that cited articles were actually retrieved."""
        return _check_article_grounding(trajectory)

    # ── 9. Final answer validation ───────────────────────────────────────

    def validate_final_answer(
        self,
        final_answer: str,
        tool_trace: list[ToolObservation],
        request_type: str,
        language: str = "fr",
    ) -> list[str]:
        """Validate the final answer against tool evidence.

        Checks: non-empty, exact-text fidelity, JSON wrapping,
        unjustified certainty, URL/citation/article grounding,
        language consistency.
        """
        errors: list[str] = []
        final = final_answer.strip()

        if not final:
            errors.append("réponse finale vide")
            return errors

        # Exact text retrieval: word-for-word match
        article_tools = PRECISE_ARTICLE_TOOLS.get(request_type)
        if article_tools:
            official = _find_official_text(tool_trace, article_tools)
            if not official:
                errors.append(
                    "texte officiel complet de l'article indisponible")
            elif final != official:
                errors.append(
                    "texte d'article précis non reproduit "
                    "intégralement mot pour mot")
        elif request_type == "article_explanation":
            official = _find_official_text(
                tool_trace, ("get_ccq_articles", "get_cpc_articles"))
            if not official:
                errors.append(
                    "texte officiel complet de l'article indisponible")
            elif official not in final:
                errors.append(
                    "explication sans reproduction intégrale "
                    "du texte officiel")

        # JSON wrapping
        if final.startswith("{") and final.endswith("}"):
            try:
                wrapped = json.loads(final)
            except ValueError:
                wrapped = None
            if isinstance(wrapped, dict) and "answer" in wrapped:
                errors.append(
                    "réponse finale enveloppée dans un objet JSON")

        # Unjustified certainty
        if CERTAINTY_RE.search(final) and (
            not tool_trace
            or any(o.error or o.truncated for o in tool_trace)
        ):
            errors.append(
                "certitude non justifiée par les preuves disponibles")

        # URL grounding
        citable = [
            o for o in tool_trace
            if o.tool_name not in {
                "semantic_search_ccq", "semantic_search_cpc"}
        ]
        available_urls = {
            u.rstrip(".,;)") for o in citable for u in o.source_urls}
        final_urls = {u.rstrip(".,;)") for u in URL_RE.findall(final)}
        invented = final_urls - available_urls
        if invented:
            errors.append(
                f"URL absente des réponses d'outils : {sorted(invented)}")

        # Citation grounding
        available_citations = {
            c.casefold() for o in citable for c in o.citations}
        for citation in CITATION_MARK_RE.findall(final):
            if citation.startswith("["):
                if not final_urls:
                    errors.append(
                        f"citation {citation} sans source récupérée")
            elif citation.casefold() not in available_citations:
                errors.append(
                    f"citation absente des réponses d'outils : {citation}")

        # Article grounding
        evidence_text = "\n".join(
            o.normalized_response for o in citable
        ).casefold()
        for article in ARTICLE_CITATION_RE.findall(final):
            if not re.search(
                rf"\barticle\s+{re.escape(article)}\b",
                evidence_text, re.IGNORECASE,
            ):
                errors.append(
                    f"article {article} absent des réponses d'outils")

        # Language mismatch
        lang_err = _detect_language_mismatch(final, language)
        if lang_err:
            errors.append(lang_err)

        return errors

    # ── 10. Full trajectory validation ───────────────────────────────────

    def validate_trajectory(
        self,
        trajectory: TrainingTrajectory,
        allow_mock: bool = False,
        max_tool_calls: int = 4,
        seen_fingerprints: Optional[set[str]] = None,
        exempt_tools: Optional[list[str]] = None,
    ) -> ValidationResult:
        """Full deterministic trajectory validation."""
        return _validate_trajectory(
            trajectory, self.catalog,
            allow_mock=allow_mock,
            max_tool_calls=max_tool_calls,
            seen_fingerprints=seen_fingerprints,
            exempt_tools=exempt_tools,
        )

    # ── 11. Acceptance decision ──────────────────────────────────────────

    def compute_acceptance(
        self,
        trajectory: TrainingTrajectory,
        validation: ValidationResult,
        legal_critic: Optional[CriticResult] = None,
        agentic_critic: Optional[CriticResult] = None,
        legal_min_score: float = 0.7,
        agentic_min_score: float = 0.7,
        state: Optional[ResearchState] = None,
    ) -> AcceptanceResult:
        """Single function determining accept/reject with full reasons."""
        return compute_acceptance_status(
            trajectory, validation, legal_critic, agentic_critic,
            legal_min_score=legal_min_score,
            agentic_min_score=agentic_min_score,
            state=state,
        )

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def compute_exempt_tools(
        tool_history: list[ToolObservation],
    ) -> list[str]:
        """Compute tools exempted due to empty upstream searches.

        When a semantic search returns empty results twice, the
        dependent fetch tool is exempted from route requirements.

        This logic was previously duplicated in orchestrator.py,
        validators.py (via parameter), and agentic_critic.py.
        """
        exempt: list[str] = []
        for search_tool, fetch_tool in _SEARCH_TO_FETCH:
            searches = [
                o for o in tool_history if o.tool_name == search_tool]
            if len(searches) >= 2 and all(
                not o.ok
                or (o.normalized_response or "").strip() in ("", "[]", "{}")
                for o in searches
            ):
                exempt.append(fetch_tool)
        return exempt

    def find_first_invalid_step(
        self,
        tool_history: list[ToolObservation],
        request_type: str,
        exempt_tools: Optional[list[str]] = None,
    ) -> Optional[int]:
        """Find the index of the first invalid step for repair routing.

        Returns ``None`` if the trajectory has no invalid steps.
        """
        from lexior.agentic.taxonomy import REQUEST_TYPES

        if not tool_history:
            return None

        rt = REQUEST_TYPES.get(request_type)
        if not rt:
            return 0

        expected = rt.expected_route.required_tools()
        exempt_set = set(exempt_tools or [])
        sequence = [o.tool_name for o in tool_history]

        # Out-of-order required tools
        expected_idx = 0
        for i, tool_name in enumerate(sequence):
            if tool_name in expected:
                pos = expected.index(tool_name)
                if pos < expected_idx:
                    return i
                expected_idx = pos + 1

        # Missing required tools → step after last valid tool
        for tool in expected:
            if tool not in sequence and tool not in exempt_set:
                return len(sequence)

        # Failed tool calls
        for i, obs in enumerate(tool_history):
            if not obs.ok and obs.error:
                return i

        return None
