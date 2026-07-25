# -*- coding: utf-8 -*-
"""Centralized acceptance logic for the agentic pipeline.

Single source of truth: every accept/reject decision flows through
``compute_acceptance_status``.  Blocking errors cause rejection;
warnings are recorded but do not block.
"""

from __future__ import annotations

import re
from typing import Any, Optional, TYPE_CHECKING

from .error_codes import BLOCKING_CODES, ErrorCode, extract_code, tag
from .schemas import (
    AcceptanceResult, CriticResult, ResearchState,
    TrainingTrajectory,
)
from .taxonomy import NO_JURISPRUDENCE, REQUEST_TYPES

if TYPE_CHECKING:
    from .validators import ValidationResult

# ── Blocking vs non-blocking classification ──────────────────────────────

BLOCKING_PATTERNS: list[str] = [
    "mauvaise juridiction",
    "jurisdiction_mismatch",
    "outil requis absent",
    "outil obligatoire absent",
    "article .* absent des réponses",
    "URL absente des réponses",
    "citation absente des réponses",
    "réponse finale vide",
    "texte officiel complet de l'article indisponible",
    "texte d'article précis non reproduit",
    "explication sans reproduction intégrale",
    "tool_call sans tool message",
    "tool message .* sans tool_call",
    "tool_trace ne correspond pas",
    "réponse MCP fabriquée",
    "boucle d'appels identiques",
    "doublon exact",
    "quasi-duplicat",
    "outil interdit",
    "outil hors route",
    "chemin local temporaire",
    "réponse finale enveloppée",
    "unsupported_claim",
    "fabricated_case_law_pattern",
    "unsupported_deadline",
    "wrong_jurisdiction",
    "unsupported_article",
    "wrong_document_type_accepted",
    "unretrieved_article_used",
]

NON_BLOCKING_PATTERNS: list[str] = [
    "thinking.*long",
    "style.*imparfait",
    "register_informal",
    "thinking_too_long",
    "requête améliorable",
    "absence de jurisprudence",
    "recherche de jurisprudence inutile",
    "appel d'outil et texte substantiel",
    "certitude non justifiée",
    "bad_query",
    "mechanical_route_following",
]


def _is_blocking(error: str) -> bool:
    """Bloquant ? Sur le CODE quand il y en a un, sinon sur le texte.

    Les validators émettent désormais un code (``[ungrounded_article] …``)
    et le français ne sert plus qu'à l'affichage : renommer un message ne
    peut plus changer le critère d'acceptation du dataset. Les messages des
    critics LLM n'ont pas de code et retombent sur les motifs historiques —
    où BLOCKING est maintenant testé AVANT NON_BLOCKING, sans quoi un
    message correspondant aux deux passait en simple avertissement.
    """
    code = extract_code(error)
    if code is not None:
        return code in BLOCKING_CODES
    folded = error.casefold()
    for pattern in BLOCKING_PATTERNS:
        if re.search(pattern, folded, re.IGNORECASE):
            return True
    for pattern in NON_BLOCKING_PATTERNS:
        if re.search(pattern, folded, re.IGNORECASE):
            return False
    return True


def _classify_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    blocking = []
    warnings = []
    for error in errors:
        if _is_blocking(error):
            blocking.append(error)
        else:
            warnings.append(error)
    return blocking, warnings


# Erreurs qui reprochent l'absence de sources : sans objet quand la
# trajectoire conclut précisément qu'aucune source pertinente n'existe.
_SOURCE_DEPENDENT_CODES = frozenset({
    ErrorCode.UNGROUNDED_ARTICLE,
    ErrorCode.UNGROUNDED_CITATION,
    ErrorCode.UNGROUNDED_URL,
    ErrorCode.OFFICIAL_TEXT_MISSING,
    ErrorCode.OFFICIAL_TEXT_NOT_REPRODUCED,
    ErrorCode.REQUIRED_TOOL_MISSING,
})

_CANNOT_CONCLUDE_MARKERS = (
    "aucune disposition", "aucun article", "pas couvert", "non couvert",
    "ne peux pas conclure", "ne peut pas conclure", "hors de ma couverture",
    "aucune source", "je ne peux pas répondre",
)

MIN_CANNOT_CONCLUDE_REASONING_CHARS = 80


def is_cannot_conclude(trajectory: TrainingTrajectory,
                       state: Optional[ResearchState] = None) -> bool:
    """La trajectoire se termine-t-elle par un aveu d'impossibilité ?"""
    stop_reason = getattr(state, "stop_reason", "") if state else ""
    if stop_reason in ("cannot_conclude", "jurisdiction_not_covered"):
        return True
    decision = getattr(state, "latest_decision", None) if state else None
    value = getattr(getattr(decision, "decision", None), "value", "")
    if value == "cannot_conclude":
        return True
    final = trajectory.final_answer().casefold()
    return any(marker in final for marker in _CANNOT_CONCLUDE_MARKERS)


def _cannot_conclude_reasoning_issues(
        trajectory: TrainingTrajectory) -> list[str]:
    """Ce qu'on continue d'exiger : une explication, pas juste un refus."""
    final = trajectory.final_answer().strip()
    if len(final) < MIN_CANNOT_CONCLUDE_REASONING_CHARS:
        return [tag(ErrorCode.FINAL_ANSWER_EMPTY,
                    "conclusion négative sans justification : "
                    f"{len(final)} caractères")]
    return []


def _jurisprudence_optional_for(request_type: str) -> bool:
    if request_type in NO_JURISPRUDENCE:
        return True
    rt = REQUEST_TYPES.get(request_type)
    if not rt:
        return False
    for step in rt.expected_route.steps:
        if step.tool == "search_quebec_jurisprudence" and step.optional:
            return True
    return False


def compute_acceptance_status(
    trajectory: TrainingTrajectory,
    validation: ValidationResult,
    legal_critic: Optional[CriticResult],
    agentic_critic: Optional[CriticResult],
    legal_min_score: float = 0.7,
    agentic_min_score: float = 0.7,
    state: Optional[ResearchState] = None,
) -> AcceptanceResult:
    """Single function determining accept/reject with full reasons.

    Returns an ``AcceptanceResult`` with:
    - ``accepted``: True only if no blocking errors remain
    - ``blocking_errors``: list of hard-failure reasons
    - ``warnings``: list of non-blocking issues (recorded, not blocking)
    - ``reasons``: combined list for backward compatibility
    - ``failed_checks``: structured dict of which checks failed
    """
    blocking_errors: list[str] = []
    warnings: list[str] = []
    failed_checks: dict[str, Any] = {}

    # ── 0. Issue « je ne peux pas conclure » ─────────────────────────────
    # Une trajectoire qui se termine honnêtement par « aucune disposition
    # pertinente » ou « juridiction non couverte » subissait les mêmes
    # contrôles de citations qu'une trajectoire aboutie, et se faisait
    # pénaliser pour absence de sources. C'est une issue valide de plein
    # droit; elle a ses propres critères.
    concluded_negatively = is_cannot_conclude(trajectory, state)

    # ── 1. Deterministic validation ──────────────────────────────────────
    det_blocking, det_warnings = _classify_errors(validation.errors)

    # Reclassify jurisprudence-absence as warning when optional
    if _jurisprudence_optional_for(trajectory.request_type):
        reclassified = []
        for err in det_blocking:
            if "jurisprudence" in err.casefold() and "inutile" in err.casefold():
                det_warnings.append(err)
            else:
                reclassified.append(err)
        det_blocking = reclassified

    blocking_errors.extend(det_blocking)
    warnings.extend(det_warnings)
    warnings.extend(validation.warnings)

    if det_blocking:
        failed_checks["deterministic_validation"] = det_blocking

    # ── 2. Jurisdiction consistency ──────────────────────────────────────
    jurisdiction_issues = _check_jurisdiction_consistency(trajectory, state)
    if jurisdiction_issues:
        blocking_errors.extend(jurisdiction_issues)
        failed_checks["jurisdiction_consistency"] = jurisdiction_issues

    # ── 3. Clarification consistency ─────────────────────────────────────
    clarification_issues = _check_clarification_consistency(trajectory, state)
    if clarification_issues:
        route_requires = (
            state.scenario.expected_route.requires_clarification
            if state and state.scenario else True
        )
        if route_requires:
            blocking_errors.extend(clarification_issues)
            failed_checks["clarification_consistency"] = clarification_issues
        else:
            warnings.extend(clarification_issues)

    # ── 4. Article grounding ─────────────────────────────────────────────
    grounding_issues = _check_article_grounding(trajectory)
    if grounding_issues:
        blocking_errors.extend(grounding_issues)
        failed_checks["article_grounding"] = grounding_issues

    # ── 4bis. Critères propres à `cannot_conclude` ───────────────────────
    if concluded_negatively:
        blocking_errors = [
            error for error in blocking_errors
            if extract_code(error) not in _SOURCE_DEPENDENT_CODES
        ]
        for key in ("article_grounding",):
            failed_checks.pop(key, None)
        # Ce qui reste exigible : un raisonnement présent et motivé.
        reasoning = _cannot_conclude_reasoning_issues(trajectory)
        if reasoning:
            blocking_errors.extend(reasoning)
            failed_checks["cannot_conclude_reasoning"] = reasoning

    # ── 5. Legal critic ──────────────────────────────────────────────────
    if legal_critic is not None:
        legal_ok = legal_critic.accepted and legal_critic.score >= legal_min_score
        if not legal_ok:
            reasons = _critic_failure_reasons(
                "legal_critic", legal_critic, legal_min_score)
            has_hard = any(
                label in (legal_critic.dimensional_scores.labels if legal_critic.dimensional_scores else [])
                for label in ("unsupported_claim", "fabricated_case_law_pattern",
                              "unsupported_deadline", "wrong_jurisdiction")
            )
            if has_hard or legal_critic.hard_failures:
                blocking_errors.extend(reasons)
                failed_checks["legal_critic"] = reasons
            else:
                warnings.extend(reasons)

    # ── 6. Agentic critic ────────────────────────────────────────────────
    if agentic_critic is not None:
        agentic_ok = agentic_critic.accepted and agentic_critic.score >= agentic_min_score
        if not agentic_ok:
            reasons = _critic_failure_reasons(
                "agentic_critic", agentic_critic, agentic_min_score)
            has_hard = any(
                label in (agentic_critic.dimensional_scores.labels if agentic_critic.dimensional_scores else [])
                for label in ("unsupported_claim", "fabricated_case_law_pattern",
                              "wrong_jurisdiction", "wrong_tool")
            )
            if has_hard or agentic_critic.hard_failures:
                blocking_errors.extend(reasons)
                failed_checks["agentic_critic"] = reasons
            else:
                warnings.extend(reasons)

    # ── Final decision ───────────────────────────────────────────────────
    accepted = len(blocking_errors) == 0
    all_reasons = list(dict.fromkeys(blocking_errors + warnings))

    return AcceptanceResult(
        accepted=accepted,
        reasons=all_reasons,
        blocking_errors=list(dict.fromkeys(blocking_errors)),
        warnings=list(dict.fromkeys(warnings)),
        failed_checks=failed_checks,
    )


# ── Sub-checks ───────────────────────────────────────────────────────────

_QUEBEC_MARKERS = re.compile(
    r"(?:code\s+civil|ccq|c\.c\.q|code\s+de\s+procédure\s+civile|cpc|"
    r"légisquébec|québ[eé]c|article\s+\d{1,4}(?:\.\d+)?\s+(?:du\s+)?(?:code\s+civil|ccq|c\.c\.q))",
    re.IGNORECASE,
)
_FEDERAL_MARKERS_RE = re.compile(
    r"(?:loi\s+fédérale|fédéral|canada|lrc|l\.r\.c|cour\s+suprême\s+du\s+canada|"
    r"cour\s+fédérale|scc|csc|failli|insolv|brevet|marque\s+de\s+commerce|maritime|"
    r"code\s+criminel|banque|bancaire)",
    re.IGNORECASE,
)
_OTHER_PROVINCIAL_RE = re.compile(
    r"(?:ontario|alberta|colombie-britannique|manitoba|saskatchewan|"
    r"nouveau-brunswick|nouvelle-écosse|île-du-prince-édouard|terre-neuve)",
    re.IGNORECASE,
)
_MUNICIPAL_RE = re.compile(
    r"(?:municipal|arrondissement|ville\s+de|règlement\s+municipal)",
    re.IGNORECASE,
)

ARTICLE_CITATION_RE = re.compile(
    r"\barticle\s+(\d{1,4}(?:\.\d+)?)\b", re.IGNORECASE)


def _infer_jurisdiction_from_query(query: str) -> str:
    has_qc = bool(_QUEBEC_MARKERS.search(query))
    has_fed = bool(_FEDERAL_MARKERS_RE.search(query))
    if has_qc and has_fed:
        return "undetermined"
    if has_qc:
        return "supported_quebec"
    if has_fed:
        return "supported_federal"
    if _OTHER_PROVINCIAL_RE.search(query):
        return "supported_other_canadian"
    if _MUNICIPAL_RE.search(query):
        return "municipal_coverage_uncertain"
    return "undetermined"


def _check_jurisdiction_consistency(
    trajectory: TrainingTrajectory,
    state: Optional[ResearchState] = None,
) -> list[str]:
    issues: list[str] = []

    # Get scenario data from state if available
    scenario = state.scenario if state else None
    query = scenario.user_query if scenario else ""
    if not query:
        query = next(
            (m.content for m in trajectory.messages if m.role.value == "user"), "")

    declared_status = (
        scenario.jurisdiction_status if scenario else ""
    )
    inferred = _infer_jurisdiction_from_query(query)

    # If query clearly mentions CCQ/CPC but jurisdiction is not quebec
    if inferred == "supported_quebec" and declared_status not in (
        "supported_quebec", "undetermined", "",
    ):
        issues.append(
            f"jurisdiction_mismatch: la question mentionne le droit québécois "
            f"mais jurisdiction_status={declared_status}")

    # If query clearly mentions federal law but jurisdiction is quebec
    if inferred == "supported_federal" and declared_status == "supported_quebec":
        # Only flag if no Quebec markers are also present
        if not _QUEBEC_MARKERS.search(query):
            issues.append(
                f"jurisdiction_mismatch: la question mentionne le droit fédéral "
                f"mais jurisdiction_status={declared_status}")

    # Check tool usage matches jurisdiction
    if scenario and declared_status == "supported_federal":
        from .taxonomy import QUEBEC_ONLY_TOOLS
        qc_tools_used = [
            o.tool_name for o in (state.tool_history if state else [])
            if o.tool_name in QUEBEC_ONLY_TOOLS
        ]
        if qc_tools_used and scenario.request_type != "comparative_law":
            issues.append(
                f"jurisdiction_mismatch: outils québécois {qc_tools_used} utilisés "
                f"avec jurisdiction_status=supported_federal")

    return issues


def _check_clarification_consistency(
    trajectory: TrainingTrajectory,
    state: Optional[ResearchState] = None,
) -> list[str]:
    issues: list[str] = []

    scenario = state.scenario if state else None
    if not scenario:
        return issues

    stage = scenario.clarification_stage
    synth_answer = scenario.synthetic_clarification_answer
    clar_answer = scenario.clarification_answer

    # Find actual clarification in messages
    has_clarification_question = False
    has_clarification_response = False
    for i, msg in enumerate(trajectory.messages):
        if (msg.role.value == "assistant" and msg.content.rstrip().endswith("?")
                and i < len(trajectory.messages) - 1
                and trajectory.messages[i + 1].role.value == "user"):
            has_clarification_question = True
            has_clarification_response = True
            break

    if stage == "none":
        if has_clarification_question:
            # Clarification present but stage says none — downgrade to warning
            # (the model may legitimately ask clarification sometimes)
            pass
        if synth_answer or clar_answer:
            issues.append(
                "clarification_inconsistency: clarification_stage=none "
                "mais synthetic_clarification_answer n'est pas null")
    elif stage in ("before_search", "after_initial_research"):
        if not has_clarification_question and (synth_answer or clar_answer):
            issues.append(
                f"clarification_inconsistency: clarification_stage={stage} "
                "avec une réponse synthétique mais aucune question de "
                "clarification dans les messages")

    return issues


def _check_article_grounding(trajectory: TrainingTrajectory) -> list[str]:
    """Check that articles cited in the final answer were actually retrieved."""
    issues: list[str] = []
    final = trajectory.final_answer().strip()
    if not final:
        return issues

    cited_articles = set(ARTICLE_CITATION_RE.findall(final))
    if not cited_articles:
        return issues

    # Build set of articles that appear in tool results
    retrieved_text = "\n".join(
        o.normalized_response for o in trajectory.tool_trace
        if o.ok and o.tool_name not in (
            "semantic_search_ccq", "semantic_search_cpc")
    ).casefold()

    for article in cited_articles:
        if not re.search(
            rf"\barticle\s+{re.escape(article)}\b",
            retrieved_text, re.IGNORECASE,
        ):
            issues.append(f"unsupported_article: article {article} cité "
                          "dans la réponse mais absent des résultats d'outils")

    return issues


def _critic_failure_reasons(
    label: str, result: CriticResult, minimum: float,
) -> list[str]:
    reasons: list[str] = []
    if not result.accepted:
        reasons.append(f"{label}: décision rejected")
    if result.score < minimum:
        reasons.append(
            f"{label}: score {result.score:.2f} < seuil {minimum:.2f}")
    reasons.extend(f"{label}: {issue}" for issue in result.issues)
    reasons.extend(
        f"{label}: affirmation non étayée: {claim}"
        for claim in result.unsupported_claims)
    reasons.extend(
        f"{label}: source manquante: {source}"
        for source in result.missing_sources)
    return reasons
