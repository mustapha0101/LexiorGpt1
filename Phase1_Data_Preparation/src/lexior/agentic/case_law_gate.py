# -*- coding: utf-8 -*-

"""
Deterministic relevance gate for jurisprudence (case law) search results.

Classifies each raw result from ``search_quebec_jurisprudence`` against the
target provisions, jurisdiction, and user situation, then filters down to the
usable subset.  No LLM calls -- pure regex + heuristic scoring.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .citations import QUEBEC_CITATION_RE
from .schemas import CaseRelevanceResult, CaseLawSearchStatus

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_QC_CITATION = QUEBEC_CITATION_RE
_RE_CASE_NAME = re.compile(
    r"([A-ZÀ-Ÿ][\w'-]+)\s+c\.\s+([A-ZÀ-Ÿ][\w'-]+)"
)
_RE_ARTICLE_REF = re.compile(
    r"\barticle\s+(\d{1,4}(?:\.\d+)?)\b", re.IGNORECASE
)

_RESULT_SEPARATOR = re.compile(r"\n-{3,}\n|\n\d+\.\s")
_URL_RE = re.compile(r"https?://[^\s\"',\]\)]+")


def _is_fetchable_quebec_url(url: str) -> bool:
    folded = (url or "").casefold()
    return ("soquij.qc.ca" in folded or "soquij" in folded
            or "canlii.org" in folded and any(token in folded for token in (
                "/qccq/", "/qcca/", "/qccs/", "/qc/", "qccq", "qcca", "qccs")))


def _is_candidate_chunk(chunk: str, url: str) -> bool:
    folded = chunk.casefold()
    if not _is_fetchable_quebec_url(url):
        return False
    return not any(marker in folded for marker in (
        "aucun resultat", "aucun résultat", "no result", "federal court",
        "cour supreme des etats-unis", "unrelated",
    ))


# ---------------------------------------------------------------------------
# Single-result classifier
# ---------------------------------------------------------------------------

def classify_case_result(
    result_text: str,
    target_articles: list[str],
    user_situation: str,
    source_url: str = "",
    jurisdiction: str = "Québec",
) -> CaseRelevanceResult:
    """Classify a single jurisprudence search result for relevance.

    Parameters
    ----------
    result_text:
        Raw text of one search result (a single case).
    target_articles:
        Article numbers the trajectory is about (e.g. ``["1457", "1458"]``).
    user_situation:
        Free-text description of the user's legal issue, used for keyword
        matching against the result.
    jurisdiction:
        Expected jurisdiction label (default ``"Québec"``).

    Returns
    -------
    CaseRelevanceResult
        Populated dataclass with usability flag and relevance score.
    """

    text_lower = result_text.lower()

    # -- Jurisdiction check --------------------------------------------------
    citation_match = _RE_QC_CITATION.search(result_text)
    correct_jurisdiction = citation_match is not None

    citation = ""
    court = ""
    date = ""
    if citation_match:
        date = citation_match.group(1)
        court = citation_match.group(2)
        citation = citation_match.group(0)

    # -- Case name -----------------------------------------------------------
    name_match = _RE_CASE_NAME.search(result_text)
    case_name = name_match.group(0) if name_match else ""

    # -- Target provision check ----------------------------------------------
    found_articles = _RE_ARTICLE_REF.findall(result_text)
    mentions_target_provision = any(
        art in target_articles for art in found_articles
    )
    target_provisions = [a for a in found_articles if a in target_articles]

    # -- Legal-issue keyword match -------------------------------------------
    situation_keywords = _extract_keywords(user_situation)
    hit_count = sum(1 for kw in situation_keywords if kw in text_lower)
    matches_legal_issue = hit_count >= max(1, len(situation_keywords) // 3)

    # -- Usability decision --------------------------------------------------
    usable = correct_jurisdiction and mentions_target_provision

    # -- Relevance score (0.0 .. 1.0) ----------------------------------------
    score = 0.0
    if correct_jurisdiction:
        score += 0.3
    if mentions_target_provision:
        score += 0.35
    if matches_legal_issue:
        score += 0.25
    if case_name:
        score += 0.05
    if date:
        score += 0.05
    score = min(score, 1.0)

    # -- Build reason string -------------------------------------------------
    reasons: list[str] = []
    if not correct_jurisdiction:
        reasons.append("no Quebec court citation found")
    if not mentions_target_provision:
        reasons.append(
            f"does not mention target articles {target_articles}"
        )
    if not matches_legal_issue:
        reasons.append("weak keyword match with user situation")
    reason = "; ".join(reasons) if reasons else "all checks passed"

    return CaseRelevanceResult(
        source_type="jurisprudence",
        correct_jurisdiction=correct_jurisdiction,
        mentions_target_provision=mentions_target_provision,
        matches_legal_issue=matches_legal_issue,
        usable=usable,
        reason=reason,
        case_name=case_name,
        citation=citation,
        court=court,
        date=date,
        target_provisions=target_provisions,
        source_url=source_url,
        relevance_score=round(score, 2),
    )


# ---------------------------------------------------------------------------
# Batch gate
# ---------------------------------------------------------------------------

def gate_search_results(
    results_text: str,
    target_articles: list[str],
    user_situation: str,
    source_urls: list[str] | None = None,
) -> Tuple[List[CaseRelevanceResult], CaseLawSearchStatus]:
    """Split *results_text* into individual results and classify each one.

    Parameters
    ----------
    results_text:
        Full text returned by ``search_quebec_jurisprudence``, possibly
        containing multiple results separated by ``---`` lines or numbered
        items.
    target_articles:
        Article numbers the trajectory targets.
    user_situation:
        Free-text description of the legal issue.

    Returns
    -------
    tuple[list[CaseRelevanceResult], CaseLawSearchStatus]
        Classified results and an overall status enum.
    """

    if not results_text or not results_text.strip():
        return [], CaseLawSearchStatus.empty

    chunks = _split_results(results_text)
    if not chunks:
        return [], CaseLawSearchStatus.empty

    classified: list[CaseRelevanceResult] = []
    available_urls = list(source_urls or [])
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        urls = _URL_RE.findall(chunk)
        url = urls[0] if urls else (
            available_urls[0] if len(available_urls) == 1 else "")
        classified.append(
            classify_case_result(chunk, target_articles, user_situation,
                                 source_url=url)
        )

    if not classified:
        return [], CaseLawSearchStatus.empty

    has_usable = any(r.usable for r in classified)
    has_fetchable_candidate = any(
        (not result.usable) and _is_candidate_chunk(
            chunk, result.source_url)
        for chunk, result in zip(chunks, classified))
    if has_usable:
        status = CaseLawSearchStatus.usable
    elif has_fetchable_candidate:
        status = CaseLawSearchStatus.candidate_pending_fetch
    else:
        status = CaseLawSearchStatus.irrelevant

    return classified, status


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------

def filter_usable_cases(
    results: list[CaseRelevanceResult],
) -> list[CaseRelevanceResult]:
    """Return only usable cases, sorted by *relevance_score* descending."""

    return sorted(
        [r for r in results if r.usable],
        key=lambda r: r.relevance_score,
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_results(text: str) -> list[str]:
    """Split raw search output into individual result chunks."""

    parts = _RESULT_SEPARATOR.split(text)
    # If the regex didn't produce multiple chunks, fall back to double-newline.
    if len(parts) <= 1:
        parts = text.split("\n\n")
    return [p for p in parts if p and p.strip()]


def _extract_keywords(text: str, min_len: int = 4) -> list[str]:
    """Produce lowercase keywords from *text*, dropping short words."""

    words = re.findall(r"[a-zA-ZÀ-ɏ]{2,}", text.lower())
    return [w for w in words if len(w) >= min_len]


def is_verified_quebec_decision(text: str) -> bool:
    """Vrai seulement si un texte contient une citation de décision québécoise."""
    if not (text or "").strip() or not _RE_QC_CITATION.search(text):
        return False
    folded = text.casefold()
    return len(text.strip()) >= 120 and not any(marker in folded for marker in (
        "aucune décision", "no decision", "résultat de recherche", "search result",
    ))
