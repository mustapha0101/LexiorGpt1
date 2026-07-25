# -*- coding: utf-8 -*-
"""Unified search-result classification with 8 statuses.

Consolidates the classification logic from:
  - ``validators.classify_search_result``  (basic tool-level classification)
  - ``case_law_gate.classify_case_result`` (jurisprudence relevance scoring)

into one ``ResultClassifier`` class that returns ``SearchResultStatus`` values.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from lexior.agentic.citations import CASE_CITATION_RE, CASE_NAME_RE
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

_CASE_CITATION_RE = CASE_CITATION_RE
_CASE_NAME_RE = CASE_NAME_RE
_LEGISLATION_KEYWORDS = ("loi sur", "règlement sur", "code civil",
                         "code de procédure")
_TRUNCATION_MARKERS = ("...", "[suite]", "[truncated]", "[tronqué]")

_QC_STUB_RE = re.compile(
    r"^\s*\(\s*(?:abrog|omis|modification)", re.IGNORECASE)
_FED_REPEAL_RE = re.compile(r"\[\s*(?:Abrog|Repealed)", re.IGNORECASE)

MIN_CONTENT_CHARS = 30

# Outils dont la réponse ne contient aucun texte d'article (uniquement des
# libellés et des scores) : une comparaison lexicale avec la question n'y
# veut rien dire. Leur pertinence est gouvernée par les planchers absolus de
# ``legal_rag`` (RAGConfig.min_dense_score / min_hybrid_score).
_NO_CONTENT_TOOLS = {"semantic_search_ccq", "semantic_search_cpc", "coverage"}

# Mots-outils français d'au moins quatre caractères. Les plus courts sont
# déjà écartés par la longueur minimale.
_STOPWORDS = frozenset("""
alors apres aussi autre autres avec avoir avait avais cela celle celles celui
cent cependant certain certaine ces cet cette chaque chez comme comment dans
depuis donc dont elle elles encore entre etait etaient etais este etre eux
fait faire faut leur leurs lorsque mais meme memes mien moins notre nous
parce pendant peut peuvent peux plus pour pourquoi pouvoir presque puis
quand quel quelle quelles quels quelque qu_il sans sauf selon seulement
soit sont sous suis sur tous tout toute toutes tres trop vers votre vous
etes ainsi bien beaucoup deja dire doit doivent donne encore enfin ici
jamais juste maintenant moi mon ma mes ton ta tes son sa ses non oui
""".split())


def _content_stems(text: str) -> set[str]:
    """Mots de contenu repliés et tronqués, pour une comparaison grossière.

    La troncature à six caractères rapproche « locataire » et « locatif »
    sans imposer de lemmatiseur.
    """
    normalized = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in normalized if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", folded.lower())
    return {token[:6] for token in tokens
            if len(token) >= 4 and token not in _STOPWORDS}


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
        user_query: str = "",
    ) -> SearchResultStatus:
        """Classify a single tool result.

        ``user_query`` reste optionnel pour ne pas casser les appelants
        existants, mais sans lui ``usable`` ne signifie que « bien formé et
        non vide » — jamais « répond à la question ».
        """
        status = self._classify_shape(tool_name, response, ok)
        if (status == SearchResultStatus.usable
                and self.is_off_topic(tool_name, response, user_query)):
            return SearchResultStatus.irrelevant
        return status

    def is_off_topic(
        self, tool_name: str, response: str, user_query: str,
    ) -> bool:
        """Vrai quand le résultat ne partage aucun mot de contenu avec la
        question.

        Critère volontairement conservateur : un recouvrement nul est un
        signal grossier de hors-sujet, pas une mesure de pertinence fine.
        """
        if not user_query or tool_name in _NO_CONTENT_TOOLS:
            return False
        question_stems = _content_stems(user_query)
        if not question_stems:
            return False
        return not question_stems.intersection(_content_stems(response))

    def _classify_shape(
        self, tool_name: str, response: str, ok: bool,
    ) -> SearchResultStatus:
        """Classification de forme : succès technique, vide, tronqué, périmé."""
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
        self, observation: ToolObservation, user_query: str = "",
    ) -> SearchResultStatus:
        """Classify a ToolObservation directly."""
        return self.classify(
            observation.tool_name,
            observation.normalized_response,
            observation.ok,
            observation.error,
            user_query=user_query,
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
