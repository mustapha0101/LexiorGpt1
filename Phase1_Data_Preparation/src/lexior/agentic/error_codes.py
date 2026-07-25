# -*- coding: utf-8 -*-
"""Codes d'erreur de validation — la décision ne dépend plus du français.

Avant : ``acceptance._is_blocking`` reclassait des messages d'erreur en
français libre à coups d'expressions régulières. Renommer un message
changeait silencieusement le critère d'acceptation du dataset.

Maintenant : un validateur émet un CODE, le texte français ne sert plus
qu'à l'affichage. La classification bloquant / non bloquant se fait sur le
code, ici, en un seul endroit.

Format sur le fil : ``"[code] message en français"``. Les messages
produits ailleurs (critics LLM notamment) n'ont pas de code et retombent
sur la classification historique par motifs.
"""

from __future__ import annotations

import re
from enum import Enum


class ErrorCode(str, Enum):
    """Codes émis par ``validators.py``."""

    # ── Décision du planner ──────────────────────────────────────────────
    DECISION_MISSING_TOOL = "decision_missing_tool"
    DECISION_UNEXPECTED_TOOL = "decision_unexpected_tool"
    CLARIFICATION_EMPTY = "clarification_empty"

    # ── Route et outils ──────────────────────────────────────────────────
    TOOL_FORBIDDEN = "tool_forbidden"
    TOOL_OFF_ROUTE = "tool_off_route"
    REQUIRED_TOOL_MISSING = "required_tool_missing"
    TOOL_ORDER_WRONG = "tool_order_wrong"
    TOOL_CALL_LOOP = "tool_call_loop"
    TOO_MANY_TOOL_CALLS = "too_many_tool_calls"
    USELESS_JURISPRUDENCE_SEARCH = "useless_jurisprudence_search"

    # ── Forme des messages ───────────────────────────────────────────────
    TOOL_CALL_MALFORMED = "tool_call_malformed"
    TOOL_CALL_UNPAIRED = "tool_call_unpaired"
    TOOL_MESSAGE_UNPAIRED = "tool_message_unpaired"
    TOOL_MESSAGE_MISMATCH = "tool_message_mismatch"
    TOOL_RESPONSE_FROM_ASSISTANT = "tool_response_from_assistant"
    TEMP_PATH_LEAK = "temp_path_leak"
    FINAL_ANSWER_WRAPPED = "final_answer_wrapped"
    FINAL_ANSWER_EMPTY = "final_answer_empty"

    # ── Observations ─────────────────────────────────────────────────────
    OBSERVATION_FABRICATED = "observation_fabricated"
    OBSERVATION_MOCK_FORBIDDEN = "observation_mock_forbidden"
    OBSERVATION_UNHASHED = "observation_unhashed"
    OBSERVATION_UNLINKED = "observation_unlinked"
    TRACE_MESSAGE_MISMATCH = "trace_message_mismatch"

    # ── Ancrage dans les sources ─────────────────────────────────────────
    UNGROUNDED_URL = "ungrounded_url"
    UNGROUNDED_CITATION = "ungrounded_citation"
    UNGROUNDED_ARTICLE = "ungrounded_article"
    OFFICIAL_TEXT_MISSING = "official_text_missing"
    OFFICIAL_TEXT_NOT_REPRODUCED = "official_text_not_reproduced"
    ANSWER_FROM_MEMORY = "answer_from_memory"
    UNJUSTIFIED_CERTAINTY = "unjustified_certainty"

    # ── Langue et unicité ────────────────────────────────────────────────
    LANGUAGE_MISMATCH = "language_mismatch"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"

    # ── Avertissements ───────────────────────────────────────────────────
    QUERY_TOO_SHORT = "query_too_short"
    QUERY_TOO_GENERIC = "query_too_generic"
    QUERY_IMPROVABLE = "query_improvable"
    THINKING_TOO_LONG = "thinking_too_long"
    TOOL_CALL_WITH_PROSE = "tool_call_with_prose"


# Une trajectoire porteuse d'un de ces codes ne peut pas entrer dans le
# dataset. Tout code absent d'ici est un simple avertissement.
BLOCKING_CODES: frozenset[ErrorCode] = frozenset({
    ErrorCode.DECISION_MISSING_TOOL,
    ErrorCode.DECISION_UNEXPECTED_TOOL,
    ErrorCode.CLARIFICATION_EMPTY,
    ErrorCode.TOOL_FORBIDDEN,
    ErrorCode.TOOL_OFF_ROUTE,
    ErrorCode.REQUIRED_TOOL_MISSING,
    ErrorCode.TOOL_ORDER_WRONG,
    ErrorCode.TOOL_CALL_LOOP,
    ErrorCode.TOO_MANY_TOOL_CALLS,
    ErrorCode.TOOL_CALL_MALFORMED,
    ErrorCode.TOOL_CALL_UNPAIRED,
    ErrorCode.TOOL_MESSAGE_UNPAIRED,
    ErrorCode.TOOL_MESSAGE_MISMATCH,
    ErrorCode.TOOL_RESPONSE_FROM_ASSISTANT,
    ErrorCode.TEMP_PATH_LEAK,
    ErrorCode.FINAL_ANSWER_WRAPPED,
    ErrorCode.FINAL_ANSWER_EMPTY,
    ErrorCode.OBSERVATION_FABRICATED,
    ErrorCode.OBSERVATION_MOCK_FORBIDDEN,
    ErrorCode.OBSERVATION_UNHASHED,
    ErrorCode.OBSERVATION_UNLINKED,
    ErrorCode.TRACE_MESSAGE_MISMATCH,
    ErrorCode.UNGROUNDED_URL,
    ErrorCode.UNGROUNDED_CITATION,
    ErrorCode.UNGROUNDED_ARTICLE,
    ErrorCode.OFFICIAL_TEXT_MISSING,
    ErrorCode.OFFICIAL_TEXT_NOT_REPRODUCED,
    ErrorCode.ANSWER_FROM_MEMORY,
    ErrorCode.EXACT_DUPLICATE,
    ErrorCode.NEAR_DUPLICATE,
    # Un dataset bilingue ne peut pas accepter une réponse dans la mauvaise
    # langue : c'était un simple avertissement, c'est un rejet ferme.
    ErrorCode.LANGUAGE_MISMATCH,
})

_TAG_RE = re.compile(r"^\[([a-z_]+)\]\s*")


def tag(code: ErrorCode, message: str) -> str:
    """Préfixe un message d'affichage par son code."""
    return f"[{code.value}] {message}"


def extract_code(error: str) -> ErrorCode | None:
    """Code porté par un message, ou ``None`` s'il n'en porte pas."""
    match = _TAG_RE.match(error or "")
    if not match:
        return None
    try:
        return ErrorCode(match.group(1))
    except ValueError:
        return None


def strip_code(error: str) -> str:
    """Message sans son code, pour l'affichage à un humain."""
    return _TAG_RE.sub("", error or "")
