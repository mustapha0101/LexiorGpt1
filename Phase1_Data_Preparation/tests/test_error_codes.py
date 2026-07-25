# -*- coding: utf-8 -*-
"""L'acceptation dépend d'un code, plus du texte français d'un message."""

from __future__ import annotations

import pytest

from lexior.agentic.acceptance import _is_blocking
from lexior.agentic.error_codes import (
    BLOCKING_CODES,
    ErrorCode,
    extract_code,
    strip_code,
    tag,
)
from lexior.agentic.validators import FingerprintIndex


# ── Codes ────────────────────────────────────────────────────────────────


def test_a_tagged_message_carries_its_code():
    message = tag(ErrorCode.UNGROUNDED_ARTICLE,
                  "article 1457 absent des réponses d'outils")

    assert extract_code(message) is ErrorCode.UNGROUNDED_ARTICLE
    assert strip_code(message) == "article 1457 absent des réponses d'outils"


def test_an_untagged_message_has_no_code():
    assert extract_code("unsupported_claim: la réponse invente un délai") is None


def test_an_unknown_code_is_not_invented():
    assert extract_code("[code_inexistant] message") is None


def test_rewording_a_message_does_not_change_the_verdict():
    """Le point de tout l'exercice : le français ne décide plus."""
    before = tag(ErrorCode.UNGROUNDED_ARTICLE,
                 "article 1457 absent des réponses d'outils")
    after = tag(ErrorCode.UNGROUNDED_ARTICLE,
                "aucune source récupérée ne contient l'article 1457")

    assert _is_blocking(before) is _is_blocking(after) is True


# ── Classification ───────────────────────────────────────────────────────


@pytest.mark.parametrize("code", sorted(BLOCKING_CODES, key=lambda c: c.value))
def test_every_blocking_code_blocks(code):
    assert _is_blocking(tag(code, "message d'affichage"))


@pytest.mark.parametrize("code", [
    ErrorCode.QUERY_TOO_SHORT,
    ErrorCode.QUERY_TOO_GENERIC,
    ErrorCode.QUERY_IMPROVABLE,
    ErrorCode.THINKING_TOO_LONG,
    ErrorCode.TOOL_CALL_WITH_PROSE,
    ErrorCode.USELESS_JURISPRUDENCE_SEARCH,
    ErrorCode.UNJUSTIFIED_CERTAINTY,
])
def test_advisory_codes_do_not_block(code):
    assert not _is_blocking(tag(code, "message d'affichage"))


def test_language_mismatch_is_now_a_hard_rejection():
    """Un dataset bilingue ne peut pas accepter la mauvaise langue."""
    assert ErrorCode.LANGUAGE_MISMATCH in BLOCKING_CODES
    assert _is_blocking(tag(ErrorCode.LANGUAGE_MISMATCH,
                            "language mismatch : réponse en anglais"))


def test_blocking_patterns_win_over_non_blocking_ones():
    """Un message des critics correspondant aux deux familles bloque.

    L'ordre inverse laissait passer en avertissement un message qui
    correspondait à un motif bloquant.
    """
    message = ("requête améliorable, et par ailleurs URL absente des "
               "réponses d'outils")

    assert _is_blocking(message)


def test_an_unknown_critic_message_blocks_by_default():
    assert _is_blocking("quelque chose d'inattendu s'est produit")


# ── Empreintes anti-doublon ──────────────────────────────────────────────


def test_the_index_detects_an_exact_duplicate():
    index = FingerprintIndex()
    index.add("mon locateur refuse mon chat|get_ccq_articles")

    assert "mon locateur refuse mon chat|get_ccq_articles" in index


def test_the_index_detects_a_near_duplicate():
    index = FingerprintIndex()
    index.add("locateur refuse chat dans appartement bail signe montreal "
              "quebec|get_ccq_articles")

    assert index.near_duplicate(
        "locateur refuse chat dans appartement bail signe montreal ville "
        "quebec|get_ccq_articles", 0.80)


def test_a_different_question_is_not_a_near_duplicate():
    index = FingerprintIndex()
    index.add("locateur refuse chat appartement|get_ccq_articles")

    assert not index.near_duplicate(
        "vice cache maison sous-sol inonde vendeur|get_ccq_articles", 0.90)


def test_the_threshold_is_honoured():
    index = FingerprintIndex()
    index.add("a b c d e f g h i j")

    candidate = "a b c d e f g h i k"
    assert index.near_duplicate(candidate, 0.80)
    assert not index.near_duplicate(candidate, 0.95)


def test_tokens_are_precomputed_once_per_fingerprint():
    """Le coût ne doit plus être quadratique en re-découpage de chaînes."""
    index = FingerprintIndex()
    for number in range(50):
        index.add(f"question numero {number} distincte|get_ccq_articles")

    assert len(index._tokens) == 50
    assert all(isinstance(tokens, frozenset)
               for tokens in index._tokens.values())


def test_an_existing_set_can_seed_the_index():
    index = FingerprintIndex({"empreinte a", "empreinte b"})

    assert "empreinte a" in index
    assert index.near_duplicate("empreinte a", 0.90)
