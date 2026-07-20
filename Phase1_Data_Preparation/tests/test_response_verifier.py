# -*- coding: utf-8 -*-
"""Tests du vérificateur déterministe des réponses MCP."""

import json

from agentic_generation.response_verifier import verify_observation
from agentic_generation.schemas import ToolObservation


def _obs(tool_name, normalized_response, **kwargs):
    defaults = dict(
        tool_name=tool_name, server="test", arguments={},
        raw_response=None, normalized_response=normalized_response,
        ok=True, mock=True,
    )
    defaults.update(kwargs)
    return ToolObservation(**defaults).finalize_hash()


# ---------------------------------------------------------------------------
# search_legal_documents — filtrage provincial
# ---------------------------------------------------------------------------

def test_search_legal_documents_keeps_federal_results():
    data = {"results": [
        {"dataset": "LEGISLATION-FED", "name_fr": "Loi sur les banques"},
        {"dataset": "LEGISLATION-FED", "name_fr": "Loi sur la faillite"},
    ]}
    obs = _obs("search_legal_documents", json.dumps(data))
    cleaned, issues = verify_observation(obs)
    assert cleaned.ok
    assert not issues
    assert "Loi sur les banques" in cleaned.normalized_response


def test_search_legal_documents_filters_provincial():
    data = {"results": [
        {"dataset": "LEGISLATION-FED", "name_fr": "Loi sur les banques"},
        {"dataset": "LEGISLATION-QC", "name_fr": "Code civil du Québec"},
    ]}
    obs = _obs("search_legal_documents", json.dumps(data))
    cleaned, issues = verify_observation(obs)
    assert cleaned.ok
    assert any("non fédéral" in i for i in issues)
    parsed = json.loads(cleaned.normalized_response)
    assert len(parsed["results"]) == 1
    assert parsed["results"][0]["dataset"] == "LEGISLATION-FED"


def test_search_legal_documents_all_provincial_becomes_error():
    data = {"results": [
        {"dataset": "LEGISLATION-QC", "name_fr": "Code civil du Québec"},
        {"dataset": "LEGISLATION-ON", "name_fr": "Ontario statute"},
    ]}
    obs = _obs("search_legal_documents", json.dumps(data))
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert any("FATAL" in i for i in issues)
    assert "aucun résultat fédéral" in cleaned.error


def test_search_legal_documents_court_datasets_are_federal():
    data = {"results": [
        {"dataset": "SCC", "name_en": "R v Smith"},
        {"dataset": "FCA", "name_en": "Appeal case"},
    ]}
    obs = _obs("search_legal_documents", json.dumps(data))
    cleaned, issues = verify_observation(obs)
    assert cleaned.ok
    assert not issues


# ---------------------------------------------------------------------------
# fetch_document — abrogation fédérale
# ---------------------------------------------------------------------------

def test_fetch_document_clean_passes():
    text = json.dumps({
        "citation_fr": "LRC 1985, c B-3",
        "unofficial_text_fr": "49. (1) Le débiteur insolvable " + "x" * 200,
    })
    obs = _obs("fetch_document", text)
    cleaned, issues = verify_observation(obs)
    assert cleaned.ok
    assert not any("FATAL" in i for i in issues)


def test_fetch_document_repealed_is_rejected():
    text = "[Abrogé, 2017, ch. 33, art. 228]"
    obs = _obs("fetch_document", text)
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert any("abrogé" in i for i in issues)


def test_fetch_document_empty_is_rejected():
    obs = _obs("fetch_document", "   ")
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert any("vide" in i for i in issues)


# ---------------------------------------------------------------------------
# get_ccq_articles / get_cpc_articles — articles québécois
# ---------------------------------------------------------------------------

def test_qc_article_live_passes():
    text = (
        "Article 1457\n"
        "Toute personne a le devoir de respecter les règles de conduite qui, "
        "suivant les circonstances, les usages ou la loi, s'imposent à elle."
    )
    obs = _obs("get_ccq_articles", text)
    cleaned, issues = verify_observation(obs)
    assert cleaned.ok
    assert not issues


def test_qc_article_repealed_is_rejected():
    obs = _obs("get_ccq_articles", "(Abrogé).")
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert any("abrogé" in i for i in issues)


def test_qc_article_omitted_is_rejected():
    obs = _obs("get_cpc_articles", "(Omis).")
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert any("omis" in i for i in issues)


def test_qc_article_spent_is_rejected():
    obs = _obs("get_ccq_articles", "(Modification intégrée au c. B-1, a. 125).")
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert any("épuisé" in i for i in issues)


def test_qc_article_empty_is_rejected():
    obs = _obs("get_ccq_articles", "")
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert any("vide" in i for i in issues)


def test_qc_article_too_short_is_rejected():
    obs = _obs("get_ccq_articles", "Art. 99")
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok


# ---------------------------------------------------------------------------
# Outils non vérifiés — pass-through
# ---------------------------------------------------------------------------

def test_unknown_tool_passes_through():
    obs = _obs("coverage", '{"results": []}')
    cleaned, issues = verify_observation(obs)
    assert cleaned.ok
    assert not issues
    assert cleaned.normalized_response == obs.normalized_response


def test_already_failed_observation_passes_through():
    obs = _obs("get_ccq_articles", "(Abrogé).", ok=False,
               error="panne MCP simulée")
    cleaned, issues = verify_observation(obs)
    assert not cleaned.ok
    assert not issues


# ---------------------------------------------------------------------------
# Recherche québécoise — vérification légère
# ---------------------------------------------------------------------------

def test_qc_search_empty_is_flagged():
    obs = _obs("search_ccq_keywords", "")
    _, issues = verify_observation(obs)
    assert any("vide" in i for i in issues)


def test_qc_search_with_content_passes():
    obs = _obs("search_ccq_keywords",
               "Article 1726 — garantie contre les vices cachés du bien vendu.")
    _, issues = verify_observation(obs)
    assert not issues
