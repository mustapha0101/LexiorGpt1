# -*- coding: utf-8 -*-
"""Tests for ResultClassifier — unified 8-status classification."""

import pytest

from agentic_generation.schemas import (
    CaseLawSearchStatus,
    SearchResultStatus,
    ToolObservation,
)
from lexior.agent_graph.result_classifier import ResultClassifier


@pytest.fixture
def classifier():
    return ResultClassifier()


# ── Tool-error status ────────────────────────────────────────────────────


class TestToolError:
    def test_failed_tool(self, classifier):
        assert classifier.classify(
            "semantic_search_ccq", "", ok=False, error="timeout"
        ) == SearchResultStatus.tool_error

    def test_failed_no_message(self, classifier):
        assert classifier.classify(
            "get_ccq_articles", "", ok=False
        ) == SearchResultStatus.tool_error


# ── Empty status ─────────────────────────────────────────────────────────


class TestEmpty:
    @pytest.mark.parametrize("response", ["", "[]", "{}", "null"])
    def test_empty_values(self, classifier, response):
        assert classifier.classify(
            "semantic_search_ccq", response, ok=True
        ) == SearchResultStatus.empty

    def test_very_short(self, classifier):
        assert classifier.classify(
            "semantic_search_ccq", "too short", ok=True
        ) == SearchResultStatus.empty


# ── Jurisprudence classification ─────────────────────────────────────────


class TestJurisprudence:
    def test_usable_with_citation(self, classifier):
        text = (
            "Dupont c. Martin — 2020 QCCS 1234. "
            "Le tribunal a conclu que l'article 1457 s'applique."
        )
        assert classifier.classify(
            "search_quebec_jurisprudence", text, ok=True
        ) == SearchResultStatus.usable

    def test_irrelevant_no_citation(self, classifier):
        text = "Ce document traite de questions administratives sans citation."
        assert classifier.classify(
            "search_quebec_jurisprudence", text, ok=True
        ) == SearchResultStatus.irrelevant

    def test_wrong_document_type(self, classifier):
        text = (
            "Loi sur la protection du consommateur, "
            "règlement sur les normes du travail"
        )
        assert classifier.classify(
            "search_quebec_jurisprudence", text, ok=True
        ) == SearchResultStatus.wrong_document_type


# ── Article classification ───────────────────────────────────────────────


class TestArticle:
    def test_usable_article(self, classifier):
        text = (
            "1457. Toute personne a le devoir de respecter les règles "
            "de conduite qui, suivant les circonstances, les usages ou "
            "la loi, s'imposent à elle, de manière à ne pas causer "
            "de préjudice à autrui."
        )
        assert classifier.classify(
            "get_ccq_articles", text, ok=True
        ) == SearchResultStatus.usable

    def test_stale_abrogated(self, classifier):
        assert classifier.classify(
            "get_ccq_articles", "(Abrogé)", ok=True
        ) == SearchResultStatus.stale

    def test_stale_omitted(self, classifier):
        assert classifier.classify(
            "get_ccq_articles", "(Omis)", ok=True
        ) == SearchResultStatus.stale


# ── Federal document classification ─────────────────────────────────────


class TestFederalDoc:
    def test_usable_doc(self, classifier):
        text = (
            "Section 12. Every person has the right to life, liberty "
            "and security of the person and the right not to be "
            "deprived thereof."
        )
        assert classifier.classify(
            "fetch_document", text, ok=True
        ) == SearchResultStatus.usable

    def test_fully_repealed(self, classifier):
        text = "[Repealed, 2020, c. 1, s. 45]"
        assert classifier.classify(
            "fetch_document", text, ok=True
        ) == SearchResultStatus.stale


# ── Observation convenience ──────────────────────────────────────────────


class TestClassifyObservation:
    def test_observation_ok(self, classifier):
        obs = ToolObservation(
            tool_name="semantic_search_ccq",
            ok=True,
            normalized_response="article 1457 du Code civil du Québec" * 2,
        )
        assert classifier.classify_observation(obs) == SearchResultStatus.usable

    def test_observation_error(self, classifier):
        obs = ToolObservation(
            tool_name="semantic_search_ccq",
            ok=False,
            error="connection refused",
            normalized_response="",
        )
        assert classifier.classify_observation(obs) == SearchResultStatus.tool_error


# ── Case relevance classification ────────────────────────────────────────


class TestClassifyCase:
    def test_usable_case(self, classifier):
        text = (
            "Dupont c. Martin, 2020 QCCS 1234. "
            "Application de l'article 1457 du Code civil. "
            "Vice caché dans un immeuble résidentiel."
        )
        result = classifier.classify_case(
            text, ["1457"], "vice caché immeuble")
        assert result.usable
        assert result.correct_jurisdiction
        assert result.mentions_target_provision
        assert result.relevance_score > 0.5

    def test_wrong_jurisdiction(self, classifier):
        text = "Smith v. Jones, a dispute about contract breach."
        result = classifier.classify_case(
            text, ["1457"], "vice caché")
        assert not result.usable
        assert not result.correct_jurisdiction


# ── Gate batch classification ────────────────────────────────────────────


class TestGateSearchResults:
    def test_empty_text(self, classifier):
        results, status = classifier.gate_search_results("", ["1457"], "vice")
        assert status == CaseLawSearchStatus.empty
        assert results == []

    def test_mixed_results(self, classifier):
        text = (
            "Dupont c. Martin, 2020 QCCS 1234. "
            "Article 1457 appliqué.\n"
            "---\n"
            "Document sans pertinence juridique aucune."
        )
        results, status = classifier.gate_search_results(
            text, ["1457"], "vice caché")
        assert status == CaseLawSearchStatus.usable
        usable = classifier.filter_usable(results)
        assert len(usable) >= 1
