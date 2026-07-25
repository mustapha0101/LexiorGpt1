# -*- coding: utf-8 -*-
"""Le texte rédigé par un modèle n'est pas une preuve.

Certains serveurs MCP renvoient une synthèse générée au lieu du texte
source. Sans détection, le contrôle de grounding valide les citations de
la réponse finale contre… du texte généré.
"""

from __future__ import annotations

from agentic_generation.schemas import (
    Message, Role, ToolObservation, TrainingTrajectory,
)
from agentic_generation.validators import validate_trajectory
from lexior.agentic.mcp_executor import normalize_mcp_response
from lexior.agentic.response_verifier import (
    contains_generated_summary,
    strip_reader_directed,
)
from lexior.services import ResultVerificationService
from lexior.services.evidence import EvidenceLevel

# Forme réellement observée le 2026-07-24 en interrogeant le serveur.
GENERATED_RESPONSE = (
    "### [Loi sur la protection du consommateur, RLRQ c P-40.1]"
    "(https://www.canlii.org/fr/qc/legis/lois/rlrq-c-p-40.1/derniere/"
    "rlrq-c-p-40.1.html)\n"
    "**Résumé :** Summary:\n"
    "- Le cadre protège les consommateurs contre certaines clauses.\n"
    "- L'article 1437 encadre les clauses abusives.\n"
    "Si vous souhaitez, je peux extraire les passages exacts."
)

SOURCE_RESPONSE = (
    "Article 1457\n"
    "Toute personne a le devoir de respecter les règles de conduite qui, "
    "suivant les circonstances, les usages ou la loi, s'imposent à elle."
)


# ── Détection ────────────────────────────────────────────────────────────


def test_a_generated_summary_is_detected():
    assert contains_generated_summary(GENERATED_RESPONSE)


def test_official_source_text_is_not_flagged():
    assert not contains_generated_summary(SOURCE_RESPONSE)


def test_empty_response_is_not_flagged():
    assert not contains_generated_summary("")


# ── Nettoyage ────────────────────────────────────────────────────────────


def test_reader_directed_sentences_are_removed():
    cleaned = strip_reader_directed(GENERATED_RESPONSE)

    assert "Si vous souhaitez" not in cleaned
    assert "je peux extraire" not in cleaned
    assert "protection du consommateur" in cleaned, "le contenu reste"


def test_normalization_removes_them_end_to_end():
    text, _urls, _citations, _truncated = normalize_mcp_response(
        {"text": GENERATED_RESPONSE}, 10_000)

    assert "Si vous souhaitez" not in text
    assert "RLRQ c P-40.1" in text


def test_cleaning_does_not_make_the_content_citable():
    """Retirer les offres ne transforme pas une synthèse en source."""
    cleaned = strip_reader_directed(GENERATED_RESPONSE)

    assert contains_generated_summary(cleaned)


# ── Verdict ──────────────────────────────────────────────────────────────


def _observation(response: str, tool_name: str = "search_quebec_regulations"):
    return ToolObservation(
        tool_name=tool_name, normalized_response=response, ok=True)


def test_generated_content_is_not_citable_even_from_an_official_tool():
    assessment = ResultVerificationService().assess(
        _observation(GENERATED_RESPONSE))

    assert assessment.official, "le serveur reste une source officielle"
    assert not assessment.citable
    assert not assessment.usable_as_evidence
    assert assessment.evidence_level == EvidenceLevel.candidate.value
    assert assessment.reason


def test_source_text_from_the_same_family_stays_citable():
    assessment = ResultVerificationService().assess(
        _observation(SOURCE_RESPONSE, tool_name="get_ccq_articles"))

    assert assessment.citable and assessment.usable_as_evidence


# ── Grounding ────────────────────────────────────────────────────────────


def _trajectory(tool_name: str, response: str, final: str):
    observation = ToolObservation(
        tool_name=tool_name, arguments={"query": "clause abusive"},
        raw_response=response, normalized_response=response,
        mock=True).finalize_hash()
    return TrainingTrajectory(
        scenario_id="s", scenario_family_id="f",
        request_type="exact_text_retrieval",
        expected_jurisdiction="Québec", resolved_jurisdiction="Québec",
        messages=[
            Message(role=Role.user, content="Une clause abusive est-elle valide?"),
            Message(role=Role.assistant, content=(
                f'<tool_call>\n{{"name":"{tool_name}",'
                f'"arguments":{{"query":"clause abusive"}}}}\n</tool_call>')),
            Message(role=Role.tool, name=tool_name, content=response),
            Message(role=Role.assistant, content=final),
        ], tool_trace=[observation])


def test_an_article_seen_only_in_generated_text_is_not_grounded(catalog):
    row = _trajectory("search_quebec_regulations", GENERATED_RESPONSE,
                      "L'article 1437 rend la clause abusive nulle.")

    result = validate_trajectory(row, catalog, allow_mock=True)

    assert any("article 1437 absent" in error for error in result.errors), (
        f"le grounding a accepté du texte généré : {result.errors}")


def test_the_same_article_seen_in_source_text_is_grounded(catalog):
    row = _trajectory("get_ccq_articles",
                      "Article 1437\nLa clause abusive d'un contrat "
                      "d'adhésion est nulle.",
                      "L'article 1437 rend la clause abusive nulle.")

    result = validate_trajectory(row, catalog, allow_mock=True)

    assert not any("article 1437 absent" in error for error in result.errors), (
        result.errors)


# ── Citation vs prose : deux niveaux distincts ───────────────────────────

DECISION_RESPONSE = (
    "### [D É C I S I O N](https://www.canlii.org/fr/qc/qctal/doc/2023/"
    "2023qctal37140/2023qctal37140.html)\n"
    "**Résumé :** Tribunal administratif du logement — 2023 QCTAL 37140. "
    "Le tribunal résilie le bail pour non-paiement et ordonne l'expulsion. "
    "L'article 1883 du Code civil est appliqué.\n"
    "Si vous souhaitez, je peux détailler le calcul de la dette."
)


def _decision_trajectory(final: str):
    tool = "search_quebec_jurisprudence"
    observation = ToolObservation(
        tool_name=tool, arguments={"query": "TAL décision résiliation bail"},
        raw_response=DECISION_RESPONSE,
        normalized_response=DECISION_RESPONSE,
        source_urls=["https://www.canlii.org/fr/qc/qctal/doc/2023/"
                     "2023qctal37140/2023qctal37140.html"],
        citations=["2023 QCTAL 37140"],
        mock=True).finalize_hash()
    return TrainingTrajectory(
        scenario_id="s", scenario_family_id="f",
        request_type="case_law_research",
        expected_jurisdiction="Québec", resolved_jurisdiction="Québec",
        messages=[
            Message(role=Role.user,
                    content="Y a-t-il des décisions sur la résiliation pour "
                            "non-paiement?"),
            Message(role=Role.assistant, content=(
                f'<tool_call>\n{{"name":"{tool}","arguments":'
                f'{{"query":"TAL décision résiliation bail"}}}}\n</tool_call>')),
            Message(role=Role.tool, name=tool, content=DECISION_RESPONSE),
            Message(role=Role.assistant, content=final),
        ], tool_trace=[observation])


def test_a_decision_citation_stays_usable_despite_the_generated_summary(
        catalog):
    """Sinon l'outil devient inutilisable pour ce à quoi il sert."""
    row = _decision_trajectory(
        "Voir 2023 QCTAL 37140 : https://www.canlii.org/fr/qc/qctal/doc/2023/"
        "2023qctal37140/2023qctal37140.html")

    result = validate_trajectory(row, catalog, allow_mock=True)

    assert not any("citation absente" in error for error in result.errors), (
        result.errors)
    assert not any("URL absente" in error for error in result.errors), (
        result.errors)


def test_an_article_asserted_only_by_the_summary_is_still_refused(catalog):
    """La prose générée ne peut pas ancrer un article, même à côté d'une
    vraie décision."""
    row = _decision_trajectory(
        "L'article 1883 du Code civil règle la question.")

    result = validate_trajectory(row, catalog, allow_mock=True)

    assert any("article 1883 absent" in error for error in result.errors), (
        result.errors)


# ── URLs SOQUIJ tronquées ────────────────────────────────────────────────


def test_a_truncated_soquij_url_is_repaired():
    from lexior.agentic.mcp_executor import normalize_soquij_urls

    repaired = normalize_soquij_urls(
        "Voir https://citoyens.soquij.qc.ca/ID=7E787E320A66D6492CD2EA3108BAEED7")

    assert ("https://citoyens.soquij.qc.ca/php/decision.php"
            "?ID=7E787E320A66D6492CD2EA3108BAEED7") in repaired


def test_a_well_formed_soquij_url_is_left_alone():
    from lexior.agentic.mcp_executor import normalize_soquij_urls

    url = ("https://citoyens.soquij.qc.ca/php/decision.php"
           "?ID=5B90AAD41A54AEB822A1ADE2CFAF44AD")

    assert normalize_soquij_urls(url) == url


def test_normalization_repairs_the_url_it_extracts():
    text, urls, _citations, _truncated = normalize_mcp_response(
        {"text": "Décision : https://citoyens.soquij.qc.ca/"
                 "ID=DD05DD72C30B29FEFF734F7BFF713F5F"}, 10_000)

    assert all("/php/decision.php?ID=" in url for url in urls), urls
    assert "/ID=" not in text.replace("?ID=", "")
