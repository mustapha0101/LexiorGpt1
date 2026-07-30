# -*- coding: utf-8 -*-

"""Une clarification ne doit pas faire oublier la question posée.

``handle_clarification`` écrase ``latest_user_message`` par la réponse de
l'usager — « au Québec », « préciser quoi? ». ``classify_tool_result``
comparait l'article récupéré à cette réponse : sur une question de morsure de
chien, l'article 1466 — qui traite précisément du préjudice causé par un
animal — ressortait ``irrelevant``, motif « résultat sans rapport thématique
avec la question posée », et la trajectoire finissait sans réponse.

``update_active_task`` conserve la question d'origine dans ``active_issue``.
C'est elle que le contrôle de pertinence doit lire.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.agentic.schemas import ToolObservation  # noqa: E402
from lexior.agentic.tool_catalog import load_catalog  # noqa: E402
from lexior.agentic.config import load_config  # noqa: E402
from lexior.agent_graph.context import GraphContext  # noqa: E402
from lexior.agent_graph.nodes import classify_tool_result  # noqa: E402
from lexior.services.result_verification import (  # noqa: E402
    ResultVerificationService,
)

QUESTION = ("Mon chien a mordu un passant dans la rue. Est-ce que je suis "
            "responsable des blessures?")

ARTICLE_1466 = (
    "Article 1466\n"
    "Le propriétaire d’un animal est tenu de réparer le préjudice que "
    "l’animal a causé, soit qu’il fût sous sa garde ou sous celle d’un "
    "tiers, soit qu’il fût égaré ou échappé.\n"
    "La personne qui se sert de l’animal en est aussi, pendant ce temps, "
    "responsable avec le propriétaire.")


class _Services:
    verification = ResultVerificationService()


@pytest.fixture(scope="module")
def ctx():
    config = load_config()
    return GraphContext(config=config,
                        catalog=load_catalog(config.catalog_path),
                        services=_Services())


def _etat(latest_user_message: str, active_issue: str):
    return {
        "tool_history": [ToolObservation(
            tool_name="get_ccq_articles", ok=True,
            arguments={"start_article": 1466},
            normalized_response=ARTICLE_1466)],
        "latest_user_message": latest_user_message,
        "active_issue": active_issue,
        "resolved_jurisdiction": "Québec",
        "requested_court_scope": "",
        "search_evaluations": [],
        "usable_evidence": [],
        "candidate_sources": [], "citable_sources": [],
        "usable_evidence_entries": [], "alternative_sources": [],
        "invalidated_sources": [],
    }


def test_apres_clarification_le_bon_article_reste_usable(ctx):
    """Le cas exact du premier test réel : clarification puis article 1466."""
    updates = classify_tool_result.run(
        _etat(latest_user_message="preciser quoi?", active_issue=QUESTION),
        ctx)
    assert updates["last_tool_result_status"] == "usable", (
        "l'article 1466 traite du préjudice causé par un animal : comparé à "
        "la question d'origine il est pertinent. S'il ressort irrelevant, "
        "c'est que la réponse de clarification a de nouveau écrasé la "
        "question.")


def test_sans_clarification_le_comportement_ne_change_pas(ctx):
    """active_issue vaut la question : rien ne bouge pour les cas simples."""
    updates = classify_tool_result.run(
        _etat(latest_user_message=QUESTION, active_issue=QUESTION), ctx)
    assert updates["last_tool_result_status"] == "usable"


def test_active_issue_absent_on_retombe_sur_le_dernier_message(ctx):
    """Repli : une trajectoire sans active_issue doit rester classable."""
    etat = _etat(latest_user_message=QUESTION, active_issue="")
    updates = classify_tool_result.run(etat, ctx)
    assert updates["last_tool_result_status"] == "usable"


def test_un_article_hors_sujet_reste_ecarte(ctx):
    """La correction ne doit pas rendre le contrôle complaisant."""
    etat = _etat(latest_user_message="preciser quoi?", active_issue=QUESTION)
    etat["tool_history"] = [ToolObservation(
        tool_name="get_ccq_articles", ok=True,
        arguments={"start_article": 2098},
        normalized_response=(
            "Article 2098\nLe contrat d’entreprise ou de service est celui "
            "par lequel une personne s’engage envers une autre à réaliser "
            "un ouvrage matériel ou intellectuel."))]
    updates = classify_tool_result.run(etat, ctx)
    assert updates["last_tool_result_status"] != "usable"
