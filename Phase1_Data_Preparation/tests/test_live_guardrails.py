# -*- coding: utf-8 -*-
"""Garde-fous du parcours live — aucun appel réseau ni modèle.

Ces tests couvrent les défauts qui avaient atteint la démonstration :
chaque cas correspond à un comportement observé, pas à une hypothèse.
Ils s'exécutent hors ligne et doivent le rester.

    ..\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from lexior.agent_graph.nodes.validate_plan import _distinct_legal_terms
from lexior.agent_graph.state import LexiorState, initial_state
from lexior.agentic.citations import (
    extract_article_citations,
    mentions_article,
)
from lexior.agentic.response_verifier import strip_reader_directed
from lexior.agentic.schemas import ScenarioSpec
from lexior.services.text_folding import fold_text

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "lexior"


# ── Classification hors droit ────────────────────────────────────────────
# « Quelle est la meilleure recette de poutine? » recevait
# « Dans quelle province êtes-vous? » : en live, seul un salut menait à
# non_legal, tout le reste héritant du défaut « case_analysis » du runner.

# ── Repliement typographique ─────────────────────────────────────────────
# Le corpus officiel compte 21 932 apostrophes U+2019 pour UNE ASCII, et
# NFKD ne les replie pas : un extrait recopié avec une apostrophe droite
# était déclaré absent de sa propre source, puis abandonné en silence.

def test_apostrophe_typographique_repliee():
    assert fold_text("L’obligation") == fold_text("L'obligation")


def test_extrait_retrouve_dans_sa_source():
    source = fold_text("Le propriétaire d’un animal est tenu de réparer")
    assert fold_text("proprietaire d'un animal") in source


@pytest.mark.parametrize("brut,attendu", [
    ("1457‑1458", "1457-1458"),
    ("« texte »", '" texte "'),
    ("a b", "a b"),
])
def test_ponctuation_uniformisee(brut, attendu):
    assert fold_text(brut) == attendu


# ── Citations d'articles ─────────────────────────────────────────────────
# Le motif ne reconnaissait que « article N » au singulier : « les articles
# 1457 et 1465 » ne livrait aucun numéro, et 1465 — jamais récupéré —
# partait vers l'utilisateur comme vérifié.

@pytest.mark.parametrize("texte,attendu", [
    ("article 1457", ["1457"]),
    ("art. 1457", ["1457"]),
    ("Les articles 1457 et 1465 rendent une personne responsable",
     ["1457", "1465"]),
    ("les articles 1457, 1465 et 1470", ["1457", "1465", "1470"]),
    ("Selon 1465 C.c.Q., la personne est responsable", ["1465"]),
    ("en vertu de 1457 du Code civil", ["1457"]),
    ("article 1074.1", ["1074.1"]),
    ("aucune citation ici", []),
])
def test_extraction_des_citations(texte, attendu):
    assert extract_article_citations(texte) == attendu


@pytest.mark.parametrize("source", [
    "Article 1466\nLe propriétaire d'un animal...",
    "art. 1466",
    "1466 C.c.Q.",
])
def test_source_reconnue_quelle_que_soit_la_forme(source):
    assert mentions_article(source, "1466")


def test_article_absent_non_reconnu():
    assert not mentions_article("Article 1466\ntexte", "1465")


# ── Préservation de la source ────────────────────────────────────────────
# L'art. 513 C.p.c. disparaissait entièrement : chacune de ses lignes
# portait un marqueur d'adresse au lecteur.

def test_nettoyage_ne_vide_jamais_une_source():
    tout_marque = "vous pouvez demander une chose\nil est recommandé de faire ceci"
    assert strip_reader_directed(tout_marque) == tout_marque


def test_nettoyage_retire_ce_qui_peut_l_etre():
    texte = "Article 1457\nvous pouvez demander"
    assert strip_reader_directed(texte).strip() == "Article 1457"


# ── Seconde formulation de recherche ─────────────────────────────────────
# Recopier la question dans « legal_terms » passe le schéma mais rend
# l'union dégénérée : le jeu de candidats se réduit aux premiers rangs
# denses et les candidats purement lexicaux disparaissent.

def test_legal_terms_recopie_rejete():
    question = "Mon voisin a un chien qui a mordu mon fils"
    assert _distinct_legal_terms(question, question) == ""
    assert _distinct_legal_terms(question + " svp", question) == ""


def test_legal_terms_distinct_conserve():
    question = "Mon voisin a un chien qui a mordu mon fils"
    termes = "responsabilite du proprietaire d'un animal, reparation du prejudice"
    assert _distinct_legal_terms(termes, question) == termes


# ── Canaux d'état ────────────────────────────────────────────────────────
# LangGraph IGNORE en silence toute clé non déclarée : une famille entière
# d'événements était écrite puis jetée.

def _cles_retournees_par_les_noeuds() -> dict[str, set[str]]:
    resultat: dict[str, set[str]] = {}

    def cles(noeud):
        if isinstance(noeud, ast.Dict):
            for k in noeud.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    yield k.value

    fichiers = list((SRC / "agent_graph" / "nodes").glob("*.py"))
    fichiers.append(SRC / "agent_graph" / "graph.py")
    for chemin in fichiers:
        if chemin.name == "__init__.py":
            continue
        arbre = ast.parse(chemin.read_text(encoding="utf-8"))
        for noeud in ast.walk(arbre):
            if isinstance(noeud, ast.Return) and noeud.value is not None:
                for cle in cles(noeud.value):
                    resultat.setdefault(cle, set()).add(chemin.name)
            if isinstance(noeud, ast.Call):
                nom = (getattr(noeud.func, "id", None)
                       or getattr(noeud.func, "attr", None))
                if nom == "Command":
                    for kw in noeud.keywords:
                        if kw.arg == "update":
                            for cle in cles(kw.value):
                                resultat.setdefault(cle, set()).add(chemin.name)
    return resultat


@pytest.mark.parametrize("canal", [
    "claim_events", "claim_ledger_rebuilt", "failure_resolved",
    "answer_repair_started", "answer_repair_succeeded", "answer_repair_failed",
    "safe_fallback_built", "remedy_events", "preflight_grounding_issues",
    "open_grounding_failures_total", "resolved_grounding_failures_total",
    "clarification_unresolved", "node_failed", "error_type",
    "grounding_failures", "case_law_retry_count",
])
def test_canal_de_telemetrie_declare(canal):
    assert canal in LexiorState.__annotations__


def test_initial_state_couvre_tous_les_canaux():
    """Invariant documenté dans state.py : l'état initial est complet."""
    etat = initial_state(
        ScenarioSpec(scenario_id="t", scenario_family_id="t",
                     request_type="case_analysis", language="fr",
                     user_query="test", jurisdiction=""),
        mode="live")
    manquants = sorted(set(LexiorState.__annotations__) - set(etat))
    assert not manquants, f"absents de initial_state : {manquants}"


# ── Budgets ──────────────────────────────────────────────────────────────

def test_budgets_de_reformulation_separes():
    """Un seul compteur pour deux budgets épuisait l'un via l'autre."""
    texte = (SRC / "agent_graph" / "nodes" / "validate_plan.py").read_text(
        encoding="utf-8")
    assert "case_law_retry_count" in texte
    assert '"reformulation_count"] = state.get(' not in texte
