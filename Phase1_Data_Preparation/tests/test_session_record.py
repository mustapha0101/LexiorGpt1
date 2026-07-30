# -*- coding: utf-8 -*-

"""Enregistrement des sessions de chat : désactivé par défaut, complet sinon.

Trois exigences vérifiées ici :
  * rien ne s'écrit tant que ``chat_sessions_dir`` est vide ;
  * l'écriture porte tout ce qu'on veut relire — question, outils avec leurs
    arguments et leur classification, preuves, réponse INTÉGRALE, verdict ;
  * **les rejets sont enregistrés aussi.** ``route_after_acceptance`` envoie
    tout rejet vers le nœud ``reject``, y compris en live : un enregistrement
    branché sur le seul chemin accepté perdrait les échecs.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.services.session_record import (  # noqa: E402
    construire_entree, enregistrer_tour,
)


@dataclass
class _Config:
    chat_sessions_dir: str = ""


class _Obs:
    def __init__(self, tool_name, arguments, reponse, ok=True):
        self.tool_name = tool_name
        self.server = "lexior-legisquebec-mcp"
        self.arguments = arguments
        self.ok = ok
        self.error = None
        self.normalized_response = reponse
        self.content_hash = "sha256:abc"
        self.source_urls = []
        self.citations = []
        self.latency_ms = 120


class _Evaluation:
    def __init__(self, index, statut, motif=""):
        self.tool_call_index = index
        self.tool_name = "get_ccq_articles"
        self.result_status = statut
        self.result_reason = motif


class _Acceptance:
    accepted = False
    reasons = ["[ungrounded_url] URL absente des réponses d'outils"]
    blocking_errors = ["[ungrounded_url] URL absente des réponses d'outils"]
    warnings = ["[query_improvable] séquence: jurisprudence citée sans article"]
    failed_checks = ["grounding_urls"]


REPONSE_LONGUE = "Selon l'article 1726 du Code civil du Québec, " + "x" * 3000


def _etat(**kw):
    base = {
        "thread_id": "session-test",
        "latest_user_message": "Mon voisin passe sur mon terrain, que faire?",
        "request_type": "case_analysis",
        "resolved_jurisdiction": "Québec",
        "jurisdiction_status": "supported_quebec",
        "jurisdiction_basis": "explicit_user_statement",
        "tool_history": [
            _Obs("semantic_search_ccq", {"query": "passage terrain voisin"},
                 "Article 1177 — score 0.71"),
            _Obs("get_ccq_articles", {"start_article": 1177,
                                      "end_article": 1180},
                 "Article 1177\nLa servitude est une charge imposée…"),
        ],
        "search_evaluations": [
            _Evaluation(0, "usable"),
            _Evaluation(1, "irrelevant", "retrieval-only: candidate evidence"),
        ],
        "usable_evidence_entries": [
            {"tool_name": "get_ccq_articles", "tool_history_index": 1,
             "detailed_status": "official_and_relevant",
             "articles": ["1177"], "citations": [], "source_urls": [],
             "content_hash": "sha256:abc"},
        ],
        "final_answer": REPONSE_LONGUE,
        "latest_decision": {"thinking_text": "La question porte sur une "
                                             "servitude de passage."},
        "acceptance_result": _Acceptance(),
        "stop_reason": "validation_failed",
        "validation_issues": ["[tool_call_with_prose] …"],
    }
    base.update(kw)
    return base


# ── Désactivé par défaut ─────────────────────────────────────────────────


def test_rien_ne_secrit_sans_reglage(tmp_path):
    assert enregistrer_tour(_etat(), _Config(chat_sessions_dir=""),
                            "accepted") is None
    assert not list(tmp_path.iterdir())


# ── Contenu ──────────────────────────────────────────────────────────────


def test_lentree_porte_tout_ce_quon_veut_relire():
    e = construire_entree(_etat(), "accepted")
    assert e["question"] == "Mon voisin passe sur mon terrain, que faire?"
    assert e["request_type"] == "case_analysis"
    assert e["juridiction"]["retenue"] == "Québec"
    assert [o["nom"] for o in e["outils"]] == ["semantic_search_ccq",
                                               "get_ccq_articles"]
    assert e["outils"][1]["arguments"] == {"start_article": 1177,
                                           "end_article": 1180}
    assert e["outils"][0]["classification"] == "usable"
    assert e["outils"][1]["classification"] == "irrelevant"
    assert e["outils"][1]["motif_classification"].startswith("retrieval-only")
    assert e["preuves"][0]["articles"] == ["1177"]


def test_la_reponse_nest_jamais_tronquee():
    """Découvrir une réponse coupée au moment de l'analyser serait pire
    que le poids du fichier."""
    e = construire_entree(_etat(), "accepted")
    assert e["reponse_finale"] == REPONSE_LONGUE
    assert len(e["reponse_finale"]) > 3000
    assert e["outils"][1]["reponse"].startswith("Article 1177")


# ── Les rejets, surtout ──────────────────────────────────────────────────


def test_un_rejet_est_enregistre_avec_ses_erreurs(tmp_path):
    fichier = enregistrer_tour(
        _etat(), _Config(chat_sessions_dir=str(tmp_path)), "rejected")
    assert fichier is not None
    entree = json.loads(Path(fichier).read_text(encoding="utf-8").strip())
    assert entree["statut"] == "rejected"
    assert entree["acceptation"]["accepte"] is False
    assert any("ungrounded_url" in m
               for m in entree["acceptation"]["erreurs"])
    assert any("query_improvable" in m
               for m in entree["acceptation"]["avertissements"])
    assert entree["acceptation"]["controles_echoues"] == ["grounding_urls"]
    # Un échec doit rester aussi lisible qu'une réussite.
    assert len(entree["outils"]) == 2
    assert entree["reponse_finale"] == REPONSE_LONGUE


def test_plusieurs_tours_sajoutent_au_meme_fichier(tmp_path):
    cfg = _Config(chat_sessions_dir=str(tmp_path))
    enregistrer_tour(_etat(), cfg, "accepted")
    enregistrer_tour(_etat(), cfg, "rejected")
    fichiers = list(tmp_path.glob("*.jsonl"))
    assert len(fichiers) == 1
    lignes = fichiers[0].read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(x)["statut"] for x in lignes] == ["accepted",
                                                         "rejected"]


def test_le_fichier_est_hors_du_corpus(tmp_path):
    """Ces sessions sont un jeu de test, pas des données d'entraînement."""
    fichier = enregistrer_tour(
        _etat(), _Config(chat_sessions_dir=str(tmp_path)), "accepted")
    chemin = Path(fichier).resolve()
    assert "accepted.jsonl" not in chemin.name
    assert "agentic" not in chemin.parent.name
