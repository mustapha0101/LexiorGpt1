# -*- coding: utf-8 -*-

"""Ce que le modèle LIT doit expliquer ce qu'il VOIT.

Deux défauts trouvés sur une trace Qwen :

* les articles étaient affichés dans l'ordre du classement FUSIONNÉ mais
  étiquetés avec le score ABSOLU, qui ne l'explique pas. Sur une question de
  dommage causé par un mineur, le bon article sortait premier à 0,546 pendant
  que des articles hors sujet affichaient 0,61-0,62. Qwen a suivi les
  nombres et récupéré les mauvais articles ; gpt-4o-mini a suivi l'ordre et
  réussi. Les deux lectures étaient défendables, une seule marchait ;

* ``get_ccq_articles`` n'avait aucune borne de plage : {1318 -> 1621} a été
  accepté — 304 articles — puis {1604 -> 273}, où le début est après la fin.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG  # noqa: E402
from lexior.agentic.tool_catalog import (  # noqa: E402
    MAX_ARTICLES_PAR_APPEL, load_catalog,
)
from tests.test_legal_rag import FakeEmbedder, _document  # noqa: E402


# ── La sortie de recherche ───────────────────────────────────────────────


class _RerankerInverseur:
    """Renvoie l'ordre inverse : le rang affiché doit suivre, pas résister."""

    def __init__(self, numeros):
        self.numeros = list(numeros)

    def complete_json(self, role, messages, **kw):
        return {"ranking": list(reversed(self.numeros)), "rejected": []}


def _rag(tmp_path, reranker=None, **overrides):
    documents = [
        _document("CCQ", 1459, "Le titulaire de l'autorité parentale répond du mineur.", "obligations"),
        _document("CCQ", 1460, "La personne à qui la garde du mineur est confiée.", "obligations"),
        _document("CCQ", 1708, "La vente est un contrat translatif de propriété.", "vente"),
    ]
    embeddings = np.asarray([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
                            dtype=np.float32)
    settings = dict(index_dir=str(tmp_path), top_k=3, candidate_k=3,
                    min_dense_score=0.0, min_hybrid_score=0.0,
                    llm_rerank_enabled=bool(reranker))
    settings.update(overrides)
    return LegalRAG(RAGConfig(**settings), FakeEmbedder(), documents,
                    embeddings,
                    {"embedding_model": FakeEmbedder.model,
                     "corpus_hash": "test"}, reranker=reranker)


def test_chaque_article_porte_un_rang_numerote(tmp_path):
    payload = _rag(tmp_path).call(
        "semantic_search_ccq", {"query": "autorité parentale mineur"})
    lignes = [l for l in payload["text"].splitlines() if l.strip()
              and l[0].isdigit()]
    assert lignes, payload["text"]
    for position, ligne in enumerate(lignes, start=1):
        assert ligne.startswith(f"{position}. "), ligne


def test_le_score_affiche_est_nomme_pour_ce_quil_est(tmp_path):
    """« confiance », pas « pertinence » : ce n'est pas le critère de tri."""
    payload = _rag(tmp_path).call(
        "semantic_search_ccq", {"query": "autorité parentale mineur"})
    assert "confiance" in payload["text"]
    assert "PAS le critère de classement" in payload["text"]


def test_lentete_previent_quun_mieux_classe_peut_scorer_plus_bas(tmp_path):
    payload = _rag(tmp_path).call(
        "semantic_search_ccq", {"query": "autorité parentale mineur"})
    entete = payload["text"].split("\n\n")[0]
    assert "rang fait foi" in entete


def test_le_rang_du_json_suit_lordre_reellement_renvoye(tmp_path):
    """``rank`` était figé AVANT le reranker, qui peut réordonner."""
    sans = _rag(tmp_path).call(
        "semantic_search_ccq", {"query": "autorité parentale mineur"})
    numeros = [r["article_number"] for r in sans["results"]]

    avec = _rag(tmp_path, reranker=_RerankerInverseur(numeros)).call(
        "semantic_search_ccq", {"query": "autorité parentale mineur"})
    rangs = [r["rank"] for r in avec["results"]]
    assert rangs == list(range(1, len(rangs) + 1)), (
        "le champ rank doit décrire la liste renvoyée, pas l'ordre d'avant "
        "le reranker")
    reranked = [r["article_number"] for r in avec["results"]]
    assert reranked[0] == numeros[0], "le noyau de rappel reste protégé"
    assert reranked[1:] == list(reversed(numeros))[0:-1], (
        "le reranker réordonne les candidats hors noyau")


def test_aucun_resultat_reste_lisible_comme_vide(tmp_path):
    rag = _rag(tmp_path, min_dense_score=0.99, min_hybrid_score=0.99,
               dense_floor_exempt_top_k=0)
    payload = rag.call("semantic_search_ccq", {"query": "sujet absent"})
    assert payload["results"] == []
    assert payload["text"] == "Aucun article CCQ trouvé."


# ── Les bornes de plage ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def catalogue():
    return load_catalog(load_config().catalog_path)


@pytest.mark.parametrize("outil", ["get_ccq_articles", "get_cpc_articles"])
def test_une_plage_inversee_est_refusee(catalogue, outil):
    """Le cas exact de la trace : {1604 -> 273}."""
    erreurs = catalogue.validate_call(
        outil, {"start_article": 1604, "end_article": 273})
    assert erreurs and "inversée" in erreurs[0]


@pytest.mark.parametrize("outil", ["get_ccq_articles", "get_cpc_articles"])
def test_une_plage_demesuree_est_refusee(catalogue, outil):
    """Le cas exact de la trace : {1318 -> 1621}, soit 304 articles."""
    erreurs = catalogue.validate_call(
        outil, {"start_article": 1318, "end_article": 1621})
    assert erreurs and "304" in erreurs[0]


def test_la_borne_est_inclusive(catalogue):
    limite = MAX_ARTICLES_PAR_APPEL
    assert not catalogue.validate_call(
        "get_ccq_articles", {"start_article": 1, "end_article": limite})
    assert catalogue.validate_call(
        "get_ccq_articles", {"start_article": 1, "end_article": limite + 1})


@pytest.mark.parametrize("arguments", [
    {"start_article": 1459},
    {"start_article": 1459, "end_article": 1459},
    {"start_article": 1459, "end_article": 1460},
    {"start_article": 596.1, "end_article": 596.1},
])
def test_les_demandes_sensees_passent(catalogue, arguments):
    assert not catalogue.validate_call("get_ccq_articles", arguments)


def test_les_autres_outils_ne_sont_pas_concernes(catalogue):
    assert not catalogue.validate_call(
        "search_ccq_keywords", {"keyword": "responsabilité"})
