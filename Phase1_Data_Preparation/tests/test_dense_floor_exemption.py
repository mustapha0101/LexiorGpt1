# -*- coding: utf-8 -*-

"""Le plancher absolu ne doit pas écarter ce que le canal du sens a bien classé.

``min_dense_score = 0,40`` a été calibré sur des requêtes en style juridique,
qui scorent 0,58 à 0,70. Il rejetait donc le langage naturel par
construction : « mon fils a cassé la vitrine du dépanneur » score 0,305, alors
que l'article 1459 y est au rang 8 du canal dense.

Le plancher filtre sur le SCORE dense, pas sur l'ORIGINE du candidat. Son
travail utile — écarter ce qui entre par BM25 seul — est préservé en exemptant
les N premiers du canal dense.

La largeur compte. Mesuré sur les 40 questions répondables du jeu annoté :

    exemption   hit@3   hit@10    MRR    candidats retenus
    aucune      0,500   0,600    0,405        2254
    top-10      0,500   0,625    0,408        2297
    top-40      0,475   0,600    0,389        2640

top-40 dégrade : exempter largement laisse entrer des candidats médiocres qui
faussent la normalisation min-max.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG  # noqa: E402


def _rag(largeur: int) -> LegalRAG:
    """Une instance nue : ``_exemptes_du_plancher`` ne lit que ``cfg``."""
    instance = object.__new__(LegalRAG)
    instance.cfg = RAGConfig(dense_floor_exempt_top_k=largeur)
    return instance


# ── Le réglage ───────────────────────────────────────────────────────────


def test_le_defaut_est_dix():
    assert RAGConfig().dense_floor_exempt_top_k == 10


def test_le_yaml_porte_le_reglage():
    cfg = load_config()
    assert cfg.rag.dense_floor_exempt_top_k == 10


def test_le_reglage_est_dans_la_representation_publique():
    assert "dense_floor_exempt_top_k" in RAGConfig().redacted()


# ── L'exemption ──────────────────────────────────────────────────────────


def test_zero_desactive_lexemption():
    """0 rend le comportement identique à l'ancien plancher."""
    candidats = np.arange(20)
    positions = np.arange(20)
    masque = _rag(0)._exemptes_du_plancher(candidats, positions)
    assert not masque.any()


@pytest.mark.parametrize("largeur", [1, 3, 10, 40])
def test_seuls_les_n_premiers_du_canal_dense_sont_exemptes(largeur):
    positions_dense = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                60, 61, 62, 63, 64])
    candidats = np.concatenate([positions_dense, np.array([900, 901])])
    masque = _rag(largeur)._exemptes_du_plancher(candidats, positions_dense)
    attendu = min(largeur, len(positions_dense))
    assert int(masque.sum()) == attendu
    # Les candidats venus des mots seuls ne sont jamais exemptés.
    assert not masque[-1] and not masque[-2]


def test_un_candidat_hors_top_dense_nest_jamais_exempte():
    """C'est le bruit lexical : il doit rester soumis au plancher."""
    positions_dense = np.arange(10)
    candidats = np.array([7, 999])          # 999 n'entre que par BM25
    masque = _rag(10)._exemptes_du_plancher(candidats, positions_dense)
    assert masque[0] and not masque[1]


def test_lexemption_sajoute_au_plancher_elle_ne_le_remplace_pas():
    """Un candidat sous le plancher MAIS bien classé en dense est gardé;
    un candidat sous le plancher et mal classé est écarté."""
    rag = _rag(10)
    rag.cfg.min_dense_score = 0.40
    rag.cfg.min_hybrid_score = 0.0
    candidats = np.array([3, 900])
    positions_dense = np.arange(10)
    dense = np.array([0.305, 0.305])        # les deux sous le plancher
    absolute = np.array([0.30, 0.30])
    garde = (rag._above_floor(dense, absolute)
             | rag._exemptes_du_plancher(candidats, positions_dense))
    assert garde[0], "bien classé en dense : gardé malgré le plancher"
    assert not garde[1], "entré par les mots seuls : écarté"


def test_un_bon_score_reste_garde_sans_exemption():
    rag = _rag(0)
    rag.cfg.min_dense_score = 0.40
    rag.cfg.min_hybrid_score = 0.0
    dense = np.array([0.70])
    absolute = np.array([0.65])
    assert rag._above_floor(dense, absolute)[0]


# ── Ce que l'exemption COÛTE ─────────────────────────────────────────────


def test_lexemption_retire_la_garantie_de_liste_vide(tmp_path):
    """À dire franchement : avec l'exemption, un corpus hors sujet répond.

    Le plancher garantissait qu'un corpus sans réponse produise une liste
    vide. L'exemption la retire, puisque le top-N dense existe toujours.

    Mesuré sur l'index réel (4 278 articles), cette garantie ne s'exerçait
    déjà pas : score dense MAXIMAL par question,

        12 questions sans réponse   0,407 – 0,670   12/12 au-dessus de 0,40
        12 questions répondables    0,429 – 0,605   12/12 au-dessus de 0,40
        2 en langage naturel        0,330 – 0,346    0/2 au-dessus

    Le plancher ne distingue donc pas « sans réponse » de « avec réponse » —
    il ne distingue que « rédigé en juridique » de « rédigé en langage
    courant ». Ce qui rejette réellement les questions sans réponse est le
    reranker LLM (faux positifs 1,000 -> 0,333, lot 1).
    """
    import numpy as np
    from lexior.agentic.legal_rag import LegalRAG as _RAG
    from tests.test_legal_rag import _document, FixedQueryEmbedder  # type: ignore

    documents = [
        _document("CCQ", 1863, "Lorsque le locateur refuse, le tribunal peut trancher.", "louage"),
        _document("CCQ", 1726, "Le vendeur garantit l'acheteur contre les vices.", "vente"),
    ]
    embeddings = np.asarray([[0.9755, 0.22, 0.0], [0.9950, 0.10, 0.0]],
                            dtype=np.float32)
    base = dict(index_dir=str(tmp_path), top_k=5, candidate_k=5,
                min_dense_score=0.30, min_hybrid_score=0.30)
    meta = {"embedding_model": FixedQueryEmbedder.model, "corpus_hash": "test"}
    question = "Mon locateur refuse que j'apporte mon chat."

    sans = _RAG(RAGConfig(**base, dense_floor_exempt_top_k=0),
                FixedQueryEmbedder(), documents, embeddings, meta)
    avec = _RAG(RAGConfig(**base, dense_floor_exempt_top_k=10),
                FixedQueryEmbedder(), documents, embeddings, meta)

    assert sans.search(question, "CCQ") == [], "sans exemption : liste vide"
    assert avec.search(question, "CCQ") != [], "avec exemption : le top dense sort"
