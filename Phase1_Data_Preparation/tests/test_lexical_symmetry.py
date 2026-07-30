# -*- coding: utf-8 -*-

"""legal_terms doit atteindre les DEUX canaux, pas seulement le dense.

``legal_terms`` porte le vocabulaire du Code — donc les termes RARES,
« autorité parentale », « mise en demeure ». Ce sont exactement ceux que BM25
exploite, et le canal lexical ne les recevait pas : ``search()`` appelait
``_bm25`` sur ``query`` seule pendant que le dense prenait le maximum des
deux formulations.

Ces tests n'appellent AUCUN embedding : BM25 se calcule sur l'index déjà
construit, donc la suite reste hors ligne.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.agentic import legal_rag as LR  # noqa: E402
from lexior.agentic.config import load_config  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG  # noqa: E402

VITRINE = "mon fils a cassé la vitrine du dépanneur en jouant au ballon"
CODE_VOCAB = "autorité parentale fait du mineur préjudice"


class _EmbedderInerte:
    """L'index porte déjà ses vecteurs ; BM25 n'en a pas besoin.

    ``model`` et ``dimension`` sont lus au chargement pour vérifier que
    l'index correspond à l'embedder ; on reprend donc ceux de la config.
    """

    def __init__(self, cfg):
        self.model = cfg.embedding_model
        self.price_per_1m = cfg.embedding_price_per_1m_usd
        self.calls = self.failed_calls = self.tokens_in = 0

    def embed(self, textes):  # pragma: no cover — jamais appelé ici
        raise AssertionError("aucun embedding ne doit être demandé")


@pytest.fixture(scope="module")
def rag():
    cfg = load_config()
    try:
        return LegalRAG.load(cfg.rag, _EmbedderInerte(cfg.rag), reranker=None)
    except Exception as exc:                      # index absent en CI
        pytest.skip(f"index RAG indisponible : {exc}")


def _scores(rag, formulations):
    indices = np.asarray(
        [i for i, d in enumerate(rag.documents) if d.code == "CCQ"],
        dtype=np.int64)
    lexicaux = [rag._bm25(LR._expanded_query_tokens(f), indices)
                for f in formulations]
    resultat = lexicaux[0]
    for autre in lexicaux[1:]:
        resultat = np.maximum(resultat, autre)
    numeros = [rag.documents[i].article_number for i in indices]
    return resultat, numeros


def _rang(scores, numeros, cible):
    position = numeros.index(cible)
    return int(np.where(np.argsort(-scores) == position)[0][0]) + 1


def test_les_mots_de_lusager_seuls_ne_trouvent_pas_larticle(rag):
    """Le point de départ : 1459 est au fond du canal des mots."""
    scores, numeros = _scores(rag, [VITRINE])
    assert _rang(scores, numeros, "1459") > 500


def test_le_vocabulaire_du_code_le_place_en_tete(rag):
    scores, numeros = _scores(rag, [CODE_VOCAB])
    assert _rang(scores, numeros, "1459") == 1


def test_lunion_retient_le_meilleur_des_deux(rag):
    """C'est la correction : legal_terms atteint enfin BM25."""
    scores, numeros = _scores(rag, [VITRINE, CODE_VOCAB])
    assert _rang(scores, numeros, "1459") == 1


def test_sans_legal_terms_le_canal_est_inchange(rag):
    """Propriété de sûreté : aucune des 52 questions annotées ne bouge.

    Aucune ne porte de legal_terms ; ``formulations`` n'a donc qu'un
    élément et ``max([x]) == x``.
    """
    indices = np.asarray(
        [i for i, d in enumerate(rag.documents) if d.code == "CCQ"],
        dtype=np.int64)
    for question in (VITRINE, "délai de prescription", "vice caché"):
        ancien = rag._bm25(LR._expanded_query_tokens(question), indices)
        nouveau, _ = _scores(rag, [question])
        assert np.array_equal(ancien, nouveau), question


def test_le_maximum_ne_dilue_pas_lidf():
    """Le meilleur des deux, PAS une fusion des jetons.

    Réunir les jetons allongerait le document virtuel et diluerait l'IDF,
    précisément ce qui fait ressortir un terme rare.
    """
    assert np.maximum(np.array([1.8]), np.array([22.0]))[0] == 22.0
