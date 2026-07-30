# -*- coding: utf-8 -*-

"""Un numéro d'article ne doit pas sortir de la mémoire du modèle.

Sur une question de morsure de chien, après deux recherches par mot-clé
vides, le planner a appelé ``get_ccq_articles(1465)`` en écrivant « je sais
que l'article 1465 stipule que le propriétaire d'un animal est responsable ».
C'est faux — 1465 vise les biens, 1466 les animaux — et ça contredit la
première ligne du prompt système.

La même garde sert deux contrôles opposés :
  * le planner REFUSE un numéro sans provenance ;
  * le contrôle de pertinence n'applique PAS son veto lexical à un numéro qui
    en a une, sans quoi « le chien de ma voisine m'a mordue » ne partage
    aucune racine avec « le propriétaire d'un animal est tenu de réparer le
    préjudice » et le BON article ressort ``irrelevant``.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.services.provenance import (  # noqa: E402
    a_une_provenance, numero_demande, numeros_dans, reponses_reussies,
)
from lexior.agent_graph.result_classifier import ResultClassifier  # noqa: E402
from lexior.agent_graph.nodes.validate_final import (  # noqa: E402
    _regle_sans_source,
)

QUESTION = ("le chien de ma voisine m'a mordue pendant que je marchais sur le "
            "trotoir est ce que je peux porter pleinte?")
ART_1466 = ("Article 1466\nLe propriétaire d’un animal est tenu de réparer le "
            "préjudice que l’animal a causé, soit qu’il fût sous sa garde ou "
            "sous celle d’un tiers.")


class _Obs:
    def __init__(self, tool_name, reponse, ok=True):
        self.tool_name = tool_name
        self.ok = ok
        self.normalized_response = reponse


# ── Provenance ───────────────────────────────────────────────────────────


def test_un_numero_invente_na_pas_de_provenance():
    """Le cas exact : deux recherches vides, puis 1465 sorti de nulle part."""
    vides = [_Obs("search_ccq_keywords", "Aucun article trouvé."),
             _Obs("search_ccq_keywords", "Aucun article trouvé.")]
    assert not a_une_provenance(
        "1465", QUESTION, reponses_reussies(vides))


def test_un_numero_issu_dune_recherche_a_une_provenance():
    trouve = [_Obs("semantic_search_ccq",
                   "Article 1466 — score de pertinence 0.709")]
    assert a_une_provenance("1466", QUESTION, reponses_reussies(trouve))


def test_un_numero_ecrit_par_lusager_a_une_provenance():
    assert a_une_provenance(
        "1457", "Peux-tu me donner le texte de l'article 1457?", [])


def test_seules_les_reponses_reussies_comptent():
    echec = [_Obs("search_ccq_keywords", "Article 1465", ok=False)]
    assert not a_une_provenance("1465", QUESTION, reponses_reussies(echec))


def test_les_outils_hors_recuperation_ne_sont_pas_concernes():
    assert numero_demande("semantic_search_ccq", {"query": "1466"}) is None
    assert numero_demande("get_ccq_articles", {"start_article": 1466}) == "1466"
    assert numero_demande("get_ccq_articles", {}) is None


def test_les_articles_a_decimale_sont_reconnus():
    assert "199.5" in numeros_dans("voir l'article 199.5 du Code")


# ── Le veto lexical et l'écart de vocabulaire ────────────────────────────


class TestVetoLexical:
    """Le contrôle de pertinence est un recouvrement de racines.

    « chien / mordue / voisine » ne recoupe pas « animal / préjudice /
    propriétaire ». C'est le même écart de vocabulaire qui a fermé les sept
    pistes d'optimisation de la recherche.
    """

    def test_sans_provenance_le_bon_article_est_ecarte(self):
        assert ResultClassifier().classify(
            "get_ccq_articles", ART_1466, True, user_query=QUESTION,
        ).value == "irrelevant"

    def test_avec_provenance_le_bon_article_passe(self):
        assert ResultClassifier().classify(
            "get_ccq_articles", ART_1466, True, user_query=QUESTION,
            provenance_verifiee=True,
        ).value == "usable"

    def test_le_veto_reste_actif_pour_les_recherches(self):
        """L'exemption ne vaut QUE pour la récupération par numéro."""
        assert ResultClassifier().classify(
            "search_ccq_keywords", ART_1466, True, user_query=QUESTION,
        ).value == "irrelevant"


# ── Une règle énoncée sans source ────────────────────────────────────────


class TestRegleSansSource:
    @pytest.mark.parametrize("reponse", [
        "Selon l'article 1465, le gardien d'un bien est tenu de réparer.",
        "Le gardien est tenu de réparer le préjudice, sauf s'il prouve "
        "n'avoir commis aucune faute.",
        "Le Code civil du Québec stipule que le propriétaire répond du fait "
        "de son animal.",
    ])
    def test_une_regle_enoncee_sans_preuve_est_signalee(self, reponse):
        issues = _regle_sans_source({
            "answer_contract": {"answer_mode": "no_evidence"},
            "final_answer": reponse})
        assert issues and "answer_from_memory" in issues[0]

    def test_une_reponse_qui_sabstient_ne_declenche_rien(self):
        assert not _regle_sans_source({
            "answer_contract": {"answer_mode": "no_evidence"},
            "final_answer": ("Je n'ai pas trouvé de disposition applicable. "
                             "Consultez CanLII ou SOQUIJ.")})

    def test_rien_a_signaler_quand_des_preuves_existent(self):
        assert not _regle_sans_source({
            "answer_contract": {"answer_mode": "grounded"},
            "final_answer": "Selon l'article 1466, vous êtes responsable."})

    def test_la_contradiction_exacte_de_la_trace(self):
        """Les deux affirmations coexistaient dans la même réponse."""
        reponse = ("Il est possible de porter plainte… Cependant, je n'ai pas "
                   "trouvé d'articles spécifiques sur la responsabilité des "
                   "animaux dans le Code civil du Québec. Le gardien est tenu "
                   "de réparer le préjudice, sauf s'il prouve n'avoir commis "
                   "aucune faute.")
        assert _regle_sans_source({
            "answer_contract": {"answer_mode": "no_evidence"},
            "final_answer": reponse})
