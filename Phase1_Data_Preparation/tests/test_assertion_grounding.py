# -*- coding: utf-8 -*-

"""L'affirmation doit être soutenue par le TEXTE, pas seulement par le numéro.

``validate_final`` vérifiait que le NUMÉRO d'article cité figure dans les
preuves — jamais que ce qui est affirmé à son sujet corresponde à ce que
l'article dit. Sur une question de branches d'arbre, le modèle a cité
correctement l'article 984 (« les fruits qui tombent d'un arbre appartiennent
au propriétaire de l'arbre ») puis ajouté « ce qui souligne la responsabilité
du propriétaire en cas de dommages ». Le lien n'existe pas, le numéro était
dans les preuves, la trajectoire a été acceptée.

Trois exigences, vérifiées ici :
  * la question posée au modèle porte sur l'AFFIRMATION et le TEXTE ;
  * un échec technique REJETTE — ne pas avoir pu vérifier n'est pas avoir
    vérifié ;
  * le verdict ressort dans ``problemes_validation``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.services.assertion_grounding import (  # noqa: E402
    AssertionGroundingService, articles_incompatibles_deterministes,
    affirmation_autour, textes_recuperes,
)
from lexior.agentic.trajectory_agent import _articles_retenus  # noqa: E402

ART_984 = ("Les fruits qui tombent d’un arbre sur un fonds voisin "
           "appartiennent au propriétaire de l’arbre.")
ART_985 = ("Si des branches ou racines venant du fonds voisin s’avancent sur "
           "son fonds et nuisent sérieusement à son usage, le propriétaire "
           "peut demander à son voisin de les couper.")

INVENTION = ("Selon l'article 984 du Code civil du Québec, les fruits qui "
             "tombent d'un arbre appartiennent au propriétaire de l'arbre, ce "
             "qui souligne la responsabilité du propriétaire en cas de "
             "dommages.")


class _Client:
    """Client scriptable : mémorise ce qu'on lui a réellement demandé."""

    _DEFAUT = object()

    def __init__(self, reponse=_DEFAUT, exception=None):
        self.reponse = ({"soutenue": True}
                        if reponse is _Client._DEFAUT else reponse)
        self.exception = exception
        self.appels: list[dict] = []

    def complete_json(self, role, messages, **kw):
        self.appels.append({"role": role, "messages": messages})
        if self.exception:
            raise self.exception
        return self.reponse


class _Obs:
    def __init__(self, tool_name, reponse, ok=True):
        self.tool_name = tool_name
        self.ok = ok
        self.normalized_response = reponse


# ── La question posée porte sur l'affirmation ET le texte ────────────────


def test_la_question_contient_le_texte_les_faits_et_la_reponse():
    client = _Client({"non_soutenues": []})
    AssertionGroundingService(client=client).verifier(
        INVENTION, {"984": ART_984}, faits="les fruits tombent chez moi")
    assert len(client.appels) == 1, "UN appel sur la réponse entière"
    envoye = client.appels[0]["messages"][-1]["content"]
    assert ART_984 in envoye, "le texte réel doit être soumis au juge"
    assert "responsabilité du propriétaire en cas de dommages" in envoye
    assert "les fruits tombent chez moi" in envoye, (
        "les faits sont nécessaires : un article visant un arbre qui MENACE "
        "de tomber ne s'applique pas à un arbre déjà tombé")


def test_la_consigne_exclut_le_numero_et_la_justesse_generale():
    client = _Client({"non_soutenues": []})
    AssertionGroundingService(client=client).verifier(
        INVENTION, {"984": ART_984})
    systeme = client.appels[0]["messages"][0]["content"]
    assert "NI si le numéro d'article est le bon" in systeme
    assert "NI si l'affirmation est juridiquement correcte en général" in systeme


def test_une_affirmation_non_soutenue_est_signalee():
    client = _Client({"non_soutenues": [
        {"affirmation": "ce qui souligne la responsabilité",
         "motif": "ajoute une portée absente"}]})
    verdicts = AssertionGroundingService(client=client).verifier(
        INVENTION, {"984": ART_984})
    assert len(verdicts) == 1
    assert not verdicts[0].soutenue
    assert "n'est pas soutenu par le texte" in verdicts[0].probleme()


def test_une_affirmation_soutenue_ne_produit_rien_de_negatif():
    client = _Client({"non_soutenues": []})
    verdicts = AssertionGroundingService(client=client).verifier(
        "L'article 985 permet de demander au voisin de couper les branches.",
        {"985": ART_985})
    assert all(v.soutenue for v in verdicts)


# ── L'échec technique REJETTE ────────────────────────────────────────────


def test_une_exception_du_client_ne_laisse_pas_passer():
    client = _Client(exception=RuntimeError("timeout"))
    verdicts = AssertionGroundingService(client=client).verifier(
        INVENTION, {"984": ART_984})
    assert verdicts[0].echec_technique
    assert not verdicts[0].soutenue, (
        "ne pas avoir pu vérifier n'est pas avoir vérifié")
    assert "vérification impossible" in verdicts[0].probleme()


@pytest.mark.parametrize("reponse", [
    {}, {"motif": "sans verdict"}, "pas un objet", None,
    {"non_soutenues": "pas une liste"}])
def test_une_reponse_illisible_est_un_echec_technique(reponse):
    client = _Client(reponse)
    verdicts = AssertionGroundingService(client=client).verifier(
        INVENTION, {"984": ART_984})
    assert verdicts[0].echec_technique and not verdicts[0].soutenue


# ── Les artefacts corrigés ───────────────────────────────────────────────


def test_une_affirmation_multi_articles_recoit_tous_les_textes():
    """Sinon le juge conclut « non soutenue » parce qu'un texte manque.

    C'était un artefact du contrôle, pas une invention du modèle : il
    gonflait la mesure de 36 % à 47 %.
    """
    client = _Client({"non_soutenues": []})
    AssertionGroundingService(client=client).verifier(
        "Selon les articles 984 et 985, le voisin peut couper les branches.",
        {"984": ART_984, "985": ART_985})
    envoye = client.appels[0]["messages"][-1]["content"]
    assert ART_984 in envoye and ART_985 in envoye


def test_une_omission_nest_pas_une_invention():
    """La consigne doit le dire : on ne juge pas la complétude."""
    client = _Client({"non_soutenues": []})
    AssertionGroundingService(client=client).verifier(
        INVENTION, {"984": ART_984})
    systeme = client.appels[0]["messages"][0]["content"]
    assert "une omission" in systeme.lower()
    assert "on ne juge pas si la réponse est complète" in systeme.lower()


def test_un_article_sans_texte_recupere_nest_pas_juge():
    """Le contrôle d'ancrage par numéro couvre déjà ce cas."""
    client = _Client({"non_soutenues": []})
    verdicts = AssertionGroundingService(client=client).verifier(
        "Selon l'article 1457, toute personne a le devoir…", {"984": ART_984})
    assert verdicts == [] and client.appels == []


# ── La fenêtre d'affirmation ─────────────────────────────────────────────


def test_la_fenetre_couvre_la_phrase_et_la_suivante():
    """L'invention observée débordait sur la proposition suivante."""
    texte = ("Selon l'article 984, les fruits appartiennent au propriétaire. "
             "Cela engage donc sa responsabilité en cas de dommages. "
             "Par ailleurs, un délai de prescription s'applique.")
    fenetre = affirmation_autour(texte, texte.index("984"), texte.index("984") + 3)
    assert "responsabilité" in fenetre
    assert "prescription" not in fenetre, "la fenêtre ne doit pas tout avaler"


# ── La récupération des textes ───────────────────────────────────────────


def test_les_textes_viennent_des_reponses_doutils():
    observations = [
        _Obs("get_ccq_articles", f"Article 984\n{ART_984}"),
        _Obs("semantic_search_ccq", "1. Article 984 — confiance 0.7"),
    ]
    textes = textes_recuperes(observations)
    assert set(textes) == {"984"}, "seule la récupération officielle compte"
    assert textes["984"].startswith("Les fruits")


def test_une_reponse_en_echec_ne_fournit_aucun_texte():
    assert textes_recuperes(
        [_Obs("get_ccq_articles", f"Article 984\n{ART_984}", ok=False)]) == {}


def test_plusieurs_articles_dans_une_meme_reponse():
    contenu = f"Article 984\n{ART_984}\n\nArticle 985\n{ART_985}"
    textes = textes_recuperes([_Obs("get_ccq_articles", contenu)])
    assert set(textes) == {"984", "985"}


# ── Service indisponible ─────────────────────────────────────────────────


def test_une_condition_qui_ne_correspond_pas_aux_faits_est_relevee():
    """« menace de tomber » ne couvre pas un arbre déjà tombé."""
    client = _Client({"non_soutenues": [
        {"affirmation": "vous pouvez contraindre votre voisin à abattre",
         "motif": "l'article ne s'applique pas à un arbre déjà tombé"}]})
    verdicts = AssertionGroundingService(client=client).verifier(
        "Conformément à l'article 985, vous pouvez contraindre votre voisin "
        "à abattre l'arbre.", {"985": ART_985},
        faits="un arbre pourri de mon voisin est tombé sur mon garage")
    assert verdicts and not verdicts[0].soutenue
    assert "déjà tombé" in verdicts[0].motif


def test_sans_client_le_controle_ne_bloque_rien():
    assert AssertionGroundingService(client=None).verifier(
        INVENTION, {"984": ART_984}) == []


def test_en_mode_offline_le_controle_est_inerte():
    assert AssertionGroundingService(client=_Client(), offline=True).verifier(
        INVENTION, {"984": ART_984}) == []


def test_selection_pre_redaction_ne_garde_pas_un_texte_incompatible():
    textes = {
        "10": "La mesure ne peut être prise qu'avant la survenance du fait.",
        "20": "Toute personne doit réparer le préjudice causé par sa faute.",
    }
    client = _Client({"articles": [
        {"article": "10", "statut": "incompatible",
         "motif": "le fait est déjà survenu"},
        {"article": "20", "statut": "incertain",
         "motif": "la faute reste à établir"},
    ]})
    selection = AssertionGroundingService(client=client).selectionner_articles(
        textes, faits="le fait s'est déjà produit")
    assert selection["10"].statut == "incompatible"
    assert selection["20"].statut == "incertain"
    systeme = client.appels[0]["messages"][0]["content"]
    assert "sans règle mémorisée" in systeme


def test_selection_partielle_ne_filtre_pas_les_textes_officiels():
    client = _Client({"articles": [
        {"article": "10", "statut": "incompatible", "motif": ""},
    ]})
    selection = AssertionGroundingService(client=client).selectionner_articles(
        {"10": "Texte A", "20": "Texte B"}, faits="faits")
    assert selection == {}


def test_un_lot_incomplet_nefface_pas_les_verdicts_des_lots_precedents():
    """A late failed batch must not re-authorize already excluded sources."""
    class ClientParLot:
        def __init__(self):
            self.appels = 0

        def complete_json(self, *_args, **_kwargs):
            self.appels += 1
            if self.appels == 1:
                return {"articles": [
                    {"article": str(numero),
                     "statut": "incompatible" if numero == 2 else "applicable",
                     "motif": ""}
                    for numero in range(1, 9)
                ]}
            return {"articles": []}  # incomplete second batch

    selection = AssertionGroundingService(client=ClientParLot()).selectionner_articles(
        {str(numero): f"Texte officiel {numero}." for numero in range(1, 10)},
        faits="faits")

    assert set(selection) == {str(numero) for numero in range(1, 9)}
    assert selection["2"].statut == "incompatible"


def test_cause_incompatible_force_lexclusion_du_texte():
    client = _Client({"articles": [
        {"article": "10", "cause_compatible": False,
         "statut": "applicable", "motif": ""},
    ]})
    selection = AssertionGroundingService(client=client).selectionner_articles(
        {"10": "Texte officiel."}, faits="faits")
    assert selection["10"].statut == "incompatible"


@pytest.mark.parametrize("faits", [
    "Le dommage est déjà survenu.",
    "Le dommage est deja survenu.",
    "Le fait est tombe et a causé un dommage.",
])
def test_regle_preventive_ecartee_apres_un_dommage_realise(faits):
    client = _Client({"articles": [
        {"article": "10", "statut": "applicable", "motif": ""},
    ]})
    selection = AssertionGroundingService(client=client).selectionner_articles(
        {"10": "La mesure vise à prévenir un risque avant qu'il survienne."},
        faits=faits)
    assert selection["10"].statut == "incompatible"


def test_incompatibilite_preventive_reste_active_sans_relecteur():
    exclusions = articles_incompatibles_deterministes(
        {"10": "La mesure vise à prévenir un risque avant qu'il survienne."},
        faits="Le dommage est déjà survenu.")

    assert exclusions["10"].statut == "incompatible"


def test_verifier_bloque_une_regle_preventive_apres_le_dommage():
    verdicts = AssertionGroundingService(client=_Client({
        "non_soutenues": []
    })).verifier(
        "Selon l'article 10, vous pouvez imposer une mesure préventive.",
        {"10": "La mesure vise à prévenir un risque avant qu'il survienne."},
        faits="Le dommage est déjà survenu.")

    assert verdicts and not verdicts[0].soutenue
    assert verdicts[0].motif


def test_selection_par_lots_ne_devient_pas_partielle_sur_neuf_textes():
    textes = {str(numero): f"Texte officiel {numero}." for numero in range(1, 10)}
    client = _Client({"articles": [
        {"article": str(numero), "statut": "incertain", "motif": ""}
        for numero in range(1, 10)
    ]})

    selection = AssertionGroundingService(client=client).selectionner_articles(
        textes, faits="faits")

    assert set(selection) == set(textes)
    assert len(client.appels) == 2


def test_redacteur_ne_recoit_que_les_articles_retenus():
    contenu = (
        "Article 10\nTexte incompatible.\n\n"
        "Article 20\nTexte retenu."
    )
    assert _articles_retenus(contenu, {"20"}) == "Article 20\nTexte retenu."


def test_juge_accepte_une_application_conditionnelle_dune_regle_generale():
    client = _Client({"non_soutenues": []})
    AssertionGroundingService(client=client).verifier(
        "Selon l'article 10, une obligation peut exister si ses conditions "
        "sont établies.", {"10": "Toute personne doit respecter son obligation."})
    systeme = client.appels[0]["messages"][0]["content"]
    assert "règle générale peut soutenir une application CONDITIONNELLE" in systeme
