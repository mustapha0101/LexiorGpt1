#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prépare le matériel des 40 questions écrites par une personne réelle.

Toutes nos questions d'évaluation ont été écrites par un modèle, y compris
le test à l'aveugle. Ce script ne produit donc AUCUNE question : il tire des
articles au hasard, décrit en une ligne la situation de la vie courante que
chacun gouverne, et laisse le champ « Question » vide.

Deux graines, toutes deux notées :

* ``GRAINE_TIRAGE`` — 37 articles pris au hasard dans les 4 278 du corpus.
  Aucun filtre : ni sur la longueur, ni sur le livre, ni sur l'intérêt du
  sujet. C'est précisément la sélection des sujets « intéressants » qui a
  rendu le jeu annoté optimiste (hit@3 0,500 contre 0,275 sur un tirage
  au hasard).
* ``GRAINE_AFFICHAGE`` — mélange l'ordre du fichier de travail, pour que
  la position d'une ligne ne trahisse pas son groupe. Sans ce mélange, les
  trois situations hors du Code civil se liraient en fin de fichier et la
  personne saurait, en écrivant, qu'on n'attend aucune réponse.

Les trois groupes ne peuvent pas tous sortir du tirage : une situation hors
du Code civil n'a par définition aucun article dans le corpus. Le tirage
porte donc sur 37 articles, affectés aux groupes PAR ORDRE DE TIRAGE — sans
regarder leur contenu, pour que l'affectation reste aléatoire — et les trois
dernières situations sont écrites sans article attendu.

    python scripts/preparer_situations.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

PHASE1 = Path(__file__).resolve().parents[1]
INDEX = PHASE1 / "data" / "agentic" / "rag_index" / "documents.jsonl"
# Hors de data/, qui est ignoré par git : le fichier de travail sera rempli à
# la main et ces 40 questions ne doivent pas pouvoir se perdre.
SORTIE = PHASE1 / "questions_reelles"

GRAINE_TIRAGE = 20260801
GRAINE_AFFICHAGE = 20260802
NOMBRE_TIRE = 37

# Bornes des groupes dans l'ordre du tirage. L'affectation ne regarde pas le
# contenu de l'article : le 16e tiré est « raconté » parce qu'il est le 16e.
GROUPES = (("simple", 1, 15), ("racontée", 16, 30), ("multiple", 31, 37))

# Compagnons des sept situations à plusieurs articles. Ce sont les seuls
# articles du fichier qui relèvent de mon jugement et non du tirage : la
# situation est écrite à partir de l'article tiré, les compagnons sont ceux
# qu'il faut lire en plus pour y répondre. À contester à la relecture.
COMPAGNONS = {
    31: [("CCQ", "2645"), ("CCQ", "2646"), ("CCQ", "2648")],
    32: [("CCQ", "226"), ("CCQ", "231"), ("CPC", "404")],
    33: [("CCQ", "2841"), ("CCQ", "2860")],
    34: [("CPC", "76"), ("CPC", "77")],
    35: [("CCQ", "2298"), ("CCQ", "2301"), ("CCQ", "2303")],
    36: [("CCQ", "585"), ("CCQ", "587"), ("CCQ", "590"), ("CCQ", "596")],
    37: [("CCQ", "2840"), ("CCQ", "2860")],
}

# Une ligne par article tiré : la situation de la vie courante qu'il gouverne.
# Aucun numéro d'article, aucun terme du Code, aucune formulation reprise du
# texte de loi. Ce sont des situations, pas des questions.
SITUATIONS = {
    1: "Une entreprise a loué un navire pour transporter sa marchandise et le "
       "chargement au port a pris cinq jours de plus que ce qui était entendu ; "
       "le propriétaire du navire réclame de l'argent pour ces journées "
       "d'attente.",
    2: "Une femme loue un logement dans un immeuble à condos et on lui reproche "
       "de ne pas suivre des règles internes dont personne ne lui a jamais remis "
       "le texte.",
    3: "Un père suit depuis quatre ans une entente fixée par un juge sur le "
       "temps passé avec ses enfants, mais il change d'emploi et ses horaires ne "
       "concordent plus du tout.",
    4: "Un homme réclamait de l'argent à une entreprise pour le tort qu'elle lui "
       "avait fait, mais il meurt avant la fin des démarches et ses enfants "
       "veulent continuer à sa place.",
    5: "Une mère seule fait une demande pour un logement à loyer modique et "
       "l'organisme refuse d'inscrire son nom sur la liste d'attente.",
    6: "Deux associés ont un accord écrit et l'un d'eux respecte la lettre de "
       "l'entente tout en manœuvrant pour nuire à l'autre.",
    7: "Un homme a cessé de rembourser le prêt qui a servi à acheter sa maison, "
       "la banque veut reprendre l'immeuble et il refuse de quitter les lieux.",
    8: "Des parents ont choisi qui s'occupera de leur enfant à leur place et la "
       "grand-mère n'est pas d'accord avec cette personne.",
    9: "Un couple marié se sépare et celle qui n'a aucun revenu n'arrive ni à "
       "payer son épicerie ni à payer les frais des démarches.",
    10: "Les gens qui ont acheté dans un immeuble à condos neuf viennent de "
        "prendre la relève du constructeur et celui-ci ne leur a toujours pas "
        "remis les documents et les plans de l'immeuble.",
    11: "Deux entreprises sont en conflit devant la cour mais elles ont commencé "
        "à négocier et ne veulent pas dépenser pour préparer le procès pendant "
        "ce temps.",
    12: "Une femme veut mettre de l'argent de côté pour son fils handicapé, "
        "confié à quelqu'un d'autre qui s'en occupera, de façon que cet argent "
        "ne lui appartienne plus.",
    13: "Un homme travaille au même endroit depuis douze ans et son employeur "
        "lui annonce un vendredi que c'est terminé, avec deux semaines de "
        "préavis.",
    14: "Deux voisins ont réglé leur chicane par une entente écrite signée des "
        "deux et l'un d'eux ne respecte plus ce qui était convenu.",
    15: "Un homme a remporté une maison mise aux enchères et le vendeur tarde à "
        "se présenter pour signer les papiers.",
    16: "Un couple a acheté une maison l'été dernier ; six mois plus tard, la "
        "municipalité leur écrit que l'agrandissement construit à l'arrière "
        "n'était pas permis à cet endroit, et les anciens propriétaires n'en "
        "avaient jamais parlé.",
    17: "Un homme s'est branché en douce sur le compteur électrique de "
        "l'immeuble voisin et se sert du courant depuis des mois ; le voisin "
        "s'en aperçoit en comparant ses factures.",
    18: "Une femme mariée ne supporte plus de vivre avec son mari mais, pour des "
        "raisons de croyance, elle ne veut pas mettre fin au mariage ; ils "
        "dorment dans des chambres séparées depuis deux ans.",
    19: "Trois amis ont signé ensemble le prêt d'un quatrième pour qu'il "
        "l'obtienne ; quand celui-ci a cessé de payer, la banque a tout réclamé "
        "à un seul d'entre eux, qui a vidé ses économies.",
    20: "Un homme voulait rembourser ce qu'il devait mais l'autre personne "
        "refusait d'accepter l'argent ; il a fini par déposer la somme au palais "
        "de justice et il reçoit encore des relevés qui ajoutent des intérêts.",
    21: "Une femme attend depuis des mois que l'entrepreneur finisse sa salle de "
        "bain ; excédée, elle lui envoie un texto lui donnant jusqu'au lendemain "
        "pour terminer.",
    22: "Un garçon de neuf ans a reçu 120 000 $ après la mort de son père et "
        "c'est son oncle qui gère cet argent jusqu'à ses dix-huit ans.",
    23: "Une femme de Trois-Rivières a acheté un appareil sur un site étranger ; "
        "quand elle réclame un remboursement, l'entreprise lui répond que les "
        "petits caractères obligent à s'adresser à une cour du Delaware.",
    24: "Une restauratrice loue un local commercial pour dix ans et veut être "
        "certaine de ne pas être mise dehors si l'immeuble change de mains.",
    25: "Après l'accident vasculaire de leur mère, un des fils a été autorisé à "
        "s'occuper temporairement de la vente de sa voiture et de ses comptes, "
        "et sa sœur veut savoir ce qu'il fait de l'argent.",
    26: "Lors de la réunion annuelle des propriétaires d'un immeuble à condos, "
        "un vote serré passe de justesse ; deux personnes qui possèdent ensemble "
        "le même appartement n'étaient pas là toutes les deux et on conteste le "
        "décompte.",
    27: "Deux femmes se sont unies officiellement devant un célébrant il y a six "
        "ans, sans que ce soit un mariage ; l'une est partie vivre ailleurs et "
        "ne participe plus du tout aux dépenses de la maison.",
    28: "Deux frères se disputent un immeuble laissé par leur mère ; en "
        "attendant la fin de la chicane, un juge a confié la gestion de "
        "l'immeuble à une tierce personne, qui envoie maintenant sa facture.",
    29: "Un homme habitait depuis huit ans avec sa mère dans un logement à loyer "
        "modique ; elle est décédée au printemps et l'organisme lui écrit qu'il "
        "devra partir.",
    30: "Un couple a rempli les papiers à l'hôpital après la naissance de leur "
        "fille, en anglais, et ils attendent encore le document officiel qui "
        "prouve sa naissance.",
    31: "Un homme doit de l'argent à quatre personnes différentes et un huissier "
        "se présente chez lui ; il ne sait pas ce qu'on peut lui prendre ni "
        "comment le peu qu'il possède sera partagé.",
    32: "Une dame de quatre-vingt-quatre ans ne peut plus s'occuper de ses "
        "affaires et il faut réunir sa famille et ses proches pour décider qui "
        "va veiller sur elle.",
    33: "Une entreprise a numérisé tous ses dossiers puis jeté le papier ; "
        "aujourd'hui elle doit prouver le contenu d'une entente signée il y a "
        "six ans et n'a plus que le fichier.",
    34: "Un groupe de citoyens conteste devant la cour une règle qui touche "
        "toute la population et le juge estime que le gouvernement devrait être "
        "présent au débat.",
    35: "Un voyageur ne peut pas régler la note à la fin de son séjour ; l'hôtel "
        "garde sa valise, et lui affirme qu'un objet a disparu de sa chambre "
        "pendant la semaine.",
    36: "Une femme verse chaque mois une somme à son ex-mari depuis le jugement ; "
        "elle vient de perdre son emploi et ne sait ni comment le montant avait "
        "été fixé, ni s'il peut bouger.",
    37: "Deux personnes se disputent au sujet d'une promesse faite par courriel ; "
        "l'une dépose les échanges imprimés et l'autre soutient qu'ils ont pu "
        "être modifiés.",
    # Hors du Code civil et du Code de procédure civile : rien à trouver dans
    # le corpus. Ces trois-là vérifient que le système dit « je n'ai rien »
    # au lieu d'inventer un article plausible.
    38: "Un homme travaille cinquante-cinq heures par semaine et son employeur "
        "ne lui paie aucune majoration pour les heures faites en plus.",
    39: "Une femme reçoit une contravention pour avoir utilisé son téléphone au "
        "volant et veut la contester.",
    40: "Un couple a déposé une demande pour faire venir un parent de l'étranger "
        "et n'a aucune nouvelle depuis quatorze mois.",
}

# Termes du Code qui ne doivent jamais apparaître dans une situation : ils
# offriraient au canal lexical une correspondance gratuite et rendraient la
# mesure aussi optimiste que le jeu annoté.
INTERDITS = (
    "affréteur", "aliments", "authentique", "bail", "bonne foi", "caution",
    "cessible", "chose jugée", "consignation", "copropriété", "curateur",
    "créancier", "débiteur", "délaissement", "dommages-intérêts", "domicile",
    "fiducie", "fret", "hypothèque", "locateur", "mise en demeure",
    "opposable", "patrimoine", "prescription", "préjudice", "publicité "
    "foncière", "résiliation", "solidaire", "séquestre", "surestaries",
    "tribunal", "tuteur", "tutelle", "union civile",
)


def verifier(situation: str, attendus: list[tuple[str, str]]) -> list[str]:
    """Retourne les fuites détectées dans une situation."""
    fautes = []
    minuscule = situation.lower()
    for terme in INTERDITS:
        if terme in minuscule:
            fautes.append(f"terme du Code : « {terme} »")
    chiffres = set(re.findall(r"\d+(?:\.\d+)?", situation))
    for _, numero in attendus:
        if numero in chiffres:
            fautes.append(f"numéro d'article : {numero}")
    if situation.rstrip().endswith("?"):
        fautes.append("c'est une question, pas une situation")
    return fautes


def _contient_des_questions(fichier: Path) -> bool:
    """Vrai si au moins une question a été écrite sous une situation.

    Relancer le script ne doit jamais effacer un travail fait à la main.
    """
    lignes = fichier.read_text(encoding="utf-8").splitlines()
    for position, ligne in enumerate(lignes):
        if ligne.strip() != "**Question :**":
            continue
        for suivante in lignes[position + 1:]:
            texte = suivante.strip()
            if not texte:
                continue
            return not texte.startswith("##")
    return False


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    documents = [json.loads(ligne) for ligne
                 in INDEX.read_text(encoding="utf-8").splitlines()
                 if ligne.strip()]
    tirage = random.Random(GRAINE_TIRAGE).sample(documents, NOMBRE_TIRE)

    entrees = {}
    for interne, situation in SITUATIONS.items():
        if interne <= NOMBRE_TIRE:
            document = tirage[interne - 1]
            attendus = [(document["code"], document["article_number"])]
            attendus += COMPAGNONS.get(interne, [])
            groupe = next(nom for nom, debut, fin in GROUPES
                          if debut <= interne <= fin)
        else:
            attendus = []
            groupe = "hors corpus"
        entrees[interne] = {"groupe": groupe, "situation": situation,
                            "attendus": attendus}

    fautes_totales = 0
    for interne, entree in entrees.items():
        for faute in verifier(entree["situation"], entree["attendus"]):
            print(f"[fuite] situation interne {interne} — {faute}")
            fautes_totales += 1
    if fautes_totales:
        print(f"\n{fautes_totales} fuite(s) : le fichier n'est pas écrit.")
        return 1

    ordre = list(entrees)
    random.Random(GRAINE_AFFICHAGE).shuffle(ordre)

    SORTIE.mkdir(parents=True, exist_ok=True)
    lignes = [
        "# 40 situations — à vous d'écrire les questions",
        "",
        f"Tirage : {NOMBRE_TIRE} articles pris au hasard parmi les "
        f"{len(documents)} du corpus (graine {GRAINE_TIRAGE}), sans aucun "
        "filtre — ni sur le sujet, ni sur la longueur, ni sur le livre. Trois "
        "situations supplémentaires sortent du corpus.",
        "",
        f"L'ordre d'affichage est mélangé (graine {GRAINE_AFFICHAGE}) : la "
        "position d'une ligne ne dit rien de ce qu'on attend d'elle.",
        "",
        "Écrivez votre question sous chaque situation, avec vos mots, comme "
        "vous la poseriez si le problème était le vôtre. Les articles attendus "
        "sont dans un fichier à part, à n'ouvrir qu'une fois les 40 écrites.",
        "",
        "---",
        "",
    ]
    for affichage, interne in enumerate(ordre, 1):
        lignes += [f"## {affichage}",
                   "",
                   entrees[interne]["situation"],
                   "",
                   "**Question :**",
                   "",
                   ""]

    fichier_travail = SORTIE / "situations_a_completer.md"
    if fichier_travail.exists() and _contient_des_questions(fichier_travail):
        print(f"[situations] {fichier_travail} contient déjà des questions "
              "écrites à la main : rien n'est réécrit.")
        return 1
    fichier_travail.write_text("\n".join(lignes), encoding="utf-8")

    cle = {
        "_avertissement": "NE PAS OUVRIR avant d'avoir écrit les 40 questions.",
        "graine_tirage": GRAINE_TIRAGE,
        "graine_affichage": GRAINE_AFFICHAGE,
        "corpus": len(documents),
        "tires_au_hasard": NOMBRE_TIRE,
        "reponses": [
            {"numero": affichage,
             "groupe": entrees[interne]["groupe"],
             "articles_attendus": [f"{code}:{numero}" for code, numero
                                   in entrees[interne]["attendus"]]}
            for affichage, interne in enumerate(ordre, 1)
        ],
    }
    fichier_cle = SORTIE / "NE_PAS_OUVRIR_articles_attendus.json"
    fichier_cle.write_text(json.dumps(cle, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    compte = {}
    for entree in entrees.values():
        compte[entree["groupe"]] = compte.get(entree["groupe"], 0) + 1
    print(f"[situations] {fichier_travail}")
    print(f"[situations] {fichier_cle}")
    print(f"[situations] aucune fuite | répartition {compte}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
