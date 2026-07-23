#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Couche de nettoyage du corpus de droit québécois
(intelliwork/canadian-quebec-law-corpus).

Distincte de a2aj_cleaner.py : les deux corpus n'ont ni la même granularité
ni la même syntaxe de souche.

    a2aj/canadian-laws        -> une ligne = une LOI fédérale entière,
                                 souches entre crochets : "[Abrogé, 2017, ch. 33]"
    canadian-quebec-law-corpus-> une ligne = un ARTICLE,
                                 souches entre parenthèses : "(Abrogé)."

Le corpus québécois étant déjà propre et découpé par article, il n'y a ni
troncature ni assemblage à faire : le nettoyage se limite à écarter les
articles sans contenu normatif.

Utilisation depuis generate_ccq_data.py :

    from ccq_cleaner import load_quebec_articles
    articles = load_quebec_articles()
    for item in articles:
        item["article"], item["texte"], item["domaine"]
"""

import os
import re

# --- Juridiction retenue ---------------------------------------------------
QUEBEC_JURISDICTIONS = {"Québec (Provincial)"}

# --- Souches québécoises ---------------------------------------------------
# Le texte de l'article EST la souche : "(Abrogé).", "(Omis).",
# "(Modification intégrée au c. B-1, a. 125)."
#
# Aucune borne de longueur : certaines souches du CPC sont longues (jusqu'à
# 210 caractères, cf. art. 780). Ce qui délimite une souche n'est pas sa
# taille mais sa STRUCTURE — une parenthèse unique couvrant tout l'article.
# "[^)]" interdit toute parenthèse imbriquée et l'ancre "$" impose que rien ne
# suive : un article réel du type "(1) Le tribunal peut ... (2) ..." ne peut
# donc pas correspondre. Contrôle sur le corpus : 87 articles correspondent,
# et les 87 commencent par Abrogé / Omis / Modification — aucun faux positif.
_STUB_RE = re.compile(r"^\s*\(\s*(?P<body>[^)]*?)\s*\)\s*\.?\s*$", re.IGNORECASE)

MIN_ARTICLE_CHARS = 30

UNUSABLE_CLASSES = {"EMPTY", "REPEALED", "OMITTED", "SPENT", "STUB_OTHER", "TOO_SHORT"}
USABLE_CLASSES = {"LIVE"}

# --- Domaine juridique, dérivé de chemin_taxonomy --------------------------
# Le corpus n'a pas de champ "domaine" ; l'ancien generate_ccq_data.py en
# attendait un. On le reconstitue à partir des livres du CCQ et des titres du
# CPC, en gardant une formulation qui se lit dans la question de repli
# « ... dans une situation de {domaine.lower()} ? ».
TAXONOMY_TO_DOMAIN = {
    "ccq/livre1_personnes": "Personnes et droits de la personnalité",
    "ccq/livre2_famille": "Droit de la famille",
    "ccq/livre3_successions": "Successions",
    "ccq/livre4_biens": "Biens et propriété",
    "ccq/livre5_obligations": "Obligations et contrats",
    "ccq/livre6_priorites_suretes": "Priorités et hypothèques",
    "ccq/livre7_preuve": "Preuve",
    "ccq/livre8_prescription": "Prescription",
    "ccq/livre9_publicite_droits": "Publicité des droits",
    "ccq/livre10_droit_international_prive": "Droit international privé",
    "cpc/titre1_dispositions_communes": "Procédure civile (dispositions communes)",
    "cpc/titre2_modes_prives_reglement": "Modes privés de prévention et de règlement des différends",
    "cpc/titre3_procedure_contentieuse": "Procédure contentieuse",
    "cpc/titre4_jugement_recours": "Jugement et voies de recours",
    "cpc/titre5_execution_forcee": "Exécution forcée",
    "cpc/titre6_procedures_non_contentieuses_et_autres": "Procédures non contentieuses",
}


# --- URL officielle, construite et non inventée -----------------------------
# Le corpus n'a pas de colonne source_url. Sans URL fournie, le Teacher recopiait
# mot pour mot le gabarit du prompt système — les 58 citations d'un échantillon
# de 60 valaient toutes « .../rlrq-c-ccq-1991/... », points de suspension compris.
# Les URL de LegisQuébec sont déterministes : on les construit depuis l'id, ce
# qui donne au générateur québécois l'équivalent de la réécriture d'URL du
# générateur fédéral.
LEGISQUEBEC_BASE = "https://www.legisquebec.gouv.qc.ca/fr/document/lc"
CODE_SLUG = {
    "Code civil du Québec": "ccq-1991",
    "Code de procédure civile du Québec": "cpc",
}


def article_source_url(code, article):
    """« Code civil du Québec », « Article 1457 » -> URL LegisQuébec de l'article.

    Retourne une chaîne vide si le code est inconnu : mieux vaut aucune URL
    qu'une URL fausse.
    """
    slug = CODE_SLUG.get(code)
    if not slug:
        return ""
    m = re.search(r"([\d.]+)", article or "")
    if not m:
        return f"{LEGISQUEBEC_BASE}/{slug}"
    return f"{LEGISQUEBEC_BASE}/{slug}#se:{m.group(1)}"


def taxonomy_to_domain(chemin):
    """« provincial_quebec/ccq/livre5_obligations » -> « Obligations et contrats »."""
    key = (chemin or "").replace("provincial_quebec/", "").strip("/")
    if key in TAXONOMY_TO_DOMAIN:
        return TAXONOMY_TO_DOMAIN[key]
    # Repli lisible pour toute rubrique non répertoriée.
    tail = key.split("/")[-1] if key else ""
    tail = re.sub(r"^(livre|titre)\d+_", "", tail).replace("_", " ").strip()
    return tail.capitalize() if tail else "Droit québécois"


def classify_article(texte):
    """Classe un article du corpus québécois.

    LIVE       : droit en vigueur.                            -> utilisable
    REPEALED   : « (Abrogé). » — l'article est abrogé.        -> rejet
    OMITTED    : « (Omis). » — article non reproduit.         -> rejet
    SPENT      : disposition épuisée : « (Modification
                 intégrée au c. B-1, a. 125). ». L'article
                 modifiait une autre loi ; la modification est
                 intégrée, il ne reste rien à interpréter.
                 Équivalent québécois du « [Modifications] »
                 fédéral (cf. a2aj_cleaner.classify_section).  -> rejet
    STUB_OTHER : autre souche entre parenthèses.              -> rejet
    EMPTY      : vide.                                        -> rejet
    TOO_SHORT  : moins de MIN_ARTICLE_CHARS caractères.       -> rejet
    """
    t = (texte or "").strip()
    if not t:
        return "EMPTY"

    m = _STUB_RE.match(t)
    if m:
        body = m.group("body").strip().lower()
        if body.startswith("abrog"):
            return "REPEALED"
        if body.startswith("omis"):
            return "OMITTED"
        if body.startswith("modification"):
            return "SPENT"
        return "STUB_OTHER"

    if len(t) < MIN_ARTICLE_CHARS:
        return "TOO_SHORT"

    return "LIVE"


def clean_article(row):
    """Filtre une ligne du corpus québécois.

    Retourne None si l'article est écarté, sinon un dict de la même forme que
    l'ancienne constante CCQ_ARTICLES_SAMPLES, afin que la boucle de
    generate_ccq_data.py reste inchangée :

        {"article": "Article 1457", "texte": "...", "domaine": "...",
         "code": "...", "id": "ccq_1457"}
    """
    if row.get("jurisdiction") not in QUEBEC_JURISDICTIONS:
        return None

    if classify_article(row.get("texte")) not in USABLE_CLASSES:
        return None

    code = row.get("code") or ""
    article = row.get("article") or ""
    return {
        "article": article,
        "texte": (row.get("texte") or "").strip(),
        "domaine": taxonomy_to_domain(row.get("chemin_taxonomy")),
        "code": code,
        "id": row.get("id") or "",
        # URL construite depuis l'identifiant, jamais laissée au Teacher.
        "source_url": article_source_url(code, article),
    }


def clean_articles(rows, codes=None, limit=-1):
    """Applique clean_article à un itérable de lignes.

    codes : liste de codes à conserver (p. ex. ["Code civil du Québec"]).
            None = tous.
    limit : nombre maximal d'articles retenus ; -1 = tous.

    Retourne (articles, stats).
    """
    from collections import Counter

    out = []
    stats = Counter()
    for row in rows:
        if codes and row.get("code") not in codes:
            stats["hors_code"] += 1
            continue
        if row.get("jurisdiction") not in QUEBEC_JURISDICTIONS:
            stats["hors_juridiction"] += 1
            continue
        cls = classify_article(row.get("texte"))
        stats[cls] += 1
        if cls not in USABLE_CLASSES:
            continue
        out.append(clean_article(row))
        if limit is not None and limit >= 0 and len(out) >= limit:
            break
    return out, dict(stats)


def load_quebec_articles(dataset_name="intelliwork/canadian-quebec-law-corpus",
                         split="train", codes=None, limit=-1, token=None):
    """Charge le corpus depuis le Hub, le nettoie, et renvoie (articles, stats).

    Le dépôt est privé : un jeton est requis (argument token ou variable
    d'environnement HF_TOKEN).
    """
    from datasets import load_dataset

    token = token or os.environ.get("HF_TOKEN") or None
    ds = load_dataset(dataset_name, split=split, token=token)
    return clean_articles(ds, codes=codes, limit=limit)
