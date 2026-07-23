#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests de la couche de nettoyage du corpus de droit québécois.

    python test_ccq_cleaner.py            # tests unitaires (hors ligne)
    python test_ccq_cleaner.py --live     # + contrôle sur le corpus réel (jeton HF requis)

Aucun appel au modèle Teacher n'est effectué.
"""

import sys

from ccq_cleaner import (classify_article, clean_article, clean_articles,
                         taxonomy_to_domain)

ok = 0
ko = 0


def check(label, got, expected):
    global ok, ko
    if got == expected:
        ok += 1
        print(f"  [OK  ] {label}")
    else:
        ko += 1
        print(f"  [FAIL] {label}\n         attendu={expected!r}  obtenu={got!r}")


# --------------------------------------------------------------------------
print("\n1. classify_article — souches québécoises « (Abrogé). »")

check("article de fond",
      classify_article("Toute personne a le devoir de respecter les règles de conduite qui, "
                       "suivant les circonstances, les usages ou la loi, s'imposent à elle."),
      "LIVE")
check("(Abrogé).", classify_article("(Abrogé)."), "REPEALED")
check("(Abrogé) sans point", classify_article("(Abrogé)"), "REPEALED")
check("(Abrogée).", classify_article("(Abrogée)."), "REPEALED")
check("(abrogé). minuscule", classify_article("(abrogé)."), "REPEALED")
check("(Omis).", classify_article("(Omis)."), "OMITTED")
# Disposition épuisée du CPC (art. 813-832) : équivalent du « [Modifications] »
# fédéral. L'article n'est pas abrogé, mais il ne reste rien à interpréter.
check("(Modification intégrée ...) -> SPENT",
      classify_article("(Modification intégrée au c. B-1, a. 125)."), "SPENT")
check("(Modification intégrée ...) longue -> SPENT",
      classify_article("(Modification intégrée au c. H-4.1, a. 13)."), "SPENT")
# Souche réelle de 210 caractères (CPC art. 780) : ce n'est pas la longueur qui
# définit une souche, mais le fait qu'une parenthèse unique couvre tout l'article.
check("souche très longue (210 car., CPC art. 780) -> SPENT",
      classify_article("(Modification intégrée aux c. C-19, a. 14.1, 468.45.8, 568, 569 et 573.3.4; "
                       "C-27.1, a. 19, 614.8, 938.4, 1082 et 1094; C-37.01, a. 118.2; C-37.02, "
                       "a. 111.2; S-30.01, a. 108.2; T-14, a. 6, V-6.1, a. 204 et 358)."),
      "SPENT")
# Le garde-fou : un article réel à plusieurs alinéas parenthésés n'est PAS une
# souche, quelle que soit sa longueur.
check("article à alinéas « (1) … (2) … » -> LIVE",
      classify_article("(1) Le tribunal peut ordonner la production d'un document. "
                       "(2) Il peut aussi en dispenser une partie."),
      "LIVE")
check("souche inconnue entre parenthèses", classify_article("(Remplacé)."), "STUB_OTHER")
check("vide", classify_article(""), "EMPTY")
check("None", classify_article(None), "EMPTY")
check("trop court", classify_article("Définitions."), "TOO_SHORT")

# Le piège symétrique de celui du corpus fédéral : un article VIVANT qui
# mentionne « abrogé » au fil du texte ne doit pas être écarté.
check("article vivant mentionnant « abrogé » -> LIVE",
      classify_article("Les dispositions de la loi abrogée continuent de s'appliquer aux "
                       "instances introduites avant son remplacement."),
      "LIVE")
# La syntaxe fédérale entre crochets ne doit rien déclencher ici : ce cleaner
# ne traite que le corpus québécois.
check("crochets fédéraux ignorés ici",
      classify_article("[Abrogé, 2017, ch. 33, art. 228]"), "LIVE")

# --------------------------------------------------------------------------
print("\n2. taxonomy_to_domain")

check("livre 5 -> obligations",
      taxonomy_to_domain("provincial_quebec/ccq/livre5_obligations"),
      "Obligations et contrats")
check("livre 1 -> personnes",
      taxonomy_to_domain("provincial_quebec/ccq/livre1_personnes"),
      "Personnes et droits de la personnalité")
check("titre CPC",
      taxonomy_to_domain("provincial_quebec/cpc/titre5_execution_forcee"),
      "Exécution forcée")
check("rubrique inconnue -> repli lisible",
      taxonomy_to_domain("provincial_quebec/ccq/livre11_nouveau_sujet"),
      "Nouveau sujet")
check("vide -> repli", taxonomy_to_domain(""), "Droit québécois")
check("None -> repli", taxonomy_to_domain(None), "Droit québécois")

# --------------------------------------------------------------------------
print("\n3. clean_article — forme attendue par generate_ccq_data.py")


def row(**kw):
    base = {
        "id": "ccq_1457",
        "title": "CCQ Article 1457",
        "article": "Article 1457",
        "code": "Code civil du Québec",
        "jurisdiction": "Québec (Provincial)",
        "texte": "Toute personne a le devoir de respecter les règles de conduite qui, suivant "
                 "les circonstances, les usages ou la loi, s'imposent à elle.",
        "chemin_taxonomy": "provincial_quebec/ccq/livre5_obligations",
    }
    base.update(kw)
    return base


r = clean_article(row())
check("article vivant -> conservé", r is not None, True)
# La boucle de generate_ccq_data.py lit exactement ces trois clés.
check("  clé 'article' (attendue par le générateur)", r["article"], "Article 1457")
check("  clé 'texte'", r["texte"].startswith("Toute personne a le devoir"), True)
check("  clé 'domaine' reconstituée", r["domaine"], "Obligations et contrats")
check("  clé 'code' conservée", r["code"], "Code civil du Québec")
check("  clé 'id' conservée", r["id"], "ccq_1457")

check("(Abrogé). -> écarté", clean_article(row(texte="(Abrogé).")) is None, True)
check("(Omis). -> écarté", clean_article(row(texte="(Omis).")) is None, True)
check("texte vide -> écarté", clean_article(row(texte="")) is None, True)
check("hors juridiction -> écarté",
      clean_article(row(jurisdiction="Ontario (Provincial)")) is None, True)

# --------------------------------------------------------------------------
print("\n4. clean_articles — filtrage par lot")

batch = [
    row(id="ccq_1", article="Article 1", texte="Tout être humain possède la personnalité juridique; "
                                               "il a la pleine jouissance des droits civils."),
    row(id="ccq_106", article="Article 106", texte="(Abrogé)."),
    row(id="ccq_2190", article="Article 2190", texte="(Abrogé)."),
    row(id="cpc_822", article="Article 822", texte="(Omis).",
        code="Code de procédure civile du Québec",
        chemin_taxonomy="provincial_quebec/cpc/titre6_procedures_non_contentieuses_et_autres"),
    row(id="cpc_1", article="Article 1", texte="Pour prévenir un différend ou pour le régler, "
                                               "les parties doivent considérer le recours aux modes privés.",
        code="Code de procédure civile du Québec",
        chemin_taxonomy="provincial_quebec/cpc/titre2_modes_prives_reglement"),
]

arts, stats = clean_articles(batch)
check("2 articles vivants retenus sur 5", len(arts), 2)
check("  compteur", stats, {"LIVE": 2, "REPEALED": 2, "OMITTED": 1})
# Un article long ne doit jamais être pris pour une souche, même s'il commence
# par une parenthèse (le corpus en contient : « (1) ... »).
check("article commençant par une parenthèse -> LIVE",
      classify_article("(1) Le tribunal peut, sur demande, ordonner la production d'un document "
                       "détenu par un tiers lorsque cela est nécessaire à la solution du litige."),
      "LIVE")
check("  aucune souche ne passe",
      any("(Abrogé" in a["texte"] or "(Omis" in a["texte"] for a in arts), False)

arts, stats = clean_articles(batch, codes=["Code civil du Québec"])
check("filtre --code CCQ", [a["id"] for a in arts], ["ccq_1"])
check("  lignes hors code comptées", stats.get("hors_code"), 2)

arts, _ = clean_articles(batch, limit=1)
check("filtre --limit", len(arts), 1)

# --------------------------------------------------------------------------
if "--live" in sys.argv:
    print("\n5. Contrôle sur le corpus réel (jeton HF requis)")
    from ccq_cleaner import load_quebec_articles
    arts, stats = load_quebec_articles()
    print(f"  articles retenus : {len(arts):,}")
    print(f"  compteur : {stats}")
    codes = {}
    for a in arts:
        codes[a["code"]] = codes.get(a["code"], 0) + 1
    print(f"  par code : {codes}")
    assert all(a["texte"] and a["domaine"] for a in arts), "champ vide détecté"
    print("  aucun champ vide.")

print(f"\n{'=' * 62}\n  {ok} réussis, {ko} échoués\n{'=' * 62}")
sys.exit(1 if ko else 0)
