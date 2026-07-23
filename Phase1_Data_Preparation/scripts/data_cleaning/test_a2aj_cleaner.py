#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests de la couche de nettoyage A2AJ.

    python test_a2aj_cleaner.py            # tests unitaires (hors ligne)
    python test_a2aj_cleaner.py --live     # + contrôle sur le corpus réel

Aucun appel au modèle Teacher n'est effectué.
"""

import json
import sys

from a2aj_cleaner import classify_section, is_wholly_repealed, clean_law

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
print("\n1. classify_section — étiquetage des articles")

check("article de fond",
      classify_section("Le ministre peut, par règlement, fixer les droits exigibles "
                       "pour la délivrance d'un permis au titre de la présente partie."),
      "LIVE")
check("[blank]", classify_section("[blank]"), "BLANK")
check("[Abrogé, ...]", classify_section("[Abrogé, 2017, ch. 33, art. 228]"), "DEAD")
check("[Abrogée, ...]", classify_section("[Abrogée, 2009, ch. 23, art. 313]"), "DEAD")
check("[Abrogés, ...]", classify_section("[Abrogés, 1998, ch. 26, art. 72]"), "DEAD")
check("[Repealed, ...]", classify_section("[Repealed, 2013, c. 40, s. 268]"), "DEAD")
check("abrogé avant entrée en vigueur",
      classify_section("[Abrogé avant d'entrer en vigueur, 2008, ch. 20, art. 3]"), "DEAD")
check("[Modifications]", classify_section("[Modifications]"), "SPENT")
check("[Modification]", classify_section("[Modification]"), "SPENT")
# "[Abrogation]" = disposition abrogeant une AUTRE loi ; elle apparaît dans des
# lois en vigueur (art. 89 de la Loi sur la radiodiffusion). Ce n'est pas un
# article abrogé : SPENT, et non DEAD.
check("[Abrogation] -> SPENT et non DEAD", classify_section("[Abrogation]"), "SPENT")
check("[Abrogations] -> SPENT", classify_section("[Abrogations]"), "SPENT")
check("chaîne vide", classify_section(""), "EMPTY")
check("None", classify_section(None), "EMPTY")
check("trop court", classify_section("Définitions."), "TOO_SHORT")
check("souche inconnue", classify_section("[Voir Loi sur le Bureau du surintendant]"), "STUB_OTHER")

# Le cas central : un article VIVANT dont un sous-alinéa est biffé.
# Il doit rester exploitable — c'est le "loi non abrogée mais dont certains
# articles le sont" du cahier des charges.
partial = ("Les définitions qui suivent s'appliquent à la présente loi.\n"
           "a) « navire » Tout bâtiment conçu pour la navigation;\n"
           "b) [Abrogée, 2014, ch. 29, art. 3]\n"
           "c) « ministre » Le ministre des Transports.")
check("article vivant avec sous-alinéa abrogé -> PARTIAL", classify_section(partial), "PARTIAL")

# --------------------------------------------------------------------------
print("\n2. is_wholly_repealed — abrogation de la loi entière")

check("en-tête marqué abrogé -> abrogée",
      is_wholly_repealed("# Loi sur l'énergie nucléaire\n\n*LC 1985*\n\n[Abrogé, 1997, ch. 9, art. 88]",
                         {"1": "LIVE", "2": "LIVE", "3": "DEAD"}),
      True)
check("tous les articles morts -> abrogée",
      is_wholly_repealed("# Une loi quelconque", {"1": "DEAD", "2": "DEAD", "3": "BLANK"}),
      True)
check("en-tête propre + articles vivants -> en vigueur",
      is_wholly_repealed("# Loi sur l'accès à l'information", {"1": "LIVE", "16.31": "DEAD"}),
      False)
# Le piège des noms : une loi d'abrogation est une loi ACTIVE.
check("« Loi sur l'abrogation des lois » -> en vigueur (piège du nom)",
      is_wholly_repealed("# Loi sur l'abrogation des lois\n\n*LC 2008, ch. 20*",
                         {"1": "LIVE", "2": "LIVE"}),
      False)
check("« Loi d'abrogation de la Loi sur les titres » -> en vigueur",
      is_wholly_repealed("# Loi d'abrogation de la Loi sur les titres de biens-fonds fédéraux",
                         {"1": "LIVE"}),
      False)

# --------------------------------------------------------------------------
print("\n3. clean_law — filtrage au niveau de la loi")


def law(**kw):
    base = {
        "dataset": "LEGISLATION-FED",
        "name_fr": "Loi de test",
        "citation_fr": "LC 2001, c 1",
        "source_url_fr": "https://laws-lois.justice.gc.ca/fra/XML/T-1.xml",
        "unofficial_text_fr": "# Loi de test\n\n*LC 2001, c 1*\n\n" + ("Texte de fond. " * 30),
        "unofficial_sections_fr": json.dumps({
            "1": "Le ministre peut fixer par règlement les droits exigibles au titre de la présente partie.",
            "2": "[Abrogé, 2015, ch. 3, art. 9]",
            "3": "[blank]",
            "4": "[Modifications]",
            "5": "Quiconque contrevient à l'article 1 commet une infraction et encourt une amende.",
        }),
    }
    base.update(kw)
    return base


r = clean_law(law())
check("loi fédérale valide -> acceptée", r is not None, True)
check("  seuls les articles vivants sont retenus", r["meta"]["section_ids_used"], ["1", "5"])
check("  l'article abrogé est absent du contexte", "[Abrogé" in r["context_text"], False)
check("  [blank] est absent du contexte", "[blank]" in r["context_text"], False)
check("  [Modifications] est absent du contexte", "[Modifications]" in r["context_text"], False)
check("  compteur de classes", r["stats"],
      {"LIVE": 2, "DEAD": 1, "BLANK": 1, "SPENT": 1})

# Compatibilité descendante avec generator_a2aj.py (:160 et :175)
check("  clé 'url' conservée pour la réécriture d'URL",
      r["meta"]["url"], "https://laws-lois.justice.gc.ca/fra/XML/T-1.xml")
check("  clé 'title' conservée", r["meta"]["title"], "Loi de test")

check("provincial -> rejeté",
      clean_law(law(dataset="LEGISLATION-ON")) is None, True)
check("territorial -> rejeté",
      clean_law(law(dataset="REGULATIONS-YT")) is None, True)
check("règlement fédéral -> accepté",
      clean_law(law(dataset="REGULATIONS-FED")) is not None, True)
check("loi entièrement abrogée (en-tête) -> rejeté",
      clean_law(law(unofficial_text_fr="# Loi morte\n\n[Abrogée, 2017, ch. 33, art. 228]\n" + ("x " * 100))) is None,
      True)
check("tous articles morts -> rejeté",
      clean_law(law(unofficial_sections_fr=json.dumps({"1": "[Abrogé, 2015, ch. 3]", "2": "[blank]"}))) is None,
      True)
check("aucun article analysé -> rejeté",
      clean_law(law(unofficial_sections_fr="")) is None, True)
check("texte français absent -> rejeté",
      clean_law(law(unofficial_text_fr="")) is None, True)
check("JSON d'articles corrompu -> rejeté",
      clean_law(law(unofficial_sections_fr="{pas du json")) is None, True)

# Budget de contexte : coupure sur une frontière d'article, jamais au milieu.
big = clean_law(law(unofficial_sections_fr=json.dumps(
    {str(i): "Disposition de fond numéro %d. %s" % (i, "texte " * 60) for i in range(1, 40)})),
    max_context_chars=1000)
# Contrôle strict : les séparateurs "\n\n" comptent dans le budget.
check("budget respecté à la lettre", len(big["context_text"]) <= 1000, True)
check("  coupure sur frontière d'article", big["context_text"].rstrip().endswith("texte"), True)
check("  section_ids_used reflète ce qui est réellement envoyé",
      len(big["meta"]["section_ids_used"]) < 39, True)

# Régression : un premier article à lui seul plus long que le budget ne doit
# PAS faire rejeter la loi (il avait fait disparaître 376 lois du corpus réel).
long_first = clean_law(law(unofficial_sections_fr=json.dumps({
    "1": "Disposition unique et très longue. " + ("texte " * 2000),
    "2": "Deuxième disposition de fond, courte mais bien réelle et exploitable.",
})), max_context_chars=4000)
check("premier article > budget -> loi conservée, article tronqué",
      long_first is not None, True)
check("  budget respecté malgré la troncature",
      len(long_first["context_text"]) <= 4000, True)
check("  l'article tronqué est bien déclaré", long_first["meta"]["section_ids_used"], ["1"])

# --- whole_only : n'accepter que les lois entières -------------------------
print("\n3b. clean_law(whole_only=True) — écarter au lieu de tronquer")

court = law(unofficial_sections_fr=json.dumps({
    "1": "Le ministre peut fixer par règlement les droits exigibles au titre de la présente partie.",
    "2": "Quiconque contrevient à l'article 1 commet une infraction et encourt une amende.",
}))
long_ = law(unofficial_sections_fr=json.dumps(
    {str(i): "Disposition de fond numéro %d. %s" % (i, "texte " * 60) for i in range(1, 40)}))

check("loi entière -> acceptée même en whole_only",
      clean_law(court, max_context_chars=4000, whole_only=True) is not None, True)
check("loi trop longue -> ÉCARTÉE en whole_only",
      clean_law(long_, max_context_chars=4000, whole_only=True) is None, True)
check("loi trop longue -> tronquée (et gardée) sans whole_only",
      clean_law(long_, max_context_chars=4000, whole_only=False) is not None, True)
# Un article isolé plus long que le budget est une troncature : à écarter aussi.
mono = law(unofficial_sections_fr=json.dumps({"1": "Disposition unique. " + ("texte " * 2000)}))
check("article unique > budget -> écarté en whole_only",
      clean_law(mono, max_context_chars=4000, whole_only=True) is None, True)
check("  ... mais conservé tronqué sans whole_only",
      clean_law(mono, max_context_chars=4000, whole_only=False) is not None, True)
# whole_only ne doit jamais perdre d'article d'une loi acceptée.
r_whole = clean_law(court, max_context_chars=4000, whole_only=True)
check("aucun article perdu dans une loi acceptée",
      r_whole["meta"]["section_ids_used"], ["1", "2"])

# Ordre naturel des articles (1, 2, 5.1, 10) et non alphabétique (1, 10, 2)
order = clean_law(law(unofficial_sections_fr=json.dumps({
    "10": "Dixième disposition de fond du texte législatif de test ici.",
    "2":  "Deuxième disposition de fond du texte législatif de test ici.",
    "5.1": "Disposition cinq point un du texte législatif de test ici.",
    "1":  "Première disposition de fond du texte législatif de test ici.",
})))
check("articles triés naturellement", order["meta"]["section_ids_used"], ["1", "2", "5.1", "10"])

# --------------------------------------------------------------------------
if "--live" in sys.argv:
    print("\n4. Contrôle sur le corpus réel (a2aj/canadian-laws)")
    from datasets import load_dataset
    ds = load_dataset("a2aj/canadian-laws", split="train")
    seen = {}
    kept = 0
    for item in ds:
        if clean_law(item):
            kept += 1
        nm = item.get("name_fr")
        if nm in ("Loi sur l’énergie nucléaire", "Loi sur l’accès à l’information",
                  "Loi sur l’abrogation des lois"):
            seen[nm] = "KEPT" if clean_law(item) else "REJECTED"
    print(f"  lois retenues : {kept:,} / {len(ds):,}")
    for k, v in seen.items():
        print(f"  {k[:50]:52s} -> {v}")

print(f"\n{'=' * 62}\n  {ok} réussis, {ko} échoués\n{'=' * 62}")
sys.exit(1 if ko else 0)
