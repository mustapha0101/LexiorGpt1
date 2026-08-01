# -*- coding: utf-8 -*-
"""Met chaque question du jeu de test face au TEXTE RÉEL des articles attendus.

Produit un fichier markdown à lire ligne par ligne. Aucune connaissance
juridique requise : il s'agit de vérifier que l'article parle bien du sujet
de la question.

Usage, depuis Phase1_Data_Preparation/ :

    python relire_gold.py
    python relire_gold.py --sortie relecture.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GOLD = Path("tests/fixtures/retrieval_gold.jsonl")
INDEX = Path("data/agentic/rag_index/documents.jsonl")


def charger_articles() -> dict[tuple[str, str], dict]:
    """Indexe le corpus par (code, numéro d'article)."""
    if not INDEX.exists():
        raise SystemExit(
            f"Index introuvable : {INDEX}\n"
            "Lance d'abord : python -m agentic_generation build-rag-index"
        )
    par_cle: dict[tuple[str, str], dict] = {}
    with INDEX.open(encoding="utf-8") as handle:
        for ligne in handle:
            doc = json.loads(ligne)
            cle = (doc.get("code", ""), str(doc.get("article_number", "")))
            par_cle[cle] = doc
    return par_cle


def main() -> None:
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--sortie", default="relecture_gold.md")
    args = parseur.parse_args()

    articles = charger_articles()
    lignes_gold = [json.loads(l) for l in GOLD.open(encoding="utf-8")]

    sortie: list[str] = ["# Relecture du jeu de test\n"]
    sortie.append(
        "Pour chaque question : l'article attendu parle-t-il bien du sujet ?\n"
        "Coche `[x]` si oui, écris ta correction sinon.\n"
    )

    manquants: list[str] = []
    a_relire = [r for r in lignes_gold if r.get("answerable")]

    for numero, ligne in enumerate(a_relire, start=1):
        code = ligne.get("code", "CCQ")
        sortie.append(f"\n---\n\n## {numero}. `{ligne['id']}`\n")
        sortie.append(f"**Question** — {ligne['question']}\n")
        if ligne.get("note"):
            sortie.append(f"*Intention déclarée : {ligne['note']}*\n")

        for numero_article in ligne.get("expected_articles", []):
            doc = articles.get((code, str(numero_article)))
            sortie.append(f"\n**{code} art. {numero_article}**\n")
            if doc is None:
                manquants.append(f"{ligne['id']} → {code} {numero_article}")
                sortie.append(
                    "> ABSENT DU CORPUS — numéro inexistant ou article abrogé.\n")
                continue
            if doc.get("domain"):
                sortie.append(f"> *{doc['domain']}*\n")
            sortie.append("> " + doc.get("text", "").replace("\n", "\n> ") + "\n")

        sortie.append("\n- [ ] correspond\n- [ ] correction : \n")

    entete = [
        f"\n{len(a_relire)} questions à relire.",
        f"{len(lignes_gold) - len(a_relire)} questions sans réponse attendue "
        "(rien à vérifier).",
    ]
    if manquants:
        entete.append(
            f"\n**{len(manquants)} article(s) introuvable(s) dans le corpus — "
            "à corriger en priorité :**")
        entete.extend(f"- {m}" for m in manquants)
    sortie.insert(2, "\n".join(entete) + "\n")

    Path(args.sortie).write_text("\n".join(sortie), encoding="utf-8")
    print(f"Écrit : {args.sortie}")
    print(f"{len(a_relire)} questions, {len(manquants)} article(s) introuvable(s)")


if __name__ == "__main__":
    main()