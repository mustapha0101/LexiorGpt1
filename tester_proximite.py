# -*- coding: utf-8 -*-
"""Mesure la proximité vectorielle entre expressions, et situe un article
dans le classement dense d'une vraie question.

Usage, depuis Phase1_Data_Preparation/ :

    python tester_proximite.py --allow-remote-calls

Coût : une dizaine d'appels d'embedding, soit une fraction de centime.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "src")

from lexior.agentic.config import load_config                    # noqa: E402
from lexior.agentic.legal_rag import (                           # noqa: E402
    LegalRAG, OpenAIEmbedder, _normalize_rows,
)

# ─────────────────────────────────────────────────────────────────────
# À MODIFIER LIBREMENT
# ─────────────────────────────────────────────────────────────────────

PAIRES = [
    # (étiquette, expression A, expression B, attendu)
    ("LE CAS ÉTUDIÉ", "lettre d'avertissement", "mise en demeure", "?"),

    # témoins proches — le modèle DOIT les rapprocher
    ("témoin proche", "vice caché", "défaut caché", "proche"),
    ("témoin proche", "locateur", "propriétaire du logement", "proche"),
    ("témoin proche", "automobile", "voiture", "proche"),

    # témoins éloignés — le modèle DOIT les séparer
    ("témoin éloigné", "mise en demeure", "clôture mitoyenne", "éloigné"),
    ("témoin éloigné", "lettre d'avertissement", "moisissure", "éloigné"),

    # variantes du cas étudié
    ("variante", "avertir par écrit avant de poursuivre", "mise en demeure", "?"),
    ("variante", "lettre d'avertissement", "constituer en demeure", "?"),
    ("variante", "lettre de mise en demeure", "mise en demeure", "?"),
]

QUESTION = ("Est-ce que je suis obligé d'envoyer une lettre d'avertissement "
            "avant de poursuivre quelqu'un qui ne me paie pas?")
ARTICLES_ATTENDUS = ["1594", "1590"]
CODE = "CCQ"


def cosinus(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main() -> None:
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--allow-remote-calls", action="store_true")
    parseur.add_argument("--config", default="configs/agentic_generation.yaml")
    args = parseur.parse_args()

    if not args.allow_remote_calls:
        raise SystemExit("Relance avec --allow-remote-calls")

    cfg = load_config(args.config)
    embedder = OpenAIEmbedder(cfg.rag, allow_remote_calls=True)

    # ── PARTIE A : les expressions isolées ───────────────────────────
    print("\n" + "=" * 68)
    print("A. PROXIMITÉ ENTRE EXPRESSIONS")
    print("=" * 68)
    print("Échelle : 1,00 = identique, 0,00 = sans rapport.")
    print("Compare le CAS ÉTUDIÉ aux témoins pour situer le chiffre.\n")

    textes: list[str] = []
    for _, a, b, _ in PAIRES:
        textes.extend([a, b])
    vecteurs = _normalize_rows(embedder.embed(textes))

    for position, (etiquette, a, b, attendu) in enumerate(PAIRES):
        score = cosinus(vecteurs[2 * position], vecteurs[2 * position + 1])
        marque = "  ←" if etiquette == "LE CAS ÉTUDIÉ" else ""
        print(f"  {score:.3f}   {a}  /  {b}")
        print(f"          [{etiquette}, attendu : {attendu}]{marque}\n")

    # ── PARTIE B : la vraie question contre les vrais articles ───────
    print("=" * 68)
    print("B. LA VRAIE QUESTION CONTRE LE CORPUS")
    print("=" * 68)
    print("C'est ce que ton système compare réellement : la question\n"
          "complète contre le texte entier de chaque article.\n")

    rag = LegalRAG.load(cfg.rag, embedder)
    vecteur_question = _normalize_rows(embedder.embed([QUESTION]))[0]

    indices = [i for i, doc in enumerate(rag.documents) if doc.code == CODE]
    scores = rag.embeddings[indices] @ vecteur_question
    ordre = np.argsort(-scores)

    print(f"  Question : {QUESTION}\n")
    print(f"  Les 10 articles les plus proches sur {len(indices)} :\n")
    for rang, position in enumerate(ordre[:10], start=1):
        doc = rag.documents[indices[position]]
        extrait = doc.text[:64].replace("\n", " ")
        print(f"   {rang:3}.  {scores[position]:.3f}  "
              f"art. {doc.article_number:<8} {extrait}…")

    print(f"\n  Où se trouvent les articles attendus :\n")
    rang_par_article = {
        rag.documents[indices[p]].article_number: (r, float(scores[p]))
        for r, p in enumerate(ordre, start=1)
    }
    for numero in ARTICLES_ATTENDUS:
        if numero in rang_par_article:
            rang, score = rang_par_article[numero]
            print(f"   art. {numero} : rang {rang} sur {len(indices)}, "
                  f"score {score:.3f}")
        else:
            print(f"   art. {numero} : ABSENT DU CORPUS")

    print(f"\n  Rappel : la recherche ne garde que les "
          f"{cfg.rag.candidate_k} premiers candidats.\n")
    print(f"  Coût de ce test : {embedder.calls} appels.\n")


if __name__ == "__main__":
    main()