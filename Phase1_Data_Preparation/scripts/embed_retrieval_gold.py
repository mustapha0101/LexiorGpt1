#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Embarque une seule fois les questions du jeu de test de recherche.

Le harnais de mesure (``tests/test_retrieval_gold.py``) interroge l'index RAG
réel. Il lui faut donc le vecteur de chaque question du fixture, calculé avec
le MÊME modèle d'embeddings que l'index — sinon les similarités ne veulent
rien dire.

Ce script fait cet appel distant une fois et met les vecteurs en cache dans
``tests/fixtures/retrieval_gold_queries.npz``. Les mesures suivantes tournent
hors-ligne, sans clé d'API, et restent reproductibles.

Usage :

    python scripts/embed_retrieval_gold.py --allow-remote-calls

Nécessite ``RAG_EMBEDDING_API_KEY`` (ou ``OPENAI_API_KEY``) dans
l'environnement. Coût attendu : moins de 0,001 $ US pour 52 questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic.config import load_config  # noqa: E402
from lexior.agentic.legal_rag import OpenAIEmbedder, RAGError  # noqa: E402

GOLD_PATH = PHASE1 / "tests" / "fixtures" / "retrieval_gold.jsonl"
CACHE_PATH = PHASE1 / "tests" / "fixtures" / "retrieval_gold_queries.npz"


def gold_digest(raw: bytes) -> str:
    """Empreinte du fixture, pour détecter un cache devenu obsolète."""
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-remote-calls", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="recalculer même si le cache est à jour")
    args = parser.parse_args()

    raw = GOLD_PATH.read_bytes()
    digest = gold_digest(raw)
    entries = [json.loads(line) for line in raw.decode("utf-8").splitlines()
               if line.strip()]
    if not entries:
        print(f"[gold] fixture vide : {GOLD_PATH}", file=sys.stderr)
        return 1

    if CACHE_PATH.exists() and not args.force:
        cached = np.load(CACHE_PATH, allow_pickle=False)
        if str(cached["gold_sha256"]) == digest:
            print(f"[gold] cache déjà à jour ({len(entries)} questions) : "
                  f"{CACHE_PATH}")
            return 0
        print("[gold] fixture modifié depuis le dernier cache, recalcul.")

    cfg = load_config(args.config, {"allow_remote_calls": args.allow_remote_calls})
    try:
        embedder = OpenAIEmbedder(cfg.rag, args.allow_remote_calls)
    except RAGError as error:
        print(f"[gold] {error}", file=sys.stderr)
        return 2

    identifiers = [str(entry["id"]) for entry in entries]
    questions = [str(entry["question"]) for entry in entries]
    print(f"[gold] embarquement de {len(questions)} questions avec "
          f"{embedder.model}...")
    vectors = embedder.embed(questions)
    if len(vectors) != len(questions):
        print("[gold] nombre de vecteurs incohérent", file=sys.stderr)
        return 3

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        ids=np.asarray(identifiers, dtype=object).astype("U64"),
        vectors=np.asarray(vectors, dtype=np.float32),
        model=np.asarray(embedder.model),
        gold_sha256=np.asarray(digest),
    )
    total = embedder.cost_report().get("total", {})
    print(f"[gold] écrit {CACHE_PATH} "
          f"({len(questions)} vecteurs, {vectors.shape[1]} dimensions)")
    print(f"[gold] appels {total.get('calls', 0)} | "
          f"jetons {total.get('tokens_in', 0)} | "
          f"coût ${float(total.get('cost_usd', 0.0)):.6f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
