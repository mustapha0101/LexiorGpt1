#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Construit l'index BGE-M3 à côté de l'index OpenAI, sans y toucher.

Le corpus est repris tel quel depuis l'index existant plutôt que rechargé
depuis Hugging Face : la comparaison ne doit avoir QU'UNE variable, le
modèle d'embeddings. Recharger le corpus risquerait d'introduire une
seconde différence si la source a bougé entre-temps.

    python scripts/build_bge_index.py

Environ 37 minutes sur CPU pour 4 278 articles (mesuré, 16 cœurs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import (  # noqa: E402
    BGEEmbedder,
    LegalDocument,
    _normalize_rows,
    index_exists,
)

SOURCE_INDEX = PHASE1 / "data" / "agentic" / "rag_index"
TARGET_INDEX = PHASE1 / "data" / "agentic" / "rag_index_bge"
MODEL = "BAAI/bge-m3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=-1,
                        help="n'indexer que les N premiers (mise au point)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if not index_exists(SOURCE_INDEX):
        print(f"[bge] index source absent : {SOURCE_INDEX}", file=sys.stderr)
        return 1
    if index_exists(TARGET_INDEX) and not args.force:
        print(f"[bge] index déjà présent dans {TARGET_INDEX}; --force pour "
              "reconstruire")
        return 0

    documents = [
        LegalDocument(**json.loads(line))
        for line in (SOURCE_INDEX / "documents.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        documents = documents[:args.limit]
    print(f"[bge] corpus repris de l'index OpenAI : {len(documents)} articles",
          flush=True)

    cfg = RAGConfig(index_dir=str(TARGET_INDEX), embedding_provider="bge",
                    embedding_model=MODEL)
    print(f"[bge] chargement de {MODEL} sur CPU…", flush=True)
    embedder = BGEEmbedder(cfg)

    texts = [document.search_text for document in documents]
    batches: list[np.ndarray] = []
    size = 64
    started = time.monotonic()
    for start in range(0, len(texts), size):
        stop = min(start + size, len(texts))
        batches.append(embedder.embed(texts[start:stop]))
        elapsed = time.monotonic() - started
        rate = stop / elapsed if elapsed else 0.0
        remaining = (len(texts) - stop) / rate if rate else 0.0
        print(f"[bge] {stop}/{len(texts)} | {elapsed / 60:.1f} min écoulées | "
              f"{rate:.1f} art./s | reste ~{remaining / 60:.1f} min",
              flush=True)

    embeddings = _normalize_rows(np.vstack(batches))
    if len(embeddings) != len(documents):
        print("[bge] incohérence embeddings/documents", file=sys.stderr)
        return 2

    TARGET_INDEX.mkdir(parents=True, exist_ok=True)
    with (TARGET_INDEX / "documents.jsonl").open("w", encoding="utf-8") as h:
        for document in documents:
            h.write(json.dumps(document.__dict__, ensure_ascii=False) + "\n")
    with (TARGET_INDEX / "embeddings.npy").open("wb") as h:
        np.save(h, embeddings, allow_pickle=False)

    corpus_hash = hashlib.sha256(
        "\n".join(f"{d.id}:{d.text}" for d in documents).encode("utf-8")
    ).hexdigest()
    source_manifest = json.loads(
        (SOURCE_INDEX / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "version": 1,
        "dataset_name": source_manifest.get("dataset_name", ""),
        "dataset_split": source_manifest.get("dataset_split", ""),
        "embedding_provider": "bge",
        "embedding_model": MODEL,
        "documents": len(documents),
        "dimensions": int(embeddings.shape[1]),
        "corpus_hash": corpus_hash,
        "corpus_identique_a_openai": (
            corpus_hash == source_manifest.get("corpus_hash")),
        "usage": embedder.cost_report(),
    }
    (TARGET_INDEX / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = time.monotonic() - started
    print(f"\n[bge] écrit {TARGET_INDEX}")
    print(f"[bge] {len(documents)} articles, {embeddings.shape[1]} dimensions, "
          f"{total / 60:.1f} min")
    print(f"[bge] corpus identique à l'index OpenAI : "
          f"{manifest['corpus_identique_a_openai']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
