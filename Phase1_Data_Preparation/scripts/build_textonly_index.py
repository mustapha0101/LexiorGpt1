#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ré-indexe le corpus en n'embarquant que le texte normatif.

L'index historique préfixe chaque article de quatre étiquettes — titre,
libellé, domaine, taxonomie. Elles pèsent 22 % du texte embarqué en
médiane et jusqu'à 71 % sur un article court, alors qu'il n'existe que 16
taxonomies distinctes pour 4 278 articles. Hypothèse : elles rapprochent
artificiellement les articles d'un même livre.

Le corpus est repris de l'index existant : une seule variable change, le
texte soumis à l'embedder. L'index historique n'est pas touché.

    python scripts/build_textonly_index.py --allow-remote-calls

Environ 0,01 $ US et quelques minutes (34 appels d'embeddings).
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

from lexior.agentic.config import load_config  # noqa: E402
from lexior.agentic.legal_rag import (  # noqa: E402
    LegalDocument, OpenAIEmbedder, RAGError, _normalize_rows, index_exists,
    search_text_for,
)

SOURCE_INDEX = PHASE1 / "data" / "agentic" / "rag_index"
TARGET_INDEX = PHASE1 / "data" / "agentic" / "rag_index_textonly"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--allow-remote-calls", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if not index_exists(SOURCE_INDEX):
        print(f"[texte] index source absent : {SOURCE_INDEX}", file=sys.stderr)
        return 1
    if index_exists(TARGET_INDEX) and not args.force:
        print(f"[texte] index déjà présent dans {TARGET_INDEX}; --force pour "
              "reconstruire")
        return 0

    documents = [
        LegalDocument(**json.loads(line))
        for line in (SOURCE_INDEX / "documents.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts = [search_text_for(document, "text_only") for document in documents]
    full = [search_text_for(document, "full") for document in documents]
    economy = 1 - sum(map(len, texts)) / sum(map(len, full))
    print(f"[texte] {len(documents)} articles repris de l'index existant")
    print(f"[texte] caractères embarqués : -{economy:.1%} sans les étiquettes",
          flush=True)

    cfg = load_config(args.config).rag
    try:
        embedder = OpenAIEmbedder(cfg, args.allow_remote_calls)
    except RAGError as error:
        print(f"[texte] {error}", file=sys.stderr)
        return 2

    started = time.monotonic()
    batches: list[np.ndarray] = []
    size = max(int(cfg.embedding_batch_size), 1)
    for start in range(0, len(texts), size):
        stop = min(start + size, len(texts))
        batches.append(embedder.embed(texts[start:stop]))
        usage = embedder.cost_report().get("total", {})
        print(f"[texte] {stop}/{len(texts)} | jetons {usage.get('tokens_in', 0)} "
              f"| ${float(usage.get('cost_usd', 0)):.6f}", flush=True)
    embeddings = _normalize_rows(np.vstack(batches))

    TARGET_INDEX.mkdir(parents=True, exist_ok=True)
    with (TARGET_INDEX / "documents.jsonl").open("w", encoding="utf-8") as h:
        for document in documents:
            h.write(json.dumps(document.__dict__, ensure_ascii=False) + "\n")
    with (TARGET_INDEX / "embeddings.npy").open("wb") as h:
        np.save(h, embeddings, allow_pickle=False)

    corpus_hash = hashlib.sha256(
        "\n".join(f"{d.id}:{d.text}" for d in documents).encode("utf-8")
    ).hexdigest()
    source = json.loads(
        (SOURCE_INDEX / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "version": 1,
        "dataset_name": source.get("dataset_name", ""),
        "dataset_split": source.get("dataset_split", ""),
        "embedding_provider": "openai",
        "embedding_model": embedder.model,
        "search_text_fields": "text_only",
        "documents": len(documents),
        "dimensions": int(embeddings.shape[1]),
        "corpus_hash": corpus_hash,
        "corpus_identique_a_reference": corpus_hash == source.get("corpus_hash"),
        "usage": embedder.cost_report(),
    }
    (TARGET_INDEX / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[texte] écrit {TARGET_INDEX} en "
          f"{(time.monotonic() - started) / 60:.1f} min")
    print(f"[texte] corpus identique à la référence : "
          f"{manifest['corpus_identique_a_reference']}")
    print(f"[texte] coût "
          f"${float(embedder.cost_report()['total']['cost_usd']):.6f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
