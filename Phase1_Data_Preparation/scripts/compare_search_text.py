#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare le texte indexé : étiquettes + contenu, ou contenu seul.

Même protocole que la comparaison des recentrages — l'étalement d'abord,
qui est le diagnostic direct, puis le classement.

Le texte indexé alimente à la fois les vecteurs et BM25. La colonne
« dense seul » (dense_weight = 1.0) isole l'effet propre aux embeddings,
qui est l'objet de l'hypothèse.

    python scripts/compare_search_text.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1 / "tests"), str(PHASE1),
                  str(PHASE1 / "scripts")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG, _normalize_rows  # noqa: E402
from sweep_retrieval import embedder_and_gold, measure  # noqa: E402
from test_retrieval_gold import INDEX_DIR, K_VALUES  # noqa: E402

TEXTONLY_INDEX = PHASE1 / "data" / "agentic" / "rag_index_textonly"
RANKS = (1, 10, 50, 400)
DECISIVE = "ccq-mise-en-demeure"
VARIANTS = {
    "full": (INDEX_DIR, "full"),
    "text_only": (TEXTONLY_INDEX, "text_only"),
}


def build(name: str, embedder, **overrides) -> LegalRAG:
    index, fields = VARIANTS[name]
    production = load_config(
        str(PHASE1 / "configs" / "agentic_generation.yaml")).rag
    settings = {
        "index_dir": str(index), "top_k": max(K_VALUES),
        "llm_rerank_enabled": False,
        "candidate_k": production.candidate_k,
        "dense_weight": production.dense_weight,
        "min_dense_score": production.min_dense_score,
        "min_hybrid_score": production.min_hybrid_score,
        "search_text_fields": fields,
    }
    settings.update(overrides)
    return LegalRAG.load(RAGConfig(**settings), embedder)


def dense_profile(rag: LegalRAG, entries: list[dict]) -> dict:
    at_rank = {r: [] for r in RANKS}
    spreads, deviations = [], []
    decisive: dict[str, int | None] = {}
    for entry in entries:
        indices = np.asarray(
            [i for i, d in enumerate(rag.documents) if d.code == entry["code"]],
            dtype=np.int64)
        query = _normalize_rows(rag.embedder.embed([entry["question"]]))[0]
        dense = rag.embeddings[indices] @ query
        ordered = np.sort(dense)[::-1]
        for r in RANKS:
            if len(ordered) >= r:
                at_rank[r].append(float(ordered[r - 1]))
        if len(ordered) >= max(RANKS):
            spreads.append(float(ordered[0] - ordered[max(RANKS) - 1]))
        deviations.append(float(dense.std()))
        if entry["id"] == DECISIVE:
            numbers = [rag.documents[int(indices[p])].article_number
                       for p in np.argsort(-dense)]
            for article in ("1594", "1590"):
                decisive[article] = (numbers.index(article) + 1
                                     if article in numbers else None)
    mean = lambda v: round(sum(v) / len(v), 4) if v else 0.0
    return {
        "score_at_rank": {r: mean(at_rank[r]) for r in RANKS},
        "spread_1_to_400": mean(spreads),
        "std": mean(deviations),
        "decisive_dense_ranks": decisive,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="comparaison_search_text.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    embedder, entries = embedder_and_gold()
    names = list(VARIANTS)

    profiles = {n: dense_profile(build(n, embedder), entries) for n in names}
    print("═══ ÉTALEMENT DES SCORES DENSES (moyenne sur 52 questions) ═══\n")
    print(f"{'':22}" + "".join(f"{n:>14}" for n in names))
    print("-" * (22 + 14 * len(names)))
    for r in RANKS:
        print(f"{'score au rang ' + str(r):22}"
              + "".join(f"{profiles[n]['score_at_rank'][r]:>14.4f}"
                        for n in names))
    print(f"{'écart rang 1-400':22}"
          + "".join(f"{profiles[n]['spread_1_to_400']:>14.4f}" for n in names))
    print(f"{'écart-type':22}"
          + "".join(f"{profiles[n]['std']:>14.4f}" for n in names))

    print("\n═══ CAS DÉCISIF : rang dense de 1594 et 1590 ═══\n")
    for article in ("1594", "1590"):
        print(f"  article {article} : " + "  ".join(
            f"{n}={profiles[n]['decisive_dense_ranks'].get(article)}"
            for n in names))

    print("\n═══ CLASSEMENT ═══\n")
    quality = {}
    for label, extra in (("production (poids 0.6)", {}),
                         ("dense seul (poids 1.0)", {"dense_weight": 1.0}),
                         ("sans plancher", {"min_dense_score": 0.0})):
        print(f"{label}")
        print(f"{'':22}" + "".join(f"{n:>14}" for n in names))
        results = {n: measure(build(n, embedder, **extra), entries)
                   for n in names}
        quality[label] = results
        for metric, getter in (
                ("MRR", lambda q: q["mrr"]),
                ("hit@3", lambda q: q["hit_at"][3]),
                ("recall@3", lambda q: q["recall_at"][3]),
                ("recall@8", lambda q: q["recall_at"][8]),
                ("faux positifs", lambda q: q["false_positive_rate"]),
                ("vides", lambda q: q["empty"])):
            print(f"  {metric:20}" + "".join(
                f"{getter(results[n]):>14.4f}" for n in names))
        print()

    Path(args.output).write_text(
        json.dumps({"profiles": profiles, "quality": quality},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sortie brute : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
