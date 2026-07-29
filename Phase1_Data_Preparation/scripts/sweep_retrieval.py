#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trois balayages gratuits du retrieval : poids, candidats, expansions.

Aucun appel d'API : vecteurs du corpus sur disque, vecteurs des questions
en cache.

Le plancher dense et le poids dense agissent sur le MÊME canal : les
candidats entrent par une union (top-k dense OU top-k BM25), le plancher
n'écarte que sur le score de sens, et le classement final mélange les
deux. Le balayage du poids est donc fait DEUX fois, plancher actif puis
désactivé.

    python scripts/sweep_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1 / "tests"), str(PHASE1)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic import legal_rag  # noqa: E402
from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG, _normalize_rows  # noqa: E402
from test_retrieval_gold import (  # noqa: E402
    CachedEmbedder, INDEX_DIR, K_VALUES, QUERIES_PATH, load_gold,
)

DECISIVE = "ccq-mise-en-demeure"


def embedder_and_gold():
    entries = load_gold()
    cached = np.load(QUERIES_PATH, allow_pickle=False)
    by_id = {e["id"]: e["question"] for e in entries}
    vectors = {by_id[str(i)]: cached["vectors"][p]
               for p, i in enumerate(cached["ids"]) if str(i) in by_id}
    return CachedEmbedder(str(cached["model"]), vectors), entries


def measure(rag: LegalRAG, entries: list[dict]) -> dict:
    largest = max(K_VALUES)
    hits, recall, reciprocal = {k: [] for k in K_VALUES}, {k: [] for k in K_VALUES}, []
    false_positives, empty = [], []
    for entry in entries:
        found = [i["article_number"]
                 for i in rag.search(entry["question"], entry["code"], largest)]
        if not entry["answerable"]:
            if found:
                false_positives.append(entry["id"])
            continue
        expected = set(entry["expected_articles"])
        if not found:
            empty.append(entry["id"])
        for k in K_VALUES:
            got = expected.intersection(found[:k])
            hits[k].append(1.0 if got else 0.0)
            recall[k].append(len(got) / len(expected))
        rank = next((p for p, n in enumerate(found, 1) if n in expected), 0)
        reciprocal.append(1.0 / rank if rank else 0.0)
    unanswerable = [e for e in entries if not e["answerable"]]
    mean = lambda v: round(sum(v) / len(v), 4) if v else 0.0
    return {
        "mrr": mean(reciprocal),
        "hit_at": {k: mean(hits[k]) for k in K_VALUES},
        "recall_at": {k: mean(recall[k]) for k in K_VALUES},
        "false_positive_rate": round(
            len(false_positives) / len(unanswerable), 4),
        "empty": len(empty),
    }


def build(embedder, **overrides) -> LegalRAG:
    production = load_config(
        str(PHASE1 / "configs" / "agentic_generation.yaml")).rag
    settings = {
        "index_dir": str(INDEX_DIR), "top_k": max(K_VALUES),
        "llm_rerank_enabled": False,
        "candidate_k": production.candidate_k,
        "dense_weight": production.dense_weight,
        "min_dense_score": production.min_dense_score,
        "min_hybrid_score": production.min_hybrid_score,
    }
    settings.update(overrides)
    return LegalRAG.load(RAGConfig(**settings), embedder)


def decisive_ranks(embedder, entries, **overrides) -> dict:
    """Rangs DENSE et BM25 de 1594/1590 sur tout le sous-corpus CCQ."""
    rag = build(embedder, **overrides)
    entry = next(e for e in entries if e["id"] == DECISIVE)
    indices = np.asarray(
        [i for i, d in enumerate(rag.documents) if d.code == entry["code"]],
        dtype=np.int64)
    query = _normalize_rows(embedder.embed([entry["question"]]))[0]
    dense = rag.embeddings[indices] @ query
    lexical = rag._bm25(
        legal_rag._expanded_query_tokens(entry["question"]), indices)
    out = {}
    for name, scores in (("dense", dense), ("bm25", lexical)):
        numbers = [rag.documents[int(indices[p])].article_number
                   for p in np.argsort(-scores)]
        out[name] = {a: (numbers.index(a) + 1 if a in numbers else None)
                     for a in ("1594", "1590")}
    final = [i["article_number"]
             for i in rag.search(entry["question"], entry["code"], 8)]
    out["final_top8"] = final
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="sweep_retrieval.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    embedder, entries = embedder_and_gold()
    payload: dict = {}

    # ── 1. Poids dense, plancher actif puis désactivé ────────────────────
    print("═══ 1. dense_weight — plancher 0.40 puis désactivé ═══\n")
    print(f"{'poids':>6}  {'MRR':>7} {'hit@3':>7} {'rec@8':>7} {'FP':>6} {'vides':>6}"
          f"   |  {'MRR':>7} {'hit@3':>7} {'rec@8':>7} {'FP':>6} {'vides':>6}")
    print(f"{'':6}  {'---- plancher 0.40 ----':^36}   |  {'---- sans plancher ----':^36}")
    sweep = {}
    for step in range(11):
        weight = round(step / 10, 1)
        row = {}
        for label, floor in (("avec", None), ("sans", 0.0)):
            extra = {"dense_weight": weight}
            if floor is not None:
                extra["min_dense_score"] = floor
            row[label] = measure(build(embedder, **extra), entries)
        sweep[weight] = row
        a, s = row["avec"], row["sans"]
        print(f"{weight:>6.1f}  {a['mrr']:>7.4f} {a['hit_at'][3]:>7.4f} "
              f"{a['recall_at'][8]:>7.4f} {a['false_positive_rate']:>6.3f} "
              f"{a['empty']:>6d}   |  {s['mrr']:>7.4f} {s['hit_at'][3]:>7.4f} "
              f"{s['recall_at'][8]:>7.4f} {s['false_positive_rate']:>6.3f} "
              f"{s['empty']:>6d}")
    payload["dense_weight"] = sweep

    # ── 2. Nombre de candidats ───────────────────────────────────────────
    print("\n═══ 2. candidate_k ═══\n")
    print(f"{'k':>6}  {'MRR':>7} {'hit@3':>7} {'rec@3':>7} {'rec@8':>7} "
          f"{'FP':>6} {'vides':>6}")
    candidates = {}
    for k in (40, 100, 200):
        result = measure(build(embedder, candidate_k=k), entries)
        candidates[k] = result
        print(f"{k:>6}  {result['mrr']:>7.4f} {result['hit_at'][3]:>7.4f} "
              f"{result['recall_at'][3]:>7.4f} {result['recall_at'][8]:>7.4f} "
              f"{result['false_positive_rate']:>6.3f} {result['empty']:>6d}")
    payload["candidate_k"] = candidates

    # ── 3. Expansion « avertissement → demeure » ─────────────────────────
    print("\n═══ 3. expansion « avertissement → demeure » ═══\n")
    original = dict(legal_rag.QUERY_EXPANSIONS)
    expansions = {}
    for label, extra in (("sans", {}), ("avec", {
            "avertissement": ("demeure", "demeurer"),
            "avertir": ("demeure", "demeurer"),
    })):
        legal_rag.QUERY_EXPANSIONS.clear()
        legal_rag.QUERY_EXPANSIONS.update({**original, **extra})
        result = measure(build(embedder), entries)
        ranks = decisive_ranks(embedder, entries)
        ranks_no_floor = decisive_ranks(embedder, entries, min_dense_score=0.0)
        expansions[label] = {"quality": result, "ranks": ranks,
                             "ranks_no_floor": ranks_no_floor}
        print(f"{label:>5} : MRR {result['mrr']:.4f} | hit@3 "
              f"{result['hit_at'][3]:.4f} | rec@8 {result['recall_at'][8]:.4f} "
              f"| FP {result['false_positive_rate']:.3f}")
        print(f"        rang BM25   1594={ranks['bm25']['1594']} "
              f"1590={ranks['bm25']['1590']}")
        print(f"        rang dense  1594={ranks['dense']['1594']} "
              f"1590={ranks['dense']['1590']}")
        print(f"        top-8 final {ranks['final_top8']}")
    legal_rag.QUERY_EXPANSIONS.clear()
    legal_rag.QUERY_EXPANSIONS.update(original)
    payload["expansion"] = expansions

    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsortie brute : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
