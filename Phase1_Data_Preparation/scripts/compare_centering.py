#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare trois recentrages des vecteurs, à coût nul.

Aucun appel d'API, aucune réindexation : les vecteurs du corpus sont sur
disque et ceux des questions sont en cache.

    A. none      — aucun recentrage
    B. global    — une moyenne sur les 4 278 articles
    C. per_code  — une moyenne CCQ, une moyenne CPC

L'ÉTALEMENT des scores est mesuré en premier : c'est le diagnostic direct
de l'hypothèse. Si les scores restent tassés, le reste n'a pas d'intérêt.

    python scripts/compare_centering.py
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

from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG, _normalize_rows  # noqa: E402
from lexior.agentic.storage import JsonCache  # noqa: E402
from lexior.agentic.teacher_client import TeacherClient  # noqa: E402
from test_retrieval_gold import (  # noqa: E402
    CachedEmbedder, INDEX_DIR, K_VALUES, QUERIES_PATH, load_gold,
)

MODES = ("none", "global", "per_code")
RANKS = (1, 10, 50, 400)
DECISIVE = "ccq-mise-en-demeure"


def _embedder():
    entries = load_gold()
    cached = np.load(QUERIES_PATH, allow_pickle=False)
    by_id = {e["id"]: e["question"] for e in entries}
    vectors = {by_id[str(i)]: cached["vectors"][p]
               for p, i in enumerate(cached["ids"]) if str(i) in by_id}
    return CachedEmbedder(str(cached["model"]), vectors), entries


def _rag(mode: str, embedder, reranker=None) -> LegalRAG:
    production = load_config(
        str(PHASE1 / "configs" / "agentic_generation.yaml")).rag
    cfg = RAGConfig(
        index_dir=str(INDEX_DIR), top_k=max(K_VALUES),
        llm_rerank_enabled=reranker is not None,
        llm_rerank_k=production.llm_rerank_k,
        min_dense_score=production.min_dense_score,
        min_hybrid_score=production.min_hybrid_score,
        centering=mode,
    )
    return LegalRAG.load(cfg, embedder, reranker=reranker)


def dense_profile(rag: LegalRAG, entries: list[dict]) -> dict:
    """Scores denses bruts sur TOUT le sous-corpus du code, par question.

    C'est la mesure de l'hypothèse : le recentrage étale-t-il des scores
    tassés dans un dixième ?
    """
    at_rank = {r: [] for r in RANKS}
    spreads, deviations = [], []
    decisive: dict[str, int | None] = {}

    for entry in entries:
        indices = np.asarray(
            [i for i, d in enumerate(rag.documents) if d.code == entry["code"]],
            dtype=np.int64)
        query = _normalize_rows(rag.embedder.embed([entry["question"]]))[0]
        mean = rag._mean_for(entry["code"])
        if mean is not None:
            query = rag._recenter(query, mean)[0]
        dense = rag._centered_matrix(entry["code"], indices) @ query
        ordered = np.sort(dense)[::-1]

        for r in RANKS:
            if len(ordered) >= r:
                at_rank[r].append(float(ordered[r - 1]))
        if len(ordered) >= max(RANKS):
            spreads.append(float(ordered[0] - ordered[max(RANKS) - 1]))
        deviations.append(float(dense.std()))

        if entry["id"] == DECISIVE:
            ranking = np.argsort(-dense)
            numbers = [rag.documents[int(indices[p])].article_number
                       for p in ranking]
            for article in ("1594", "1590"):
                decisive[article] = (numbers.index(article) + 1
                                     if article in numbers else None)

    mean_of = lambda v: round(sum(v) / len(v), 4) if v else 0.0
    return {
        "score_at_rank": {r: mean_of(at_rank[r]) for r in RANKS},
        "spread_1_to_400": mean_of(spreads),
        "std": mean_of(deviations),
        "decisive_ranks": decisive,
    }


def quality(rag: LegalRAG, entries: list[dict]) -> dict:
    largest = max(K_VALUES)
    hits = {k: [] for k in K_VALUES}
    recall = {k: [] for k in K_VALUES}
    reciprocal, false_positives, empty = [], [], []
    for entry in entries:
        retrieved = [i["article_number"]
                     for i in rag.search(entry["question"], entry["code"],
                                         largest)]
        if not entry["answerable"]:
            if retrieved:
                false_positives.append(entry["id"])
            continue
        expected = set(entry["expected_articles"])
        if not retrieved:
            empty.append(entry["id"])
        for k in K_VALUES:
            found = expected.intersection(retrieved[:k])
            hits[k].append(1.0 if found else 0.0)
            recall[k].append(len(found) / len(expected))
        rank = next((p for p, n in enumerate(retrieved, 1) if n in expected), 0)
        reciprocal.append(1.0 / rank if rank else 0.0)
    unanswerable = [e for e in entries if not e["answerable"]]
    mean_of = lambda v: round(sum(v) / len(v), 4) if v else 0.0
    return {
        "hit_at": {k: mean_of(hits[k]) for k in K_VALUES},
        "recall_at": {k: mean_of(recall[k]) for k in K_VALUES},
        "mrr": mean_of(reciprocal),
        "false_positive_rate": round(
            len(false_positives) / len(unanswerable), 4),
        "false_positive_ids": false_positives,
        "empty_on_answerable_ids": empty,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="comparaison_centering.json")
    parser.add_argument("--rerank-best", action="store_true",
                        help="mesurer AUSSI la meilleure config avec rerank")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    embedder, entries = _embedder()
    profiles, qualities = {}, {}
    for mode in MODES:
        rag = _rag(mode, embedder)
        profiles[mode] = dense_profile(rag, entries)
        qualities[mode] = quality(rag, entries)

    print("═══ ÉTALEMENT DES SCORES DENSES (moyenne sur 52 questions) ═══\n")
    print(f"{'':22}" + "".join(f"{m:>14}" for m in MODES))
    print("-" * (22 + 14 * 3))
    for r in RANKS:
        print(f"{'score au rang ' + str(r):22}"
              + "".join(f"{profiles[m]['score_at_rank'][r]:>14.4f}"
                        for m in MODES))
    print(f"{'écart rang 1-400':22}"
          + "".join(f"{profiles[m]['spread_1_to_400']:>14.4f}" for m in MODES))
    print(f"{'écart-type':22}"
          + "".join(f"{profiles[m]['std']:>14.4f}" for m in MODES))

    print("\n═══ CAS DÉCISIF : rang dense de 1594 et 1590 ═══\n")
    for article in ("1594", "1590"):
        print(f"  article {article} : "
              + "  ".join(f"{m}={profiles[m]['decisive_ranks'].get(article)}"
                          for m in MODES))
    print("  (candidate_k = 40 : au-delà, l'article n'est jamais candidat)")

    print("\n═══ QUALITÉ, SANS RERANK ═══\n")
    rows = [
        ("MRR", lambda q: q["mrr"]),
        ("hit@3", lambda q: q["hit_at"][3]),
        ("recall@3", lambda q: q["recall_at"][3]),
        ("recall@8", lambda q: q["recall_at"][8]),
        ("faux positifs", lambda q: q["false_positive_rate"]),
    ]
    print(f"{'':22}" + "".join(f"{m:>14}" for m in MODES))
    print("-" * (22 + 14 * 3))
    for name, getter in rows:
        print(f"{name:22}" + "".join(f"{getter(qualities[m]):>14.4f}"
                                     for m in MODES))
    for m in MODES:
        print(f"  {m} : questions vidées "
              f"{qualities[m]['empty_on_answerable_ids']}")

    payload = {"profiles": profiles, "quality": qualities}

    if args.rerank_best:
        best = max(MODES, key=lambda m: qualities[m]["mrr"])
        print(f"\n═══ RERANK, uniquement sur la meilleure config : {best} ═══\n")
        cfg = load_config(str(PHASE1 / "configs" / "agentic_generation.yaml"))
        teacher = TeacherClient(
            cfg.teacher, allow_remote_calls=True,
            cache=JsonCache(PHASE1 / "data" / "agentic" / "cache"
                            / "rerank-centering"),
            cache_extra_key=f"centering-{best}")
        result = quality(_rag(best, embedder, teacher), entries)
        for name, getter in rows:
            print(f"{name:22}{getter(result):>14.4f}")
        print(f"  questions vidées {result['empty_on_answerable_ids']}")
        payload["rerank"] = {"mode": best, "quality": result,
                             "usage": teacher.cost_report()}

    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsortie brute : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
