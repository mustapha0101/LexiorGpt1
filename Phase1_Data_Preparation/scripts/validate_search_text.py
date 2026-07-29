#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trois contrôles avant toute adoption du texte indexé.

1. variante C — texte + domaine + taxonomie, sans les deux étiquettes de
   pure répétition (titre et libellé d'article);
2. surajustement — balayage sur une moitié du jeu de test, vérification du
   gagnant sur l'autre moitié, jamais utilisée pour choisir;
3. plancher — après recalibrage, élimine-t-il toujours du bruit et non des
   réponses ?

    python scripts/validate_search_text.py
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

from lexior.agentic import legal_rag  # noqa: E402
from lexior.agentic.config import RAGConfig  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG, _normalize_rows  # noqa: E402
from sweep_retrieval import embedder_and_gold, measure  # noqa: E402
from test_retrieval_gold import INDEX_DIR  # noqa: E402

DATA = PHASE1 / "data" / "agentic"
INDEXES = {
    "full": INDEX_DIR,
    "text_taxonomy": DATA / "rag_index_taxonomy",
    "text_only": DATA / "rag_index_textonly",
}


def build(vectors: str, bm25: str, weight: float, floor: float,
          embedder, **extra) -> LegalRAG:
    settings = {
        "index_dir": str(INDEXES[vectors]), "top_k": 8,
        "llm_rerank_enabled": False, "dense_weight": weight,
        "min_dense_score": floor, "search_text_fields": bm25,
    }
    settings.update(extra)
    return LegalRAG.load(RAGConfig(**settings), embedder)


def split_halves(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deux moitiés stables, chacune gardant la proportion de négatives."""
    first, second = [], []
    for answerable in (True, False):
        subset = sorted((e for e in entries if e["answerable"] is answerable),
                        key=lambda e: e["id"])
        first.extend(subset[0::2])
        second.extend(subset[1::2])
    return first, second


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="validation_search_text.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    embedder, entries = embedder_and_gold()
    payload: dict = {}

    # ── 1. Variante C ────────────────────────────────────────────────────
    print("═══ 1. VARIANTE C : texte + domaine + taxonomie ═══\n")
    print(f"{'vecteurs':>14} {'poids':>6} | {'MRR':>7} {'hit@3':>7} "
          f"{'rec@3':>7} {'rec@8':>7} {'FP':>7}")
    print("-" * 62)
    variant = {}
    for vectors in ("full", "text_taxonomy", "text_only"):
        for weight in (0.6, 0.8, 1.0):
            bm25 = "full" if vectors == "full" else "full"
            result = measure(build(vectors, bm25, weight, 0.40, embedder),
                             entries)
            variant[f"{vectors}@{weight}"] = result
            print(f"{vectors:>14} {weight:>6.1f} | {result['mrr']:>7.4f} "
                  f"{result['hit_at'][3]:>7.4f} {result['recall_at'][3]:>7.4f} "
                  f"{result['recall_at'][8]:>7.4f} "
                  f"{result['false_positive_rate']:>7.3f}")
    payload["variant_c"] = variant

    # ── 2. Surajustement ─────────────────────────────────────────────────
    print("\n═══ 2. SURAJUSTEMENT : choisir sur une moitié, vérifier sur "
          "l'autre ═══\n")
    first, second = split_halves(entries)
    print(f"moitié A : {len(first)} questions "
          f"({sum(1 for e in first if e['answerable'])} répondables)")
    print(f"moitié B : {len(second)} questions "
          f"({sum(1 for e in second if e['answerable'])} répondables)\n")

    grid = [(v, b, w) for v in ("full", "text_taxonomy", "text_only")
            for b in ("full", "text_only") for w in (0.6, 0.7, 0.8, 0.9, 1.0)]
    on_first = {}
    for vectors, bm25, weight in grid:
        on_first[(vectors, bm25, weight)] = measure(
            build(vectors, bm25, weight, 0.40, embedder), first)
    winner = max(on_first, key=lambda k: on_first[k]["hit_at"][3])
    baseline = ("full", "full", 0.6)

    print(f"gagnant sur A : vecteurs={winner[0]} BM25={winner[1]} "
          f"poids={winner[2]}")
    print(f"{'':28} {'moitié A':>12} {'moitié B':>12}  (B jamais utilisée "
          f"pour choisir)")
    rows = {}
    for label, config in (("référence (full, 0.6)", baseline),
                          ("gagnant sur A", winner)):
        a = on_first[config] if config in on_first else measure(
            build(*config, 0.40, embedder), first)
        b = measure(build(*config, 0.40, embedder), second)
        rows[label] = {"A": a, "B": b}
        print(f"{label:28} hit@3 {a['hit_at'][3]:>6.4f} {b['hit_at'][3]:>12.4f}")
        print(f"{'':28} MRR   {a['mrr']:>6.4f} {b['mrr']:>12.4f}")
    payload["overfitting"] = {
        "winner": list(winner), "baseline": list(baseline),
        "halves": {k: {h: v[h] for h in ("A", "B")} for k, v in rows.items()},
    }

    delta_a = (rows["gagnant sur A"]["A"]["hit_at"][3]
               - rows["référence (full, 0.6)"]["A"]["hit_at"][3])
    delta_b = (rows["gagnant sur A"]["B"]["hit_at"][3]
               - rows["référence (full, 0.6)"]["B"]["hit_at"][3])
    print(f"\n  gain hit@3 sur A (moitié de sélection) : {delta_a:+.4f}")
    print(f"  gain hit@3 sur B (moitié de contrôle)   : {delta_b:+.4f}")
    print("  -> " + ("le gain SURVIT au contrôle" if delta_b > 0
                     else "le gain NE SURVIT PAS : réglage surajusté"))

    # ── 3. Le plancher fait-il toujours son travail ? ────────────────────
    print("\n═══ 3. PLANCHER : recalibrage puis vérification du travail ═══\n")
    vectors, bm25, weight = winner
    rag = build(vectors, bm25, weight, 0.0, embedder)
    answerable = [e for e in entries if e["answerable"]]

    tops = {True: [], False: []}
    for entry in entries:
        indices = np.asarray(
            [i for i, d in enumerate(rag.documents) if d.code == entry["code"]],
            dtype=np.int64)
        query = _normalize_rows(embedder.embed([entry["question"]]))[0]
        tops[entry["answerable"]].append(
            float((rag.embeddings[indices] @ query).max()))
    print(f"score dense du rang 1 — répondables "
          f"[{min(tops[True]):.3f}–{max(tops[True]):.3f}], "
          f"sans réponse [{min(tops[False]):.3f}–{max(tops[False]):.3f}]")

    print(f"\n{'plancher':>9} | {'MRR':>7} {'hit@3':>7} {'FP':>6} {'vides':>6}")
    print("-" * 42)
    floors = {}
    for floor in (0.0, 0.35, 0.40, 0.45, 0.50):
        result = measure(build(vectors, bm25, weight, floor, embedder), entries)
        floors[floor] = result
        print(f"{floor:>9.2f} | {result['mrr']:>7.4f} "
              f"{result['hit_at'][3]:>7.4f} "
              f"{result['false_positive_rate']:>6.3f} {result['empty']:>6d}")
    best_floor = max(floors, key=lambda f: floors[f]["hit_at"][3])
    print(f"\nplancher retenu : {best_floor:.2f}")

    # Que supprime-t-il exactement ?
    lexical_only = kept = 0
    expected_lost = []
    for entry in answerable:
        indices = np.asarray(
            [i for i, d in enumerate(rag.documents) if d.code == entry["code"]],
            dtype=np.int64)
        query = _normalize_rows(embedder.embed([entry["question"]]))[0]
        dense = rag.embeddings[indices] @ query
        lex = rag._bm25(
            legal_rag._expanded_query_tokens(entry["question"]), indices)
        ck = 40
        only_lex = (set(np.argsort(-lex)[:ck].tolist())
                    - set(np.argsort(-dense)[:ck].tolist()))
        lexical_only += len(only_lex)
        kept += sum(1 for p in only_lex if dense[p] >= best_floor)
        numbers = {rag.documents[int(indices[p])].article_number
                   for p in only_lex if dense[p] < best_floor}
        if numbers & set(entry["expected_articles"]):
            expected_lost.append(
                (entry["id"], sorted(numbers & set(entry["expected_articles"]))))
    print(f"\ncandidats entrés par les MOTS seuls : {lexical_only}")
    print(f"  éliminés par le plancher {best_floor:.2f} : "
          f"{lexical_only - kept} ({1 - kept / lexical_only:.1%})")
    print(f"  dont articles ATTENDUS perdus : {len(expected_lost)}")
    for item in expected_lost:
        print(f"    {item[0]} -> {item[1]}")
    payload["floor"] = {
        "sweep": {str(f): floors[f] for f in floors},
        "chosen": best_floor,
        "lexical_only_candidates": lexical_only,
        "eliminated": lexical_only - kept,
        "expected_lost": expected_lost,
    }

    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\nsortie brute : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
