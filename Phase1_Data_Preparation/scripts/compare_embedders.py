#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compare deux modèles d'embeddings sur le jeu de test de recherche.

Quatre colonnes, produites dans le MÊME run sur le MÊME fichier
d'annotations : OpenAI et BGE-M3, chacun sans puis avec le rerank LLM.
Mesurer les colonnes à des dates différentes reviendrait à comparer des
jeux de test différents — le fichier d'annotations évolue.

Une seule variable change entre les deux modèles : le corpus indexé est
identique, les seuils sont identiques, le jeu de test est identique.

La sortie brute par question est enregistrée intégralement : position de
chaque article attendu, articles retenus, articles rejetés par le rerank
et motif. Un chiffre surprenant doit pouvoir s'expliquer sans relancer
l'indexation.

    python scripts/compare_embedders.py --output comparaison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1 / "tests"), str(PHASE1)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import (  # noqa: E402
    BGEEmbedder, LegalRAG, index_exists,
)
from lexior.agentic.storage import JsonCache  # noqa: E402
from lexior.agentic.teacher_client import TeacherClient  # noqa: E402
from test_retrieval_gold import (  # noqa: E402
    CachedEmbedder, K_VALUES, QUERIES_PATH, load_gold,
)

OPENAI_INDEX = PHASE1 / "data" / "agentic" / "rag_index"
BGE_INDEX = PHASE1 / "data" / "agentic" / "rag_index_bge"


class TracingReranker:
    """Enveloppe le teacher pour conserver la sortie brute de chaque appel."""

    def __init__(self, client):
        self.client = client
        self.exchanges: list[dict] = []

    def complete_json(self, role, messages, temperature=0.0):
        answer = self.client.complete_json(role, messages, temperature)
        payload = json.loads(messages[1]["content"])
        self.exchanges.append({
            "question": payload.get("question", ""),
            "candidats": [c["article_number"] for c in payload["candidats"]],
            "brut": answer,
        })
        return answer


def _rag(index_dir: Path, embedder, reranker=None) -> LegalRAG:
    production = load_config(
        str(PHASE1 / "configs" / "agentic_generation.yaml")).rag
    cfg = RAGConfig(
        index_dir=str(index_dir),
        embedding_model=embedder.model,
        top_k=max(K_VALUES),
        llm_rerank_enabled=reranker is not None,
        llm_rerank_k=production.llm_rerank_k,
        min_dense_score=production.min_dense_score,
        min_hybrid_score=production.min_hybrid_score,
    )
    return LegalRAG.load(cfg, embedder, reranker=reranker)


def evaluate(rag: LegalRAG, entries: list[dict], label: str) -> dict:
    """Métriques agrégées ET trace par question."""
    largest = max(K_VALUES)
    hits = {k: [] for k in K_VALUES}
    precision = {k: [] for k in K_VALUES}
    recall = {k: [] for k in K_VALUES}
    reciprocal: list[float] = []
    false_positives: list[str] = []
    empty_answerable: list[str] = []
    per_question: list[dict] = []
    latencies: list[float] = []

    for entry in entries:
        started = time.monotonic()
        results = rag.search(entry["question"], entry["code"], largest)
        latencies.append(time.monotonic() - started)
        retrieved = [item["article_number"] for item in results]
        rejection = getattr(rag, "last_rerank_rejection", None)

        trace = {
            "id": entry["id"],
            "answerable": entry["answerable"],
            "expected": entry["expected_articles"],
            "retrieved": retrieved,
            "scores": [round(item["absolute_score"], 3) for item in results],
            "positions": {
                article: (retrieved.index(article) + 1
                          if article in retrieved else None)
                for article in entry["expected_articles"]
            },
            "rerank_rejected": (rejection or {}).get("rejected", []),
            "rerank_reason": (rejection or {}).get("reason", ""),
        }
        per_question.append(trace)
        rag.last_rerank_rejection = None

        if not entry["answerable"]:
            if retrieved:
                false_positives.append(entry["id"])
            continue

        expected = set(entry["expected_articles"])
        if not retrieved:
            empty_answerable.append(entry["id"])
        for k in K_VALUES:
            found = expected.intersection(retrieved[:k])
            hits[k].append(1.0 if found else 0.0)
            precision[k].append(len(found) / k)
            recall[k].append(len(found) / len(expected))
        rank = next((position for position, number
                     in enumerate(retrieved, start=1)
                     if number in expected), 0)
        reciprocal.append(1.0 / rank if rank else 0.0)

    unanswerable = [e for e in entries if not e["answerable"]]
    mean = lambda v: round(sum(v) / len(v), 4) if v else 0.0
    return {
        "label": label,
        "hit_at": {k: mean(hits[k]) for k in K_VALUES},
        "precision_at": {k: mean(precision[k]) for k in K_VALUES},
        "recall_at": {k: mean(recall[k]) for k in K_VALUES},
        "mrr": mean(reciprocal),
        "false_positive_rate": round(
            len(false_positives) / len(unanswerable), 4),
        "false_positive_ids": false_positives,
        "empty_on_answerable_ids": empty_answerable,
        "ms_per_query": round(sum(latencies) * 1000 / len(latencies), 1),
        "per_question": per_question,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="comparaison_embedders.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    entries = load_gold()
    print(f"jeu de test : {len(entries)} questions "
          f"({sum(1 for e in entries if e['answerable'])} répondables)")

    for index in (OPENAI_INDEX, BGE_INDEX):
        if not index_exists(index):
            print(f"index absent : {index}", file=sys.stderr)
            return 1

    # Corpus identique des deux côtés : sinon la comparaison a deux
    # variables.
    hashes = {
        name: json.loads((path / "manifest.json").read_text(
            encoding="utf-8")).get("corpus_hash")
        for name, path in (("openai", OPENAI_INDEX), ("bge", BGE_INDEX))
    }
    if hashes["openai"] != hashes["bge"]:
        print(f"CORPUS DIFFÉRENTS — comparaison invalide : {hashes}",
              file=sys.stderr)
        return 2
    print(f"corpus identique des deux côtés : {hashes['openai'][:16]}…\n")

    cfg = load_config(str(PHASE1 / "configs" / "agentic_generation.yaml"))
    cache = JsonCache(PHASE1 / "data" / "agentic" / "cache" / "rerank-compare")
    teacher = TeacherClient(cfg.teacher, allow_remote_calls=True, cache=cache,
                            cache_extra_key="compare-embedders")

    cached = np.load(QUERIES_PATH, allow_pickle=False)
    by_id = {e["id"]: e["question"] for e in entries}
    vectors = {by_id[str(i)]: cached["vectors"][p]
               for p, i in enumerate(cached["ids"]) if str(i) in by_id}
    openai_embedder = CachedEmbedder(str(cached["model"]), vectors)

    print("chargement de BAAI/bge-m3…", flush=True)
    bge_embedder = BGEEmbedder(RAGConfig(embedding_model="BAAI/bge-m3"))

    columns = {}
    traces = {}
    for name, embedder, index in (("openai", openai_embedder, OPENAI_INDEX),
                                  ("bge", bge_embedder, BGE_INDEX)):
        for rerank in (False, True):
            label = f"{name}{'_rerank' if rerank else ''}"
            print(f"mesure : {label}…", flush=True)
            tracer = TracingReranker(teacher) if rerank else None
            result = evaluate(_rag(index, embedder, tracer), entries, label)
            traces[label] = result.pop("per_question")
            if tracer:
                result["rerank_exchanges"] = len(tracer.exchanges)
            columns[label] = result

    order = ["openai", "openai_rerank", "bge", "bge_rerank"]
    titles = ["OpenAI sans", "OpenAI avec", "BGE sans", "BGE avec"]
    rows = [
        ("faux positifs", lambda c: c["false_positive_rate"]),
        ("MRR", lambda c: c["mrr"]),
        ("hit@3", lambda c: c["hit_at"][3]),
        ("hit@8", lambda c: c["hit_at"][8]),
        ("recall@3", lambda c: c["recall_at"][3]),
        ("recall@8", lambda c: c["recall_at"][8]),
        ("ms/requête", lambda c: c["ms_per_query"]),
    ]
    print("\n" + " " * 16 + "".join(f"{t:>14}" for t in titles))
    print("-" * (16 + 14 * 4))
    for name, getter in rows:
        print(f"{name:16}" + "".join(
            f"{getter(columns[c]):>14.3f}" for c in order))

    payload = {"columns": columns, "per_question": traces,
               "corpus_hash": hashes["openai"],
               "gold_entries": len(entries),
               "bge_usage": bge_embedder.cost_report(),
               "teacher_usage": teacher.cost_report()}
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsortie brute par question : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
