# -*- coding: utf-8 -*-
"""Mesure de la qualité du retrieval sur un jeu de test annoté.

Trois niveaux, du moins exigeant au plus exigeant :

1. le fixture est bien formé            — toujours exécuté ;
2. chaque article attendu existe        — exige l'index RAG local ;
3. precision@k / recall@k / faux positifs — exige en plus le cache de
   vecteurs de questions produit par ``scripts/embed_retrieval_gold.py``.

La métrique qui compte est le **taux de faux positifs** sur les questions
sans réponse dans le corpus : un système qui retourne toujours quelque
chose ne peut pas déclencher de reformulation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Exécutable directement (calibrage), pas seulement sous pytest.
for _candidate in (str(Path(__file__).resolve().parents[1] / "src"),):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from lexior.agentic.config import RAGConfig  # noqa: E402
from lexior.agentic.legal_rag import LegalRAG, index_exists  # noqa: E402

PHASE1 = Path(__file__).resolve().parents[1]
GOLD_PATH = PHASE1 / "tests" / "fixtures" / "retrieval_gold.jsonl"
QUERIES_PATH = PHASE1 / "tests" / "fixtures" / "retrieval_gold_queries.npz"
BASELINE_PATH = PHASE1 / "tests" / "fixtures" / "retrieval_gold_baseline.json"
INDEX_DIR = PHASE1 / "data" / "agentic" / "rag_index"

K_VALUES = (3, 5, 8)


def load_gold() -> list[dict]:
    raw = GOLD_PATH.read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


class CachedEmbedder:
    """Rejoue les vecteurs de questions mis en cache. Aucun appel réseau."""

    def __init__(self, model: str, vectors: dict[str, np.ndarray]):
        self.model = model
        self._vectors = vectors

    def embed(self, texts):
        missing = [text for text in texts if text not in self._vectors]
        if missing:
            raise KeyError(
                f"question absente du cache d'embeddings : {missing[0][:60]}… "
                "relancer scripts/embed_retrieval_gold.py --force")
        return np.asarray([self._vectors[text] for text in texts],
                          dtype=np.float32)

    def cost_report(self):
        return {"model": self.model, "total": {
            "calls": 0, "failed_calls": 0, "tokens_in": 0,
            "tokens_cached_in": 0, "tokens_out": 0, "cost_usd": 0.0,
        }}


class GoldSetUnavailable(RuntimeError):
    """Index ou cache de vecteurs manquant : mesure impossible."""


def _require_index():
    if not index_exists(INDEX_DIR):
        pytest.skip(f"index RAG absent de {INDEX_DIR}")


def _build_rag(**overrides) -> LegalRAG:
    """Index réel + embedder rejouant le cache. Rerank LLM désactivé."""
    if not index_exists(INDEX_DIR):
        raise GoldSetUnavailable(f"index RAG absent de {INDEX_DIR}")
    if not QUERIES_PATH.exists():
        raise GoldSetUnavailable(
            "cache de vecteurs absent : lancer "
            "`python scripts/embed_retrieval_gold.py --allow-remote-calls`")
    cached = np.load(QUERIES_PATH, allow_pickle=False)
    digest = hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
    if str(cached["gold_sha256"]) != digest:
        raise GoldSetUnavailable(
            "cache de vecteurs périmé (le fixture a changé) : relancer "
            "`python scripts/embed_retrieval_gold.py --allow-remote-calls "
            "--force`")
    entries = load_gold()
    by_id = {entry["id"]: entry["question"] for entry in entries}
    vectors = {
        by_id[str(identifier)]: cached["vectors"][position]
        for position, identifier in enumerate(cached["ids"])
        if str(identifier) in by_id
    }
    settings = {
        "index_dir": str(INDEX_DIR),
        "llm_rerank_enabled": False,
        "top_k": max(K_VALUES),
    }
    settings.update(overrides)
    cfg = RAGConfig(**settings)
    return LegalRAG.load(cfg, CachedEmbedder(str(cached["model"]), vectors))


def _load_rag(**overrides) -> LegalRAG:
    try:
        return _build_rag(**overrides)
    except GoldSetUnavailable as error:
        pytest.skip(str(error))


# ── 1. Forme du fixture ──────────────────────────────────────────────────


def test_gold_set_is_well_formed():
    entries = load_gold()
    assert len(entries) >= 50, "le jeu de test doit compter au moins 50 questions"

    identifiers = [entry["id"] for entry in entries]
    assert len(set(identifiers)) == len(identifiers), "identifiants dupliqués"

    answerable = [entry for entry in entries if entry["answerable"]]
    unanswerable = [entry for entry in entries if not entry["answerable"]]
    assert len(answerable) >= 40
    assert len(unanswerable) >= 10, (
        "sans questions hors corpus, le taux de faux positifs n'est pas mesurable")

    for entry in entries:
        assert entry["code"] in ("CCQ", "CPC"), entry["id"]
        assert entry["question"].strip(), entry["id"]
        assert entry["note"].strip(), entry["id"]
        if entry["answerable"]:
            assert entry["expected_articles"], entry["id"]
        else:
            assert entry["expected_articles"] == [], entry["id"]


def test_gold_articles_exist_in_the_indexed_corpus():
    """Un libellé attendu qui n'existe pas rendrait le recall ininterprétable."""
    _require_index()
    documents = [
        json.loads(line)
        for line in (INDEX_DIR / "documents.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    indexed = {(doc["code"], doc["article_number"]) for doc in documents}
    missing = [
        (entry["id"], entry["code"], number)
        for entry in load_gold()
        for number in entry["expected_articles"]
        if (entry["code"], number) not in indexed
    ]
    assert not missing, f"articles attendus absents du corpus : {missing}"


# ── 2. Métriques ─────────────────────────────────────────────────────────


def measure(rag: LegalRAG, entries: list[dict]) -> dict:
    """precision@k, recall@k, hit@k et taux de faux positifs."""
    largest = max(K_VALUES)
    hits = {k: [] for k in K_VALUES}
    precision = {k: [] for k in K_VALUES}
    recall = {k: [] for k in K_VALUES}
    reciprocal_ranks: list[float] = []
    false_positives: list[str] = []
    empty_answers: list[str] = []

    for entry in entries:
        results = rag.search(entry["question"], entry["code"], largest)
        retrieved = [item["article_number"] for item in results]

        if not entry["answerable"]:
            if retrieved:
                false_positives.append(entry["id"])
            continue

        expected = set(entry["expected_articles"])
        if not retrieved:
            empty_answers.append(entry["id"])
        for k in K_VALUES:
            top = retrieved[:k]
            found = expected.intersection(top)
            hits[k].append(1.0 if found else 0.0)
            precision[k].append(len(found) / k)
            recall[k].append(len(found) / len(expected))
        rank = next(
            (position for position, number in enumerate(retrieved, start=1)
             if number in expected), 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    unanswerable = [entry for entry in entries if not entry["answerable"]]
    mean = lambda values: round(sum(values) / len(values), 4) if values else 0.0
    return {
        "answerable": len(reciprocal_ranks),
        "unanswerable": len(unanswerable),
        "hit_at": {k: mean(hits[k]) for k in K_VALUES},
        "precision_at": {k: mean(precision[k]) for k in K_VALUES},
        "recall_at": {k: mean(recall[k]) for k in K_VALUES},
        "mrr": mean(reciprocal_ranks),
        "false_positive_rate": round(
            len(false_positives) / len(unanswerable), 4) if unanswerable else 0.0,
        "false_positive_ids": false_positives,
        "empty_on_answerable_ids": empty_answers,
    }


def test_retrieval_metrics_do_not_regress():
    rag = _load_rag()
    if not BASELINE_PATH.exists():
        pytest.skip(
            "référence absente : lancer `python tests/test_retrieval_gold.py` "
            f"puis écrire {BASELINE_PATH.name}")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    report = measure(rag, load_gold())
    tolerance = float(baseline.get("tolerance", 0.02))

    for k in K_VALUES:
        for metric in ("hit_at", "recall_at"):
            floor = float(baseline[metric][str(k)]) - tolerance
            actual = report[metric][k]
            assert actual >= floor, (
                f"{metric}@{k} a régressé : {actual} < {floor}")

    ceiling = float(baseline["false_positive_rate"]) + tolerance
    assert report["false_positive_rate"] <= ceiling, (
        "le taux de faux positifs sur les questions hors corpus a augmenté : "
        f"{report['false_positive_rate']} > {ceiling} "
        f"({report['false_positive_ids']})")

    assert not report["empty_on_answerable_ids"], (
        "des questions répondables ne retournent plus rien : "
        f"{report['empty_on_answerable_ids']}")


if __name__ == "__main__":  # pragma: no cover - outil de calibrage
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        measured = measure(_build_rag(), load_gold())
    except GoldSetUnavailable as error:
        print(f"[gold] mesure impossible : {error}")
        raise SystemExit(1)
    print(json.dumps(measured, ensure_ascii=False, indent=2))
