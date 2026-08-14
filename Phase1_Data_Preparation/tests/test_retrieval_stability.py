# -*- coding: utf-8 -*-
"""Le classement hybride doit être reproductible à scores égaux."""

import numpy as np

from lexior.agentic.config import RAGConfig
from lexior.agentic.legal_rag import LegalDocument, LegalRAG


class _Embedder:
    model = "test-embedding"

    def embed(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def cost_report(self):
        return {}


def _document(number: str) -> LegalDocument:
    return LegalDocument(
        id=f"ccq-{number}", code="CCQ", article_number=number,
        article_label=f"Article {number}", title="Code civil du Québec",
        text="Texte identique", taxonomy="test", domain="test",
        source_url=f"https://example.invalid/{number}",
    )


def test_scores_egaux_sont_departages_par_lordre_du_corpus():
    cfg = RAGConfig(
        top_k=3, candidate_k=3, dense_weight=1.0,
        llm_rerank_enabled=False, min_dense_score=-1.0,
        min_hybrid_score=-1.0, dense_floor_exempt_top_k=0,
    )
    documents = [_document("30"), _document("10"), _document("20")]
    rag = LegalRAG(
        cfg, _Embedder(), documents,
        np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        {"embedding_model": "test-embedding", "corpus_hash": "test"},
    )
    first = rag.search("question", "CCQ", top_k=3)
    second = rag.search("question", "CCQ", top_k=3)
    expected = ["30", "10", "20"]
    assert [item["article_number"] for item in first] == expected
    assert [item["article_number"] for item in second] == expected


def test_empreinte_de_requete_est_stable_et_sensible_aux_termes():
    cfg = RAGConfig(
        top_k=1, candidate_k=1, llm_rerank_enabled=False,
        min_dense_score=-1.0, min_hybrid_score=-1.0,
        dense_floor_exempt_top_k=0,
    )
    rag = LegalRAG(
        cfg, _Embedder(), [_document("1")],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        {"embedding_model": "test-embedding", "corpus_hash": "test"},
    )
    first = rag.call("semantic_search_ccq", {"query": "question"})
    same = rag.call("semantic_search_ccq", {"query": "question"})
    changed = rag.call("semantic_search_ccq", {
        "query": "question", "legal_terms": "qualification",
    })
    assert first["query_fingerprint"] == same["query_fingerprint"]
    assert first["query_fingerprint"] != changed["query_fingerprint"]
