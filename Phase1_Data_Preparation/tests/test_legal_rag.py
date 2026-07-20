import numpy as np

from agentic_generation.config import RAGConfig
from agentic_generation.legal_rag import LegalDocument, LegalRAG


class FakeEmbedder:
    model = "fake-legal-embedding"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            folded = text.casefold()
            if "clôture" in folded or "empiète" in folded:
                vectors.append([1.0, 0.0, 0.0])
            elif "témoin" in folded or "comparaître" in folded:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.0, 1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)

    def cost_report(self):
        return {"model": self.model, "total": {
            "calls": self.calls, "failed_calls": 0, "tokens_in": 0,
            "tokens_cached_in": 0, "tokens_out": 0, "cost_usd": 0.0,
        }}


class FakeReranker:
    def complete_json(self, role, messages, temperature=0.0):
        assert role == "retrieval_reranker"
        return {"ranking": ["269", "1002", "9999"]}


def _document(code, number, text, domain):
    return LegalDocument(
        id=f"{code.lower()}_{number}", code=code, article_number=str(number),
        article_label=f"Article {number}", title=f"{code} Article {number}",
        text=text, taxonomy=domain, domain=domain, source_url=f"https://example/{number}",
    )


def test_hybrid_reranking_finds_relevant_ccq_article_and_filters_code(tmp_path):
    documents = [
        _document("CCQ", 1002, "Tout propriétaire peut clore son terrain avec des clôtures.", "biens"),
        _document("CCQ", 345, "La clôture de l'exercice financier a lieu chaque année.", "personnes morales"),
        _document("CPC", 269, "Un témoin peut être assigné à comparaître.", "preuve"),
    ]
    embeddings = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    cfg = RAGConfig(index_dir=str(tmp_path), top_k=2, candidate_k=3,
                    dense_weight=0.8)
    rag = LegalRAG(cfg, FakeEmbedder(), documents, embeddings, {
        "embedding_model": FakeEmbedder.model, "corpus_hash": "test",
    })

    results = rag.search(
        "Mon voisin a déplacé la clôture et empiète sur mon terrain.", "CCQ")

    assert results[0]["article_number"] == "1002"
    assert all(result["code"] == "CCQ" for result in results)
    assert "dense_score" in results[0]
    assert "lexical_score" in results[0]


def test_cpc_search_does_not_return_ccq_articles(tmp_path):
    documents = [
        _document("CCQ", 1002, "Clôture entre voisins.", "biens"),
        _document("CPC", 269, "Un témoin peut être assigné à comparaître.", "preuve"),
    ]
    rag = LegalRAG(
        RAGConfig(index_dir=str(tmp_path)), FakeEmbedder(), documents,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        {"embedding_model": FakeEmbedder.model, "corpus_hash": "test"},
    )

    payload = rag.call("semantic_search_cpc", {
        "query": "Comment faire comparaître un témoin?", "top_k": 5,
    })

    assert payload["results"][0]["article_number"] == "269"
    assert all(item["code"] == "CPC" for item in payload["results"])
    assert payload["retrieval"] == "dense_candidates_then_bm25_dense_rerank"
    assert "assigné à comparaître" not in payload["text"]
    assert "Article 269" in payload["text"]


def test_llm_reranker_can_only_reorder_existing_candidates(tmp_path):
    documents = [
        _document("CPC", 279, "Les témoins sont interrogés à l'audience.", "preuve"),
        _document("CPC", 269, "Les témoins sont convoqués par citation à comparaître.", "preuve"),
    ]
    cfg = RAGConfig(index_dir=str(tmp_path), llm_rerank_enabled=True,
                    llm_rerank_k=2)
    rag = LegalRAG(
        cfg, FakeEmbedder(), documents,
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.9]], dtype=np.float32),
        {"embedding_model": FakeEmbedder.model, "corpus_hash": "test"},
        reranker=FakeReranker(),
    )

    results = rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    assert [item["article_number"] for item in results] == ["269", "279"]
    assert all(item["article_number"] != "9999" for item in results)
    assert all(item["reranker"] == "llm" for item in results)
