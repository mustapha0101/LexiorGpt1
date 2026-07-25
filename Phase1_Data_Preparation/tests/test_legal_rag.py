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


class RejectingReranker:
    """Reranker qui écarte les candidats hors sujet."""

    def __init__(self, ranking, rejected):
        self.ranking = ranking
        self.rejected = rejected

    def complete_json(self, role, messages, temperature=0.0):
        return {"ranking": self.ranking, "rejected": self.rejected}


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


# ── Planchers de pertinence absolus ──────────────────────────────────────


class FixedQueryEmbedder:
    """Toute question tombe sur le même axe; la similarité vient des documents."""

    model = "fake-legal-embedding"

    def embed(self, texts):
        return np.asarray([[0.0, 1.0, 0.0]] * len(texts), dtype=np.float32)

    def cost_report(self):
        return {"model": self.model, "total": {
            "calls": 0, "failed_calls": 0, "tokens_in": 0,
            "tokens_cached_in": 0, "tokens_out": 0, "cost_usd": 0.0,
        }}


OFF_TOPIC_QUERY = "Mon locateur refuse que j'apporte mon chat."


def _off_topic_rag(tmp_path, **overrides):
    """Corpus dont aucun article ne répond vraiment à la question posée.

    Les deux articles ont une similarité cosinus faible (0.22 et 0.10) : le
    premier reste « le moins mauvais » et sort donc à 1.000 en min-max.
    """
    documents = [
        _document("CCQ", 1863, "Lorsque le locateur refuse, le tribunal peut trancher.", "louage"),
        _document("CCQ", 1726, "Le vendeur garantit l'acheteur contre les vices.", "vente"),
    ]
    embeddings = np.asarray([
        [0.9755, 0.22, 0.0],
        [0.9950, 0.10, 0.0],
    ], dtype=np.float32)
    settings = {"index_dir": str(tmp_path), "top_k": 5, "candidate_k": 5}
    settings.update(overrides)
    return LegalRAG(RAGConfig(**settings), FixedQueryEmbedder(), documents,
                    embeddings, {"embedding_model": FixedQueryEmbedder.model,
                                 "corpus_hash": "test"})


def test_minmax_score_is_never_shown_as_relevance(tmp_path):
    """Sans plancher, le meilleur candidat sort quand même à 1.000 en min-max.

    Le score exposé doit rester le score absolu, sinon la trajectoire
    apprend qu'un résultat hors sujet est parfaitement pertinent.
    """
    # Planchers explicitement désactivés : ce test isole l'effet du
    # min-max, indépendamment du calibrage de production.
    rag = _off_topic_rag(tmp_path, min_dense_score=0.0, min_hybrid_score=0.0)

    payload = rag.call("semantic_search_ccq", {
        "query": "Mon locateur refuse que j'apporte mon chat.", "top_k": 5})

    assert payload["results"], "sans plancher, la recherche retourne quand même"
    assert payload["results"][0]["score"] == 1.0
    assert payload["results"][0]["absolute_score"] < 0.5
    assert "1.000" not in payload["text"]


def test_absolute_floor_returns_nothing_when_no_article_matches(tmp_path):
    rag = _off_topic_rag(tmp_path, min_dense_score=0.30, min_hybrid_score=0.30)

    results = rag.search("Mon locateur refuse que j'apporte mon chat.", "CCQ")

    assert results == []


def test_empty_search_result_reads_as_no_article_found(tmp_path):
    """Le texte vu par le planner doit permettre au classifieur de voir `empty`."""
    rag = _off_topic_rag(tmp_path, min_dense_score=0.30, min_hybrid_score=0.30)

    payload = rag.call("semantic_search_ccq", {
        "query": "Mon locateur refuse que j'apporte mon chat.", "top_k": 5})

    assert payload["results"] == []
    assert payload["text"] == "Aucun article CCQ trouvé."


def test_absolute_floor_keeps_a_genuinely_relevant_article(tmp_path):
    documents = [
        _document("CCQ", 1002, "Tout propriétaire peut clore son terrain avec des clôtures.", "biens"),
        _document("CCQ", 345, "La clôture de l'exercice financier a lieu chaque année.", "personnes morales"),
    ]
    embeddings = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    cfg = RAGConfig(index_dir=str(tmp_path), top_k=5, candidate_k=5,
                    min_dense_score=0.30, min_hybrid_score=0.30)
    rag = LegalRAG(cfg, FakeEmbedder(), documents, embeddings,
                   {"embedding_model": FakeEmbedder.model, "corpus_hash": "test"})

    results = rag.search("Mon voisin a déplacé la clôture entre nos terrains.", "CCQ")

    assert [item["article_number"] for item in results] == ["1002"]
    assert results[0]["absolute_score"] >= 0.30


def test_thresholds_are_part_of_the_cache_signature(tmp_path):
    """Changer un plancher doit invalider les observations mises en cache."""
    permissive = _off_topic_rag(tmp_path).cache_signature
    strict = _off_topic_rag(tmp_path, min_dense_score=0.30).cache_signature

    assert permissive != strict


# ── Le reranker peut écarter, jamais inventer ────────────────────────────


def _witness_rag(tmp_path, reranker):
    documents = [
        _document("CPC", 279, "Les témoins sont interrogés à l'audience.", "preuve"),
        _document("CPC", 269, "Les témoins sont convoqués par citation à comparaître.", "preuve"),
    ]
    # Planchers désactivés : ces tests portent sur le reranker. Avec le
    # calibrage de production (0.40), une requête hors sujet est filtrée
    # AVANT le rerank et le test passerait sans jamais l'appeler.
    cfg = RAGConfig(index_dir=str(tmp_path), llm_rerank_enabled=True,
                    llm_rerank_k=2, min_dense_score=0.0,
                    min_hybrid_score=0.0)
    return LegalRAG(
        cfg, FakeEmbedder(), documents,
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.9]], dtype=np.float32),
        {"embedding_model": FakeEmbedder.model, "corpus_hash": "test"},
        reranker=reranker,
    )


def test_llm_reranker_can_drop_an_irrelevant_candidate(tmp_path):
    rag = _witness_rag(tmp_path, RejectingReranker(["269"], ["279"]))

    results = rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    assert [item["article_number"] for item in results] == ["269"]


def test_llm_reranker_may_reject_every_candidate(tmp_path):
    """« Aucun article pertinent » est une sortie valide, pas une erreur.

    La requête doit passer les planchers, sinon le rerank n'est jamais
    appelé et le test vérifierait le filtrage au lieu du rejet.
    """
    rag = _witness_rag(tmp_path, RejectingReranker([], ["269", "279"]))

    results = rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    assert results == []
    assert rag.last_rerank_rejection is not None, (
        "le rejet doit venir du reranker, pas du plancher")


def test_llm_reranker_cannot_invent_or_reject_unknown_articles(tmp_path):
    rag = _witness_rag(tmp_path, RejectingReranker(["9999", "269"], ["4242"]))

    results = rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    numbers = [item["article_number"] for item in results]
    assert numbers == ["269", "279"], "un rejet inconnu ne retire rien"
    assert "9999" not in numbers


def test_llm_reranker_ignores_a_malformed_answer(tmp_path):
    class BrokenReranker:
        def complete_json(self, role, messages, temperature=0.0):
            return {"ranking": "269", "rejected": "279"}

    rag = _witness_rag(tmp_path, BrokenReranker())

    results = rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    assert [item["article_number"] for item in results] == ["269", "279"]


# ── Le motif du rejet doit être consultable ──────────────────────────────


def test_a_total_rejection_records_its_reason(tmp_path):
    """Sans motif, un rejet total est indiscernable d'un échec de recherche.

    C'est exactement la confusion qu'a produite « ccq-mise-en-demeure » :
    liste vide, et rien pour dire si le reranker avait écarté de bons
    candidats ou si la recherche n'avait jamais remonté le bon article.
    """
    class Explaining:
        def complete_json(self, role, messages, temperature=0.0):
            return {"ranking": [], "rejected": ["269", "279"],
                    "reason": "aucun candidat ne traite du sujet demandé"}

    rag = _witness_rag(tmp_path, Explaining())

    results = rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    assert results == []
    trace = rag.last_rerank_rejection
    assert trace is not None
    assert trace["reason"] == "aucun candidat ne traite du sujet demandé"
    assert trace["rejected"] == ["269", "279"]
    assert trace["kept"] == []


def test_a_partial_rejection_records_what_survived(tmp_path):
    rag = _witness_rag(tmp_path, RejectingReranker(["269"], ["279"]))

    rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    trace = rag.last_rerank_rejection
    assert trace["rejected"] == ["279"] and trace["kept"] == ["269"]


def test_nothing_is_recorded_when_nothing_is_rejected(tmp_path):
    rag = _witness_rag(tmp_path, FakeReranker())

    rag.search("Comment assigner un témoin à comparaître?", "CPC", 2)

    assert rag.last_rerank_rejection is None
