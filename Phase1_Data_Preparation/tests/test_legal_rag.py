import json

import numpy as np
import pytest

from agentic_generation.config import RAGConfig
from agentic_generation.legal_rag import LegalDocument, LegalRAG, RAGError
from lexior.agentic.legal_rag import _normalize_rows


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


# ── Choix du modèle d'embeddings par configuration ───────────────────────


def test_the_provider_selects_the_embedder(monkeypatch):
    """Le choix se fait par configuration, pas par édition de code."""
    from lexior.agentic import legal_rag

    construits = []

    class FauxBGE:
        def __init__(self, cfg, allow_remote_calls=True, device="cpu"):
            construits.append(("bge", cfg.embedding_model))
            self.model = cfg.embedding_model

    class FauxOpenAI:
        def __init__(self, cfg, allow_remote_calls):
            construits.append(("openai", cfg.embedding_model))
            self.model = cfg.embedding_model

    monkeypatch.setattr(legal_rag, "BGEEmbedder", FauxBGE)
    monkeypatch.setattr(legal_rag, "OpenAIEmbedder", FauxOpenAI)

    legal_rag.build_embedder(
        RAGConfig(embedding_provider="openai"), allow_remote_calls=True)
    legal_rag.build_embedder(
        RAGConfig(embedding_provider="bge", embedding_model="BAAI/bge-m3"),
        allow_remote_calls=True)

    assert [nom for nom, _ in construits] == ["openai", "bge"]


def test_an_unknown_provider_is_refused():
    from lexior.agentic.legal_rag import RAGError, build_embedder

    with pytest.raises(RAGError, match="inconnu"):
        build_embedder(RAGConfig(embedding_provider="mistral"),
                       allow_remote_calls=True)


def test_an_index_built_with_another_model_is_refused(tmp_path):
    """1536 et 1024 dimensions ne se mélangent pas : l'index doit être
    rejeté avant de produire des scores absurdes."""
    documents = [_document("CCQ", 1457, "Toute personne a le devoir…", "obligations")]
    (tmp_path / "documents.jsonl").write_text(
        json.dumps(documents[0].__dict__, ensure_ascii=False) + "\n",
        encoding="utf-8")
    np.save(tmp_path / "embeddings.npy",
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"embedding_model": "BAAI/bge-m3", "corpus_hash": "t"}),
        encoding="utf-8")

    with pytest.raises(RAGError, match="reconstruire l'index"):
        LegalRAG.load(RAGConfig(index_dir=str(tmp_path)), FakeEmbedder())


# ── Recentrage des vecteurs ──────────────────────────────────────────────


def _centering_corpus(tmp_path, centering):
    documents = [
        _document("CCQ", 1457, "Toute personne a le devoir de réparer.", "obligations"),
        _document("CCQ", 1726, "Le vendeur garantit contre les vices.", "vente"),
        _document("CPC", 269, "Les témoins sont convoqués par citation.", "preuve"),
    ]
    # Vecteurs volontairement tassés : une composante commune forte, un
    # signal distinctif faible — exactement la configuration que le
    # recentrage doit corriger.
    embeddings = np.asarray([
        [0.99, 0.14, 0.02],
        [0.99, 0.02, 0.14],
        [0.99, 0.08, 0.08],
    ], dtype=np.float32)
    cfg = RAGConfig(index_dir=str(tmp_path), top_k=3, candidate_k=3,
                    centering=centering)
    return LegalRAG(cfg, FakeEmbedder(), documents, embeddings,
                    {"embedding_model": FakeEmbedder.model,
                     "corpus_hash": "test"})


def test_recentred_vectors_stay_unit_norm(tmp_path):
    """Le piège classique : sans renormalisation, le produit scalaire
    cesse d'être un cosinus."""
    rag = _centering_corpus(tmp_path, "global")
    indices = np.asarray([0, 1, 2], dtype=np.int64)

    centered = rag._centered_matrix("CCQ", indices)
    norms = np.linalg.norm(centered, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5), norms


def test_a_recentred_query_stays_unit_norm(tmp_path):
    rag = _centering_corpus(tmp_path, "global")
    mean = rag._mean_for("CCQ")

    recentred = rag._recenter(np.asarray([[0.99, 0.10, 0.05]], np.float32), mean)

    assert np.isclose(np.linalg.norm(recentred[0]), 1.0, atol=1e-5)


def test_no_centering_leaves_the_vectors_untouched(tmp_path):
    rag = _centering_corpus(tmp_path, "none")
    indices = np.asarray([0, 1], dtype=np.int64)

    assert rag._means == {}
    assert np.array_equal(rag._centered_matrix("CCQ", indices),
                          rag.embeddings[indices])


def test_global_centering_uses_one_mean_for_every_code(tmp_path):
    rag = _centering_corpus(tmp_path, "global")

    assert list(rag._means) == [""]
    assert rag._mean_for("CCQ") is rag._mean_for("CPC")


def test_per_code_centering_uses_one_mean_per_code(tmp_path):
    rag = _centering_corpus(tmp_path, "per_code")

    assert sorted(rag._means) == ["CCQ", "CPC"]
    assert not np.array_equal(rag._mean_for("CCQ"), rag._mean_for("CPC"))


def test_the_means_come_from_the_indexed_vectors(tmp_path):
    """Dérivées de l'index chargé : elles ne peuvent pas s'en désynchroniser."""
    rag = _centering_corpus(tmp_path, "per_code")

    expected = rag.embeddings[np.asarray([0, 1])].mean(axis=0)

    assert np.allclose(rag._mean_for("CCQ"), expected)


def test_centering_spreads_compressed_scores(tmp_path):
    """L'hypothèse même : retirer la composante commune étale les scores."""
    query = np.asarray([[0.99, 0.13, 0.03]], dtype=np.float32)
    plain = _centering_corpus(tmp_path, "none")
    centred = _centering_corpus(tmp_path, "global")
    indices = np.asarray([0, 1], dtype=np.int64)

    before = plain.embeddings[indices] @ _normalize_rows(query)[0]
    mean = centred._mean_for("CCQ")
    after = (centred._centered_matrix("CCQ", indices)
             @ centred._recenter(query, mean)[0])

    assert (after.max() - after.min()) > (before.max() - before.min())


def test_an_unknown_centering_mode_is_refused(tmp_path):
    with pytest.raises(RAGError, match="recentrage inconnu"):
        _centering_corpus(tmp_path, "zscore")


# ── Texte soumis à l'embedder ────────────────────────────────────────────


def test_full_mode_prefixes_the_labels():
    from lexior.agentic.legal_rag import search_text_for

    document = _document("CCQ", 1457, "Toute personne a le devoir.", "obligations")

    indexed = search_text_for(document, "full")

    assert indexed.startswith("CCQ Article 1457")
    assert "obligations" in indexed
    assert indexed.endswith("Toute personne a le devoir.")


def test_text_only_mode_keeps_the_normative_content_alone():
    from lexior.agentic.legal_rag import search_text_for

    document = _document("CCQ", 1457, "Toute personne a le devoir.", "obligations")

    assert search_text_for(document, "text_only") == "Toute personne a le devoir."


def test_the_historical_property_stays_on_full():
    document = _document("CCQ", 1457, "Toute personne a le devoir.", "obligations")

    assert document.search_text == "\n".join((
        "CCQ Article 1457", "Article 1457", "obligations", "obligations",
        "Toute personne a le devoir."))


def test_an_unknown_text_mode_is_refused():
    from lexior.agentic.legal_rag import search_text_for

    with pytest.raises(RAGError, match="texte indexé inconnu"):
        search_text_for(_document("CCQ", 1, "x", "y"), "titles_only")


def test_bm25_follows_the_configured_text(tmp_path):
    """Le mot d'une étiquette ne doit plus matcher en mode texte seul."""
    documents = [_document("CCQ", 1457, "Toute personne a le devoir.",
                           "obligations")]
    embeddings = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    manifest = {"embedding_model": FakeEmbedder.model, "corpus_hash": "test"}

    plein = LegalRAG(RAGConfig(index_dir=str(tmp_path),
                               search_text_fields="full"),
                     FakeEmbedder(), documents, embeddings, manifest)
    seul = LegalRAG(RAGConfig(index_dir=str(tmp_path),
                              search_text_fields="text_only"),
                    FakeEmbedder(), documents, embeddings, manifest)

    assert plein._token_counts[0]["obligations"] == 2
    assert "obligations" not in seul._token_counts[0]
    assert seul._token_counts[0]["personne"] == 1


# ── Le résumé sert à trouver, jamais à prouver ───────────────────────────


def test_the_excerpt_always_comes_from_the_statute(tmp_path):
    """Règle absolue : ce que voient le reranker et l'usager est le TEXTE
    DE LOI, jamais une reformulation. C'est ce qui garde le reranker
    indépendant de la façon dont l'article a été retrouvé."""
    import inspect
    from lexior.agentic import legal_rag

    source = inspect.getsource(legal_rag.LegalRAG.search)

    assert '"excerpt": document.text[:700]' in source, (
        "l'extrait doit être découpé dans document.text")
    assert "summary" not in source.lower() and "resume" not in source.lower()


def test_the_tool_response_never_carries_article_text(tmp_path):
    """`call()` n'expose que des libellés et des scores.

    Un résumé embarqué pour la recherche ne peut donc pas atteindre la
    trajectoire, ni servir de preuve — la règle du lot 5 tient par
    construction, pas par vigilance.
    """
    documents = [_document("CCQ", 1457, "Toute personne a le devoir de "
                           "respecter les règles de conduite.", "obligations")]
    rag = LegalRAG(
        RAGConfig(index_dir=str(tmp_path)), FakeEmbedder(), documents,
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        {"embedding_model": FakeEmbedder.model, "corpus_hash": "test"})

    payload = rag.call("semantic_search_ccq", {"query": "un dommage"})

    assert "devoir de respecter" not in payload["text"]
    assert "Article 1457" in payload["text"]
