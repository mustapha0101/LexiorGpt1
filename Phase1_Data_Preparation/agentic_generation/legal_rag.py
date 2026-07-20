# -*- coding: utf-8 -*-
"""Index sémantique local CCQ/CPC avec reranking hybride.

Les embeddings servent uniquement à retrouver des articles candidats. Le
Planner doit ensuite appeler l'outil MCP officiel ``get_*_articles`` avant de
rédiger sa réponse : le corpus indexé n'est donc pas traité comme la preuve
finale.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from .config import RAGConfig

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
ARTICLE_NUMBER_RE = re.compile(r"(\d{1,4}(?:\.\d+)?)")
CODE_NAMES = {
    "CCQ": "Code civil du Québec",
    "CPC": "Code de procédure civile du Québec",
}
RETRIEVAL_VERSION = "legal-rag-1.2-noncitable"


class RAGError(RuntimeError):
    """Index absent, incompatible ou appel d'embeddings impossible."""


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def cost_report(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LegalDocument:
    id: str
    code: str
    article_number: str
    article_label: str
    title: str
    text: str
    taxonomy: str
    domain: str
    source_url: str

    @property
    def search_text(self) -> str:
        return "\n".join(part for part in (
            self.title,
            self.article_label,
            self.domain,
            self.taxonomy.replace("_", " ").replace("/", " "),
            self.text,
        ) if part)


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in value if not unicodedata.combining(ch)).casefold()


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(_fold(value))


def _expanded_query_tokens(value: str) -> list[str]:
    """Ajoute des équivalents procéduraux sans remplacer la requête originale.

    Les embeddings portent le sens général; ces expansions très limitées
    aident BM25 lorsque l'utilisateur emploie un verbe courant différent du
    terme exact du Code (par exemple « assigner » plutôt que « citer »).
    """
    tokens = _tokens(value)
    token_set = set(tokens)
    expansions: dict[str, tuple[str, ...]] = {
        "assigner": ("citer", "citation", "convoquer", "convocation"),
        "assigne": ("citer", "citation", "convoquer", "convocation"),
        "assignation": ("citer", "citation", "convoquer", "convocation"),
        "temoin": ("temoins", "temoignage"),
        "temoins": ("temoin", "temoignage"),
        "comparaitre": ("comparution", "citation"),
        "empiete": ("empietement", "bornage", "limites"),
        "empietement": ("empiete", "bornage", "limites"),
        "voisin": ("voisinage", "fonds", "proprietaire"),
        "cloture": ("clore",),
    }
    for token in list(token_set):
        tokens.extend(expansions.get(token, ()))
    return list(dict.fromkeys(tokens))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _article_number(label: str) -> str:
    match = ARTICLE_NUMBER_RE.search(label or "")
    return match.group(1) if match else ""


def _source_url(code: str, article_number: str) -> str:
    slug = "ccq-1991" if code == "CCQ" else "cpc"
    return (
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/"
        f"{slug}#se:{article_number}"
    )


def load_hf_corpus(dataset_name: str, split: str = "train",
                   limit: int = -1) -> list[LegalDocument]:
    """Charge et nettoie le corpus article-par-article depuis Hugging Face."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or None
    rows = load_dataset(dataset_name, split=split, token=token)
    documents: list[LegalDocument] = []
    for row in rows:
        code_name = str(row.get("code") or "")
        if code_name == CODE_NAMES["CCQ"]:
            code = "CCQ"
        elif code_name == CODE_NAMES["CPC"]:
            code = "CPC"
        else:
            continue
        if str(row.get("jurisdiction") or "") != "Québec (Provincial)":
            continue
        text = str(row.get("texte") or "").strip()
        folded = _fold(text).strip(" .()")
        if len(text) < 30 or folded.startswith(("abroge", "omis", "modification integree")):
            continue
        label = str(row.get("article") or "").strip()
        number = _article_number(label)
        if not number:
            continue
        taxonomy = str(row.get("chemin_taxonomy") or "")
        domain = taxonomy.split("/")[-1].replace("_", " ").strip()
        documents.append(LegalDocument(
            id=str(row.get("id") or f"{code.lower()}_{number}"),
            code=code,
            article_number=number,
            article_label=label or f"Article {number}",
            title=str(row.get("title") or f"{code} Article {number}"),
            text=text,
            taxonomy=taxonomy,
            domain=domain,
            source_url=_source_url(code, number),
        ))
        if limit >= 0 and len(documents) >= limit:
            break
    return documents


class OpenAIEmbedder:
    """Client embeddings avec comptage séparé des appels, jetons et coûts."""

    def __init__(self, cfg: RAGConfig, allow_remote_calls: bool):
        if not allow_remote_calls:
            raise RAGError("embeddings distants refusés sans --allow-remote-calls")
        if not cfg.embedding_api_key:
            raise RAGError(
                "clé embeddings absente (RAG_EMBEDDING_API_KEY ou OPENAI_API_KEY)"
            )
        from openai import OpenAI

        self.model = cfg.embedding_model
        self.price_per_1m = cfg.embedding_price_per_1m_usd
        self.client = OpenAI(
            base_url=cfg.embedding_base_url,
            api_key=cfg.embedding_api_key,
            timeout=120.0,
            max_retries=3,
        )
        self.calls = 0
        self.failed_calls = 0
        self.tokens_in = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self.calls += 1
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=list(texts),
                encoding_format="float",
            )
        except Exception:
            self.failed_calls += 1
            raise
        usage = getattr(response, "usage", None)
        self.tokens_in += int(
            getattr(usage, "prompt_tokens", 0)
            or getattr(usage, "total_tokens", 0)
            or 0
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype=np.float32)

    def cost_report(self) -> dict[str, Any]:
        cost = self.tokens_in * self.price_per_1m / 1_000_000
        total = {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "tokens_in": self.tokens_in,
            "tokens_cached_in": 0,
            "tokens_out": 0,
            "cost_usd": round(cost, 6),
        }
        return {"model": self.model, "total": total}


def index_exists(index_dir: str | Path) -> bool:
    root = Path(index_dir)
    return all((root / name).exists() for name in (
        "documents.jsonl", "embeddings.npy", "manifest.json"
    ))


def build_index(cfg: RAGConfig, embedder: Embedder, *, force: bool = False,
                limit: int = -1, progress=print) -> dict[str, Any]:
    """Construit l'index persistant. Une reconstruction exige ``force=True``."""
    root = Path(cfg.index_dir)
    if index_exists(root) and not force:
        raise RAGError(
            f"index déjà présent dans {root}; ajouter --force pour le reconstruire"
        )
    progress(f"[rag] chargement du corpus {cfg.dataset_name} ({cfg.dataset_split})...")
    documents = load_hf_corpus(cfg.dataset_name, cfg.dataset_split, limit=limit)
    if not documents:
        raise RAGError("aucun article CCQ/CPC exploitable dans le corpus")
    progress(f"[rag] {len(documents)} articles à indexer avec {embedder.model}.")

    batches: list[np.ndarray] = []
    size = max(int(cfg.embedding_batch_size), 1)
    texts = [document.search_text for document in documents]
    for start in range(0, len(texts), size):
        stop = min(start + size, len(texts))
        batches.append(embedder.embed(texts[start:stop]))
        total = embedder.cost_report().get("total", {})
        progress(
            f"[rag] embeddings {stop}/{len(texts)} | appels {total.get('calls', 0)} | "
            f"jetons {total.get('tokens_in', 0)} | coût ${float(total.get('cost_usd', 0)):.6f} USD"
        )
    embeddings = _normalize_rows(np.vstack(batches))
    if len(embeddings) != len(documents):
        raise RAGError("le nombre d'embeddings ne correspond pas au corpus")

    root.mkdir(parents=True, exist_ok=True)
    documents_path = root / "documents.jsonl"
    embeddings_path = root / "embeddings.npy"
    manifest_path = root / "manifest.json"
    with documents_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")
    with embeddings_path.open("wb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
    corpus_hash = hashlib.sha256(
        "\n".join(f"{doc.id}:{doc.text}" for doc in documents).encode("utf-8")
    ).hexdigest()
    manifest = {
        "version": 1,
        "dataset_name": cfg.dataset_name,
        "dataset_split": cfg.dataset_split,
        "embedding_model": embedder.model,
        "documents": len(documents),
        "dimensions": int(embeddings.shape[1]),
        "corpus_hash": corpus_hash,
        "usage": embedder.cost_report(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


class LegalRAG:
    """Recherche dense, sélection de candidats et reranking BM25+dense."""

    def __init__(self, cfg: RAGConfig, embedder: Embedder,
                 documents: list[LegalDocument], embeddings: np.ndarray,
                 manifest: dict[str, Any], reranker: Any = None):
        self.cfg = cfg
        self.embedder = embedder
        self.documents = documents
        self.embeddings = _normalize_rows(embeddings)
        self.manifest = manifest
        self.reranker = reranker
        self._token_counts = [Counter(_tokens(doc.search_text)) for doc in documents]
        self._doc_lengths = np.asarray(
            [sum(counts.values()) for counts in self._token_counts], dtype=np.float32
        )
        self._avg_length = float(self._doc_lengths.mean()) if len(documents) else 1.0
        self._document_frequency = Counter()
        for counts in self._token_counts:
            self._document_frequency.update(counts.keys())

    @classmethod
    def load(cls, cfg: RAGConfig, embedder: Embedder,
             reranker: Any = None) -> "LegalRAG":
        root = Path(cfg.index_dir)
        if not index_exists(root):
            raise RAGError(
                f"index RAG absent dans {root}; lancer d'abord `python -m "
                "agentic_generation.cli build-rag-index --config .\\configs\\"
                "agentic_generation.yaml --allow-remote-calls`"
            )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        indexed_model = str(manifest.get("embedding_model") or "")
        if indexed_model != embedder.model:
            raise RAGError(
                f"index créé avec {indexed_model}, mais requêtes configurées avec "
                f"{embedder.model}; reconstruire l'index ou aligner RAG_EMBEDDING_MODEL"
            )
        documents = [
            LegalDocument(**json.loads(line))
            for line in (root / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        embeddings = np.load(root / "embeddings.npy", allow_pickle=False)
        if embeddings.ndim != 2 or len(documents) != len(embeddings):
            raise RAGError("index RAG incohérent: documents et embeddings divergent")
        return cls(cfg, embedder, documents, embeddings, manifest, reranker=reranker)

    def cost_report(self) -> dict[str, Any]:
        return self.embedder.cost_report()

    @property
    def cache_signature(self) -> str:
        reranker_endpoint = getattr(self.reranker, "endpoint", None)
        payload = {
            "version": RETRIEVAL_VERSION,
            "corpus_hash": self.manifest.get("corpus_hash", ""),
            "embedding_model": self.embedder.model,
            "top_k": self.cfg.top_k,
            "candidate_k": self.cfg.candidate_k,
            "dense_weight": self.cfg.dense_weight,
            "llm_rerank_enabled": self.cfg.llm_rerank_enabled,
            "llm_rerank_k": self.cfg.llm_rerank_k,
            "reranker_model": getattr(reranker_endpoint, "model", ""),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _bm25(self, query_tokens: Iterable[str], indices: np.ndarray) -> np.ndarray:
        query = list(dict.fromkeys(query_tokens))
        scores = np.zeros(len(indices), dtype=np.float32)
        n_documents = max(len(self.documents), 1)
        k1, b = 1.5, 0.75
        for token in query:
            df = self._document_frequency.get(token, 0)
            if not df:
                continue
            idf = math.log(1.0 + (n_documents - df + 0.5) / (df + 0.5))
            for position, document_index in enumerate(indices):
                frequency = self._token_counts[int(document_index)].get(token, 0)
                if not frequency:
                    continue
                length = float(self._doc_lengths[int(document_index)])
                denominator = frequency + k1 * (
                    1.0 - b + b * length / max(self._avg_length, 1.0)
                )
                scores[position] += idf * frequency * (k1 + 1.0) / denominator
        return scores

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if not len(values):
            return values
        low, high = float(values.min()), float(values.max())
        if math.isclose(low, high):
            return np.ones_like(values) if high > 0 else np.zeros_like(values)
        return (values - low) / (high - low)

    def _llm_rerank(self, query: str, code: str,
                    results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Réordonne uniquement les candidats existants, sans créer d'article."""
        if not (self.cfg.llm_rerank_enabled and self.reranker and results):
            return results
        payload = [{
            "article_number": item["article_number"],
            "excerpt": item["excerpt"],
            "hybrid_score": item["score"],
        } for item in results[:max(int(self.cfg.llm_rerank_k), 1)]]
        try:
            answer = self.reranker.complete_json(
                "retrieval_reranker",
                [
                    {
                        "role": "system",
                        "content": (
                            "Tu es un reranker de recherche législative québécoise. "
                            "Classe seulement les articles candidats fournis selon leur "
                            "capacité à répondre directement à la question. Pénalise un "
                            "article limité à un contexte spécial absent de la question "
                            "(autre province, appel, exécution, etc.). N'invente aucun "
                            "numéro. Réponds uniquement par l'objet JSON "
                            '{"ranking":["numéro", "numéro"]}, contenant chaque numéro '
                            "candidat exactement une fois."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "code": code,
                            "question": query,
                            "candidats": payload,
                        }, ensure_ascii=False),
                    },
                ],
                temperature=0.0,
            )
        except Exception:
            return results
        allowed = {str(item["article_number"]): item for item in results}
        ranking = answer.get("ranking")
        if not isinstance(ranking, list):
            return results
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for value in ranking:
            key = str(value).strip()
            if key in allowed and key not in seen:
                ordered.append(allowed[key])
                seen.add(key)
        # Une sortie incomplète ne supprime jamais un candidat récupéré.
        ordered.extend(item for item in results if item["article_number"] not in seen)
        for position, item in enumerate(ordered, start=1):
            item["rerank_position"] = position
            item["reranker"] = "llm"
        return ordered

    def search(self, query: str, code: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            raise RAGError("la requête sémantique est vide")
        code = code.upper()
        if code not in CODE_NAMES:
            raise RAGError(f"code inconnu: {code}")
        indices = np.asarray(
            [index for index, doc in enumerate(self.documents) if doc.code == code],
            dtype=np.int64,
        )
        if not len(indices):
            return []
        query_vector = _normalize_rows(self.embedder.embed([query]))[0]
        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise RAGError("dimension d'embedding incompatible avec l'index")

        dense = self.embeddings[indices] @ query_vector
        lexical = self._bm25(_expanded_query_tokens(query), indices)
        candidate_k = min(max(self.cfg.candidate_k, 1), len(indices))
        dense_positions = np.argsort(-dense)[:candidate_k]
        lexical_positions = np.argsort(-lexical)[:candidate_k]
        candidate_positions = np.asarray(
            list(dict.fromkeys([*dense_positions.tolist(), *lexical_positions.tolist()])),
            dtype=np.int64,
        )
        dense_normalized = self._minmax(dense[candidate_positions])
        lexical_normalized = self._minmax(lexical[candidate_positions])
        weight = min(max(float(self.cfg.dense_weight), 0.0), 1.0)
        reranked = weight * dense_normalized + (1.0 - weight) * lexical_normalized
        order = np.argsort(-reranked)
        wanted = min(max(int(top_k or self.cfg.top_k), 1), len(order), 20)

        pre_rerank_count = min(
            max(wanted, int(self.cfg.llm_rerank_k)), len(order), 20)
        results: list[dict[str, Any]] = []
        for rank, order_position in enumerate(order[:pre_rerank_count], start=1):
            candidate_position = int(candidate_positions[int(order_position)])
            document_index = int(indices[candidate_position])
            document = self.documents[document_index]
            results.append({
                "rank": rank,
                "code": document.code,
                "article": document.article_label,
                "article_number": document.article_number,
                "score": round(float(reranked[int(order_position)]), 6),
                "dense_score": round(float(dense[candidate_position]), 6),
                "lexical_score": round(float(lexical[candidate_position]), 6),
                "excerpt": document.text[:700],
                "source_url": document.source_url,
            })
        return self._llm_rerank(query, code, results)[:wanted]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        code = {
            "semantic_search_ccq": "CCQ",
            "semantic_search_cpc": "CPC",
        }.get(name)
        if not code:
            raise RAGError(f"outil RAG inconnu: {name}")
        results = self.search(
            str(arguments.get("query") or ""), code,
            top_k=int(arguments.get("top_k") or self.cfg.top_k),
        )
        if not results:
            text = f"Aucun article {code} trouvé."
        else:
            lines = [
                f"{item['article']} — score de pertinence {item['score']:.3f}"
                for item in results
            ]
            text = "\n\n".join(lines)
        return {
            "text": text,
            "query": arguments.get("query", ""),
            "code": code,
            "results": results,
            "retrieval": (
                "dense_candidates_then_bm25_dense_then_llm_rerank"
                if self.cfg.llm_rerank_enabled and self.reranker
                else "dense_candidates_then_bm25_dense_rerank"
            ),
        }
