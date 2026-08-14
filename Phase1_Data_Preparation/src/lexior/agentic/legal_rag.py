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
import time
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
RETRIEVAL_VERSION = "legal-rag-2.0-stable-hybrid-rerank"

# Saturation BM25 pour ramener un score lexical non borné dans [0, 1[. Seule
# la forme de la courbe compte : les planchers sont calibrés après coup sur
# tests/fixtures/retrieval_gold.jsonl.
BM25_SATURATION = 5.0


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
        """Texte indexé historique : étiquettes puis contenu."""
        return search_text_for(self, "full")


SEARCH_TEXT_MODES = ("full", "text_only", "text_taxonomy")


def search_text_for(document: "LegalDocument", mode: str = "full") -> str:
    """Texte soumis à l'embedder.

    Trois variantes, du plus au moins étiqueté :

    ``full``
        historique — titre, libellé, domaine, taxonomie, puis contenu. Les
        étiquettes pèsent 22 % du texte embarqué en médiane et jusqu'à
        71 % sur un article court, pour 16 taxonomies distinctes seulement
        sur 4 278 articles.
    ``text_taxonomy``
        retire les deux étiquettes de PURE RÉPÉTITION — « CCQ Article
        1457 » et « Article 1457 » disent la même chose, et un numéro
        d'article ne porte aucun sens exploitable par un embedding. Garde
        le domaine et la taxonomie, qui situent l'article dans le Code.
    ``text_only``
        contenu normatif seul.
    """
    if mode == "text_only":
        return document.text
    if mode == "text_taxonomy":
        return "\n".join(part for part in (
            document.domain,
            document.taxonomy.replace("_", " ").replace("/", " "),
            document.text,
        ) if part)
    if mode != "full":
        raise RAGError(
            f"mode de texte indexé inconnu : {mode!r} "
            f"(attendu parmi {SEARCH_TEXT_MODES})")
    return "\n".join(part for part in (
        document.title,
        document.article_label,
        document.domain,
        document.taxonomy.replace("_", " ").replace("/", " "),
        document.text,
    ) if part)


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in value if not unicodedata.combining(ch)).casefold()


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(_fold(value))


# Équivalents entre le vocabulaire courant et celui du Code. Au niveau du
# module pour être mesurables : l'effet de chaque entrée se chiffre sur
# tests/fixtures/retrieval_gold.jsonl.
QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
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


def _expanded_query_tokens(value: str) -> list[str]:
    """Ajoute des équivalents procéduraux sans remplacer la requête originale.

    Les embeddings portent le sens général; ces expansions très limitées
    aident BM25 lorsque l'utilisateur emploie un verbe courant différent du
    terme exact du Code (par exemple « assigner » plutôt que « citer »).
    """
    tokens = _tokens(value)
    token_set = set(tokens)
    for token in list(token_set):
        tokens.extend(QUERY_EXPANSIONS.get(token, ()))
    return list(dict.fromkeys(tokens))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _stable_descending(scores: np.ndarray,
                       tie_breaker: np.ndarray | None = None) -> np.ndarray:
    """Classe les scores décroissants avec un départage reproductible.

    ``numpy.argsort`` utilise un tri non stable par défaut. Deux articles au
    même score pouvaient donc changer d'ordre selon la plateforme ou une
    reconstruction de l'index. Le rang du document dans le corpus sert ici de
    clé secondaire explicite.
    """
    values = np.asarray(scores, dtype=np.float64)
    ties = (np.arange(len(values), dtype=np.int64)
            if tie_breaker is None else np.asarray(tie_breaker, dtype=np.int64))
    return np.lexsort((ties, -values))


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


class BGEEmbedder:
    """Embedder local multilingue — BAAI/bge-m3, licence MIT.

    Alternative à ``OpenAIEmbedder``, qu'il ne remplace pas : les deux
    respectent le Protocol ``Embedder`` et le choix se fait par
    configuration (``RAGConfig.embedding_provider``). Les dimensions
    diffèrent (1024 contre 1536), donc les index ne sont PAS
    interchangeables — chacun vit dans son répertoire.

    Aucun appel réseau après le téléchargement initial des poids
    (~2,3 Go, mis en cache par huggingface_hub).
    """

    def __init__(self, cfg: RAGConfig, allow_remote_calls: bool = True,
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = cfg.embedding_model or "BAAI/bge-m3"
        self.device = device
        self.calls = 0
        self.failed_calls = 0
        self.texts_in = 0
        self.seconds = 0.0
        self._encoder = SentenceTransformer(self.model, device=device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        self.calls += 1
        started = time.monotonic()
        try:
            vectors = self._encoder.encode(
                list(texts), batch_size=8, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True)
        except Exception:
            self.failed_calls += 1
            raise
        self.seconds += time.monotonic() - started
        self.texts_in += len(texts)
        return np.asarray(vectors, dtype=np.float32)

    def cost_report(self) -> dict[str, Any]:
        """Même forme que l'embedder distant : coût nul, temps mesuré.

        Le coût se déplace de la facture vers la latence — c'est
        précisément ce que la comparaison doit rendre visible.
        """
        total = {
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "tokens_in": 0,
            "tokens_cached_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "texts_in": self.texts_in,
            "seconds": round(self.seconds, 3),
            "ms_per_text": round(
                self.seconds * 1000 / self.texts_in, 1) if self.texts_in else 0.0,
        }
        return {"model": self.model, "device": self.device, "total": total}


def build_embedder(cfg: RAGConfig, allow_remote_calls: bool) -> Embedder:
    """Embedder désigné par ``cfg.embedding_provider``."""
    provider = (cfg.embedding_provider or "openai").strip().lower()
    if provider == "openai":
        return OpenAIEmbedder(cfg, allow_remote_calls)
    if provider in ("bge", "bge-m3", "local"):
        return BGEEmbedder(cfg, allow_remote_calls)
    raise RAGError(
        f"fournisseur d'embeddings inconnu : {provider!r} "
        "(attendu 'openai' ou 'bge')")


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
        # Dernier diagnostic de rejet du reranker. Il est conservé pour
        # l'observabilité, sans retirer de candidat de la recherche.
        self.last_rerank_rejection: dict[str, Any] | None = None
        # Moyennes de recentrage, dérivées des vecteurs indexés : elles
        # correspondent donc TOUJOURS au corpus effectivement chargé, sans
        # risque de désynchronisation avec un fichier écrit séparément.
        self.centering = (cfg.centering or "none").strip().lower()
        self._means = self._compute_means()
        self._centered: dict[str, np.ndarray] = {}
        # BM25 est indexé sur le MÊME texte que les vecteurs : garder les
        # étiquettes d'un côté et pas de l'autre introduirait une seconde
        # variable. L'effet propre aux embeddings s'isole en mesurant à
        # dense_weight = 1.0.
        self.search_text_fields = (cfg.search_text_fields or "full").strip()
        self._token_counts = [
            Counter(_tokens(search_text_for(doc, self.search_text_fields)))
            for doc in documents
        ]
        self._doc_lengths = np.asarray(
            [sum(counts.values()) for counts in self._token_counts], dtype=np.float32
        )
        self._avg_length = float(self._doc_lengths.mean()) if len(documents) else 1.0
        self._document_frequency = Counter()
        for counts in self._token_counts:
            self._document_frequency.update(counts.keys())

    # ── Recentrage ───────────────────────────────────────────────────────

    CENTERING_MODES = ("none", "global", "per_code")

    def _compute_means(self) -> dict[str, np.ndarray]:
        """Moyenne(s) à soustraire, selon le mode configuré.

        « global » retire ce que TOUT le corpus a en commun; « per_code »
        retire ce que chaque code a en propre. ``search()`` filtrant déjà
        par code, per_code compare des vecteurs à la moyenne du sous-corpus
        réellement en jeu.
        """
        if self.centering not in self.CENTERING_MODES:
            raise RAGError(
                f"mode de recentrage inconnu : {self.centering!r} "
                f"(attendu parmi {self.CENTERING_MODES})")
        if self.centering == "none" or not len(self.embeddings):
            return {}
        if self.centering == "global":
            return {"": self.embeddings.mean(axis=0)}
        means: dict[str, np.ndarray] = {}
        for code in sorted({doc.code for doc in self.documents}):
            rows = [index for index, doc in enumerate(self.documents)
                    if doc.code == code]
            if rows:
                means[code] = self.embeddings[np.asarray(rows)].mean(axis=0)
        return means

    def _mean_for(self, code: str) -> np.ndarray | None:
        if not self._means:
            return None
        return self._means.get("" if self.centering == "global" else code)

    @staticmethod
    def _recenter(matrix: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """Soustrait la moyenne PUIS renormalise.

        Sans renormalisation, les vecteurs cessent d'être unitaires et le
        produit scalaire n'est plus un cosinus : les scores deviennent
        sensibles à la norme, pas seulement à la direction. C'est l'erreur
        classique de cette transformation.
        """
        return _normalize_rows(np.asarray(matrix, dtype=np.float32) - mean)

    def _centered_matrix(self, code: str, indices: np.ndarray) -> np.ndarray:
        """Vecteurs du code, recentrés et renormalisés, mis en cache."""
        mean = self._mean_for(code)
        if mean is None:
            return self.embeddings[indices]
        if code not in self._centered:
            self._centered[code] = self._recenter(
                self.embeddings[indices], mean)
        return self._centered[code]

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
            "min_dense_score": self.cfg.min_dense_score,
            "min_hybrid_score": self.cfg.min_hybrid_score,
            "centering": self.centering,
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
        """Normalisation de CLASSEMENT uniquement.

        Elle ramène toujours le meilleur candidat à 1.0, y compris quand
        aucun ne répond à la question : ne jamais s'en servir pour décider
        de la pertinence. Voir ``_absolute_scores``.
        """
        if not len(values):
            return values
        low, high = float(values.min()), float(values.max())
        if math.isclose(low, high):
            return np.ones_like(values) if high > 0 else np.zeros_like(values)
        return (values - low) / (high - low)

    def _absolute_scores(self, dense: np.ndarray,
                         lexical: np.ndarray) -> np.ndarray:
        """Score hybride sur une échelle comparable d'une requête à l'autre.

        Le cosinus est déjà absolu; BM25 ne l'est pas et sature ici dans
        [0, 1[. Contrairement au score min-max, ce score reste bas quand
        aucun article ne correspond.
        """
        weight = min(max(float(self.cfg.dense_weight), 0.0), 1.0)
        saturated = lexical / (lexical + BM25_SATURATION)
        return weight * dense + (1.0 - weight) * saturated

    def _above_floor(self, dense: np.ndarray,
                     absolute: np.ndarray) -> np.ndarray:
        """Masque des candidats qui dépassent les deux planchers absolus."""
        return ((dense >= float(self.cfg.min_dense_score))
                & (absolute >= float(self.cfg.min_hybrid_score)))

    def _exemptes_du_plancher(self, candidats: np.ndarray,
                              positions_dense: np.ndarray) -> np.ndarray:
        """Masque des candidats que le plancher ne doit PAS écarter.

        Le plancher filtre sur le score dense, pas sur l'origine du candidat.
        Or son travail utile est d'écarter ce qui entre par BM25 seul : il a
        été calibré sur des requêtes en style juridique, qui scorent 0,58 à
        0,70, et rejetait donc le langage naturel par construction — 0,305
        pour « mon fils a cassé la vitrine du dépanneur », où le bon article
        est pourtant au rang 8 du canal dense.

        Les N premiers du canal dense en sont exemptés. La largeur compte :
        exempter TOUT le top-k (40) dégrade le jeu annoté — hit@3 0,500 ->
        0,475, MRR 0,405 -> 0,389 — parce que des candidats médiocres entrent
        et faussent la normalisation min-max. À 10, rien ne bouge sur le jeu
        annoté et les formulations en langage naturel sont rattrapées.
        """
        largeur = int(getattr(self.cfg, "dense_floor_exempt_top_k", 0) or 0)
        if largeur <= 0:
            return np.zeros(len(candidats), dtype=bool)
        return np.isin(candidats, positions_dense[:largeur])

    def _llm_rerank(self, query: str, code: str,
                    results: list[dict[str, Any]],
                    legal_terms: str = "",
                    preserve_top_k: int | None = None) -> list[dict[str, Any]]:
        """Réordonne les candidats sans réduire le rappel de la recherche.

        Le reranker est le seul point de la chaîne où la question rencontre
        le texte des articles : il peut donc mieux les classer. Il ne peut en
        revanche ni introduire un numéro absent, ni masquer un candidat du
        noyau initial. L'applicabilité est tranchée ensuite, sur les textes
        officiels complets, par le filtre dédié.
        """
        self.last_rerank_rejection = None
        if not (self.cfg.llm_rerank_enabled and self.reranker and results):
            return results
        judged = results[:max(int(self.cfg.llm_rerank_k), 1)]
        payload = [{
            "article_number": item["article_number"],
            "excerpt": item["excerpt"],
            "hybrid_score": item["score"],
        } for item in judged]
        try:
            answer = self.reranker.complete_json(
                "retrieval_reranker",
                [
                    {
                        "role": "system",
                        "content": (
                            "Tu es un reranker de recherche législative québécoise. "
                            "Classe seulement les articles candidats fournis selon leur "
                            "applicabilité aux faits et à la qualification juridique "
                            "fournies. Ne favorise pas un article uniquement parce qu'il "
                            "répète les mêmes objets ou mots que les faits : une règle "
                            "générale peut être pertinente même si son vocabulaire est plus "
                            "abstrait. À l'inverse, pénalise un article dont les conditions "
                            "d'application sont expressément incompatibles avec les faits "
                            "connus, ou limité à un contexte spécial absent de la question "
                            "(autre province, appel, exécution, etc.). Répartis les "
                            "numéros dans primary (règle la plus directement liée), "
                            "contextual (règle secondaire) et rejected "
                            "(incompatibilité manifeste). rejected est diagnostique et "
                            "ne supprime aucun candidat. N'invente aucun numéro. "
                            "Réponds uniquement par l'objet JSON "
                            '{"primary":["numéro"],"contextual":["numéro"],'
                            '"rejected":["numéro"],"reason":"une phrase"}. '
                            "Les listes peuvent être incomplètes; ne répète jamais un "
                            "numéro."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "code": code,
                            "faits": query,
                            "qualification_juridique": legal_terms,
                            "candidats": payload,
                        }, ensure_ascii=False),
                    },
                ],
                temperature=0.0,
            )
        except Exception:
            return results
        if not isinstance(answer, dict):
            return results
        allowed = {str(item["article_number"]): item for item in results}
        ranking = answer.get("ranking")
        primary = answer.get("primary")
        contextual = answer.get("contextual")
        # Compatibilité avec les rerankers déjà déployés : leur ``ranking``
        # devient primary, sans réintroduire de verrou sur le rang initial.
        if not isinstance(primary, list):
            primary = ranking
        if not isinstance(primary, list):
            return results
        if not isinstance(contextual, list):
            contextual = []
        # Un candidat non soumis au reranker n'a pas été jugé : il ne peut
        # pas être écarté.
        judged_numbers = {str(item["article_number"]) for item in judged}
        raw_rejected = answer.get("rejected")
        rejected = {
            str(value).strip() for value in raw_rejected
            if str(value).strip() in judged_numbers
        } if isinstance(raw_rejected, list) else set()

        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for values in (primary, contextual):
            for value in values:
                key = str(value).strip()
                if key in allowed and key not in seen:
                    ordered.append(allowed[key])
                    seen.add(key)
        reserve_size = (int(preserve_top_k) if preserve_top_k is not None
                        else int(getattr(
                            self.cfg, "rerank_recall_reserve_k", 2)))
        # La réserve protège le rappel sans protéger une position : le
        # reranker peut donc remonter un candidat classé loin par le hybride.
        for item in results[:max(reserve_size, 0)]:
            key = str(item["article_number"])
            if key not in seen:
                ordered.append(item)
                seen.add(key)
        ordered.extend(
            item for item in results
            if (str(item["article_number"]) not in seen
                and str(item["article_number"]) not in rejected))
        seen.update(str(item["article_number"]) for item in ordered)
        # rejected reste diagnostique : les candidats ne disparaissent pas.
        ordered.extend(item for item in results
                       if str(item["article_number"]) not in seen)
        reason = str(answer.get("reason") or "").strip()
        for position, item in enumerate(ordered, start=1):
            item["rerank_position"] = position
            item["reranker"] = "llm"
        if rejected:
            self.last_rerank_rejection = {
                "query": query, "code": code,
                "rejected": sorted(rejected),
                "kept": [item["article_number"] for item in ordered],
                "reason": reason,
            }
        return ordered

    def search(self, query: str, code: str, top_k: int | None = None,
               legal_terms: str = "") -> list[dict[str, Any]]:
        """Recherche hybride, éventuellement sur DEUX formulations.

        ``legal_terms`` est la même question rendue dans le vocabulaire du
        Code. Les deux recherches sont RÉUNIES, jamais substituées : le
        score d'un article est le meilleur des deux, DANS LES DEUX CANAUX.
        Le canal lexical ne recevait longtemps que ``query`` ; c'était à
        l'envers, puisque ``legal_terms`` porte les termes rares que BM25
        exploite le mieux.

        Mesuré sur tests/fixtures/retrieval_gold.jsonl : traduire en
        remplacement dégrade 10 des 32 questions où l'usager employait déjà
        le mot juste — « la liste des clients de mon employeur » passe du
        rang 1 au rang 74. L'union garde ces questions intactes et fait
        entrer les autres : l'article 1594 (« mise en demeure » quand la
        question dit « lettre d'avertissement ») passe du rang 413 au
        rang 17.
        """
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
        formulations = [query]
        terms = (legal_terms or "").strip()
        if terms:
            formulations.append(terms)
        # Un seul appel d'embeddings pour les deux formulations.
        vectors = _normalize_rows(self.embedder.embed(formulations))
        if vectors.shape[1] != self.embeddings.shape[1]:
            raise RAGError("dimension d'embedding incompatible avec l'index")

        # Chaque formulation est recentrée avec LA MÊME moyenne que le
        # corpus auquel on la compare : sinon les deux vivent dans des
        # systèmes de coordonnées différents et le produit scalaire ne veut
        # plus rien dire.
        mean = self._mean_for(code)
        if mean is not None:
            vectors = self._recenter(vectors, mean)
        matrix = self._centered_matrix(code, indices)
        dense_per_formulation = [matrix @ vector for vector in vectors]
        # Réunion : le meilleur des deux. Un article trouvé par l'une des
        # formulations est retenu, sans que l'autre puisse l'écarter.
        dense = dense_per_formulation[0]
        for other in dense_per_formulation[1:]:
            dense = np.maximum(dense, other)
        # Symétrie avec le canal dense. legal_terms portait jusqu'ici le
        # vocabulaire du Code — donc les termes RARES, « autorité parentale »,
        # « mise en demeure » — et n'atteignait que le dense : on refusait à
        # BM25 exactement ce qu'il exploite le mieux. Mesuré sur « mon fils a
        # cassé la vitrine » + « autorité parentale fait du mineur préjudice » :
        # l'article 1459 est au rang 1 du canal des mots pour la seconde
        # formulation, et au rang 1542 pour la première — c'est celui-là que
        # le mélange recevait.
        #
        # Meilleur score des deux, PAS une fusion des jetons : réunir les
        # jetons allongerait le document virtuel et diluerait l'IDF, qui est
        # précisément ce qui fait ressortir un terme rare.
        lexical_par_formulation = [
            self._bm25(_expanded_query_tokens(formulation), indices)
            for formulation in formulations]
        lexical = lexical_par_formulation[0]
        for autre in lexical_par_formulation[1:]:
            lexical = np.maximum(lexical, autre)
        candidate_k = min(max(self.cfg.candidate_k, 1), len(indices))
        dense_positions = _stable_descending(dense)[:candidate_k]
        lexical_positions = _stable_descending(lexical)[:candidate_k]
        candidate_positions = np.asarray(
            list(dict.fromkeys([*dense_positions.tolist(), *lexical_positions.tolist()])),
            dtype=np.int64,
        )
        absolute = self._absolute_scores(
            dense[candidate_positions], lexical[candidate_positions])
        # Aucun candidat au-dessus des planchers : le corpus ne répond pas.
        # `call()` produira « Aucun article trouvé », que le classifieur voit
        # comme `empty` et qui déclenche la reformulation.
        keep = (self._above_floor(dense[candidate_positions], absolute)
                | self._exemptes_du_plancher(candidate_positions,
                                             dense_positions))
        if not bool(keep.all()):
            candidate_positions = candidate_positions[keep]
            absolute = absolute[keep]
            if not len(candidate_positions):
                return []
        dense_normalized = self._minmax(dense[candidate_positions])
        lexical_normalized = self._minmax(lexical[candidate_positions])
        weight = min(max(float(self.cfg.dense_weight), 0.0), 1.0)
        reranked = weight * dense_normalized + (1.0 - weight) * lexical_normalized
        order = _stable_descending(reranked, candidate_positions)
        wanted = min(max(int(top_k or self.cfg.top_k), 1), len(order), 20)

        pre_rerank_count = min(
            max(wanted, int(self.cfg.llm_rerank_k)), len(order), 40)
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
                "absolute_score": round(
                    float(absolute[int(order_position)]), 6),
                "dense_score": round(float(dense[candidate_position]), 6),
                "found_by": ("legal_terms"
                             if len(dense_per_formulation) > 1
                             and dense_per_formulation[1][candidate_position]
                             > dense_per_formulation[0][candidate_position]
                             else "query"),
                "lexical_score": round(float(lexical[candidate_position]), 6),
                "excerpt": document.text[:700],
                "source_url": document.source_url,
            })
        final = self._llm_rerank(
            query, code, results, legal_terms=terms)[:wanted]
        # ``rank`` était figé AVANT le reranker, qui peut réordonner : le
        # champ annonçait une position que la liste ne respectait plus. On le
        # renumérote sur l'ordre réellement renvoyé.
        for position, item in enumerate(final, start=1):
            item["rank"] = position
        return final

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
            legal_terms=str(arguments.get("legal_terms") or ""),
        )
        if not results:
            text = f"Aucun article {code} trouvé."
        else:
            # Les articles étaient AFFICHÉS dans l'ordre du classement fusionné
            # mais ÉTIQUETÉS avec le score absolu, qui ne l'explique pas : sur
            # une question de dommage causé par un mineur, le bon article
            # sortait premier à 0,546 pendant que des articles hors sujet
            # affichaient 0,61-0,62. Un modèle qui se fie au nombre récupère
            # les mauvais articles — c'est arrivé avec Qwen là où gpt-4o-mini,
            # qui suivait l'ordre, réussissait. Les deux lectures étaient
            # défendables, une seule marchait.
            #
            # Le rang numéroté porte donc le classement, et le score est
            # nommé pour ce qu'il est : une confiance absolue, pas le critère
            # de tri. Le score de tri lui-même reste caché — en min-max il
            # vaut 1.000 pour le premier quoi qu'il arrive.
            lines = [
                f"{position}. {item['article']} — confiance "
                f"{item.get('absolute_score', item['score']):.3f}"
                for position, item in enumerate(results, start=1)
            ]
            text = (
                "Articles du plus au moins pertinent. Le rang fait foi ; la "
                "confiance est une mesure absolue, PAS le critère de "
                "classement — un article mieux classé peut afficher une "
                "confiance plus basse.\n\n" + "\n\n".join(lines))
        return {
            "text": text,
            "query": arguments.get("query", ""),
            "query_fingerprint": hashlib.sha256(json.dumps({
                "query": str(arguments.get("query") or "").strip(),
                "legal_terms": str(arguments.get("legal_terms") or "").strip(),
                "code": code,
                "retrieval_version": RETRIEVAL_VERSION,
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16],
            "code": code,
            "results": results,
            "retrieval": (
                "dense_candidates_then_bm25_dense_then_llm_rerank"
                if self.cfg.llm_rerank_enabled and self.reranker
                else "dense_candidates_then_bm25_dense_rerank"
            ),
        }
