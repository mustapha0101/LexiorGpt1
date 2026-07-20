# -*- coding: utf-8 -*-

"""
Configuration du pipeline agentique.

Priorité : variables d'environnement > YAML > défauts.

Noms indépendants du fournisseur :
    TEACHER_BASE_URL / TEACHER_API_KEY / TEACHER_MODEL
    TEACHER_TIMEOUT / TEACHER_MAX_RETRIES / TEACHER_CONCURRENCY
    CRITIC_BASE_URL / CRITIC_API_KEY / CRITIC_MODEL

Rétrocompatibilité temporaire (pipeline legacy) :
    TEACHER_BASE_URL  <- OPENAI_BASE_URL
    TEACHER_API_KEY   <- OPENAI_API_KEY
    TEACHER_MODEL     <- GEN_MODEL

Une vraie clé OpenAI n'est nécessaire que si le base_url cible réellement
api.openai.com. Pour un vLLM auto-hébergé sans authentification, une valeur
factice suffit — c'est le serveur qui décide.

AUCUN secret n'est journalisé : seuls des hachés tronqués apparaissent dans
les manifestes et les sorties.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "agentic_generation.yaml")


def _env(name: str, *fallbacks: str, default: Optional[str] = None) -> Optional[str]:
    for key in (name, *fallbacks):
        val = os.environ.get(key)
        if val:
            return val
    return default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def hash_of(value: str, length: int = 12) -> str:
    """Haché tronqué, pour référencer un secret sans jamais l'exposer."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


@dataclass
class EndpointConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    max_retries: int = 3
    concurrency: int = 4
    temperature: float = 0.3

    @property
    def base_url_hash(self) -> str:
        return hash_of(self.base_url)

    def redacted(self) -> dict[str, Any]:
        """Représentation SANS secret, pour les manifestes et le doctor."""
        return {
            "base_url_hash": self.base_url_hash,
            "api_key_set": bool(self.api_key),
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "concurrency": self.concurrency,
        }


@dataclass
class RAGConfig:
    """Configuration de la recherche sémantique locale CCQ/CPC."""

    enabled: bool = True
    index_dir: str = "data/agentic/rag_index"
    dataset_name: str = "intelliwork/canadian-quebec-law-corpus"
    dataset_split: str = "train"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 128
    embedding_price_per_1m_usd: float = 0.02
    top_k: int = 8
    candidate_k: int = 40
    dense_weight: float = 0.60
    llm_rerank_enabled: bool = True
    llm_rerank_k: int = 10

    def redacted(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "index_dir": self.index_dir,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "embedding_base_url_hash": hash_of(self.embedding_base_url),
            "embedding_api_key_set": bool(self.embedding_api_key),
            "embedding_model": self.embedding_model,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_price_per_1m_usd": self.embedding_price_per_1m_usd,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "dense_weight": self.dense_weight,
            "llm_rerank_enabled": self.llm_rerank_enabled,
            "llm_rerank_k": self.llm_rerank_k,
        }


@dataclass
class AgenticConfig:
    # --- exécution -------------------------------------------------------
    seed: int = 3407
    language: str = "fr"
    target_accepted: int = 100
    max_scenarios: int = -1            # -1 = illimité (borné par target)
    max_tool_calls: int = 4
    dry_run: bool = False
    offline: bool = False
    allow_remote_calls: bool = False   # refus par défaut : sécurité
    no_critics: bool = False
    resume: bool = True
    push_to_hf: bool = False

    # --- limites de contenu ----------------------------------------------
    max_tool_response_chars: int = 6000
    max_doc_fetch_chars: int = 12000
    near_duplicate_jaccard: float = 0.90

    # --- seuils critiques ------------------------------------------------
    legal_min_score: float = 0.7
    agentic_min_score: float = 0.7
    max_repairs: int = 1

    # --- chemins ---------------------------------------------------------
    data_root: str = "data/agentic"
    catalog_path: str = ""
    mcp_config_path: str = ""

    # --- endpoints -------------------------------------------------------
    teacher: EndpointConfig = field(default_factory=EndpointConfig)
    critic: EndpointConfig = field(default_factory=EndpointConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    # --- versions --------------------------------------------------------
    prompt_version: str = "agentic-2.0-distillation"

    # --- taxonomie / mélange / split (depuis le YAML) --------------------
    taxonomy_proportions: dict[str, float] = field(default_factory=dict)
    mix: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    hf_dataset_repo_id: str = ""

    def redacted(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "language": self.language,
            "target_accepted": self.target_accepted,
            "max_scenarios": self.max_scenarios,
            "max_tool_calls": self.max_tool_calls,
            "dry_run": self.dry_run,
            "offline": self.offline,
            "allow_remote_calls": self.allow_remote_calls,
            "no_critics": self.no_critics,
            "max_tool_response_chars": self.max_tool_response_chars,
            "max_doc_fetch_chars": self.max_doc_fetch_chars,
            "near_duplicate_jaccard": self.near_duplicate_jaccard,
            "legal_min_score": self.legal_min_score,
            "agentic_min_score": self.agentic_min_score,
            "max_repairs": self.max_repairs,
            "data_root": self.data_root,
            "prompt_version": self.prompt_version,
            "teacher": self.teacher.redacted(),
            "critic": self.critic.redacted(),
            "rag": self.rag.redacted(),
            "taxonomy_proportions": self.taxonomy_proportions,
            "mix": self.mix,
            "split": self.split,
            "hf_dataset_repo_id": self.hf_dataset_repo_id,
        }


def load_config(config_path: Optional[str] = None,
                overrides: Optional[dict[str, Any]] = None) -> AgenticConfig:
    """Charge la configuration : YAML puis environnement puis overrides CLI."""
    path = config_path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(base_dir)

    cfg = AgenticConfig()
    gen = raw.get("generation", {})
    cfg.seed = int(gen.get("seed", cfg.seed))
    cfg.language = gen.get("language", cfg.language)
    cfg.target_accepted = int(gen.get("target_accepted", cfg.target_accepted))
    cfg.max_tool_calls = int(gen.get("max_tool_calls", cfg.max_tool_calls))
    cfg.max_tool_response_chars = int(gen.get("max_tool_response_chars",
                                              cfg.max_tool_response_chars))
    cfg.max_doc_fetch_chars = int(gen.get("max_doc_fetch_chars",
                                          cfg.max_doc_fetch_chars))
    cfg.near_duplicate_jaccard = float(gen.get("near_duplicate_jaccard",
                                               cfg.near_duplicate_jaccard))
    cfg.prompt_version = gen.get("prompt_version", cfg.prompt_version)

    critics = raw.get("critics", {})
    cfg.legal_min_score = float(critics.get("legal_min_score", cfg.legal_min_score))
    cfg.agentic_min_score = float(critics.get("agentic_min_score", cfg.agentic_min_score))
    cfg.max_repairs = int(critics.get("max_repairs", cfg.max_repairs))

    paths = raw.get("paths", {})
    cfg.data_root = paths.get("data_root", cfg.data_root)
    cfg.catalog_path = paths.get(
        "catalog", os.path.join(repo_root, "docs", "mcp_tools_catalog.json"))
    cfg.mcp_config_path = paths.get(
        "mcp_config", os.path.join(repo_root, ".mcp.json"))
    # Chemins relatifs : ancrés sur Phase1_Data_Preparation.
    for attr in ("data_root", "catalog_path", "mcp_config_path"):
        val = getattr(cfg, attr)
        if val and not os.path.isabs(val):
            setattr(cfg, attr, os.path.normpath(os.path.join(base_dir, val)))

    # --- Teacher : env > YAML, avec rétrocompatibilité OPENAI_*/GEN_MODEL --
    teacher_yaml = raw.get("teacher", {})
    cfg.teacher = EndpointConfig(
        base_url=_env("TEACHER_BASE_URL", "OPENAI_BASE_URL",
                      default=teacher_yaml.get("base_url", "")) or "",
        api_key=_env("TEACHER_API_KEY", "OPENAI_API_KEY", default="") or "",
        model=_env("TEACHER_MODEL", "GEN_MODEL",
                   default=teacher_yaml.get("model", "")) or "",
        timeout=_env_float("TEACHER_TIMEOUT",
                           float(teacher_yaml.get("timeout", 120.0))),
        max_retries=_env_int("TEACHER_MAX_RETRIES",
                             int(teacher_yaml.get("max_retries", 3))),
        concurrency=_env_int("TEACHER_CONCURRENCY",
                             int(teacher_yaml.get("concurrency", 4))),
        temperature=float(teacher_yaml.get("temperature", 0.3)),
    )

    # --- Critic : par défaut, même endpoint que le Teacher -----------------
    critic_yaml = raw.get("critic", {})
    cfg.critic = EndpointConfig(
        base_url=_env("CRITIC_BASE_URL",
                      default=critic_yaml.get("base_url", "")) or cfg.teacher.base_url,
        api_key=_env("CRITIC_API_KEY", default="") or cfg.teacher.api_key,
        model=_env("CRITIC_MODEL",
                   default=critic_yaml.get("model", "")) or cfg.teacher.model,
        timeout=cfg.teacher.timeout,
        max_retries=cfg.teacher.max_retries,
        concurrency=cfg.teacher.concurrency,
        temperature=float(critic_yaml.get("temperature", 0.0)),
    )

    # --- RAG CCQ/CPC : embeddings OpenAI, indépendants du Teacher ---------
    rag_yaml = raw.get("rag", {})
    rag_index_dir = str(rag_yaml.get("index_dir", RAGConfig.index_dir))
    if rag_index_dir and not os.path.isabs(rag_index_dir):
        rag_index_dir = os.path.normpath(os.path.join(base_dir, rag_index_dir))
    cfg.rag = RAGConfig(
        enabled=_env_bool("RAG_ENABLED", bool(rag_yaml.get("enabled", True))),
        index_dir=rag_index_dir,
        dataset_name=_env(
            "RAG_DATASET_NAME",
            default=str(rag_yaml.get("dataset_name", RAGConfig.dataset_name)),
        ) or RAGConfig.dataset_name,
        dataset_split=_env(
            "RAG_DATASET_SPLIT",
            default=str(rag_yaml.get("dataset_split", RAGConfig.dataset_split)),
        ) or RAGConfig.dataset_split,
        embedding_base_url=_env(
            "RAG_EMBEDDING_BASE_URL",
            default=str(rag_yaml.get("embedding_base_url", RAGConfig.embedding_base_url)),
        ) or RAGConfig.embedding_base_url,
        embedding_api_key=_env(
            "RAG_EMBEDDING_API_KEY", "OPENAI_API_KEY", default=""
        ) or "",
        embedding_model=_env(
            "RAG_EMBEDDING_MODEL",
            default=str(rag_yaml.get("embedding_model", RAGConfig.embedding_model)),
        ) or RAGConfig.embedding_model,
        embedding_batch_size=_env_int(
            "RAG_EMBEDDING_BATCH_SIZE",
            int(rag_yaml.get("embedding_batch_size", RAGConfig.embedding_batch_size)),
        ),
        embedding_price_per_1m_usd=_env_float(
            "RAG_EMBEDDING_PRICE_PER_1M_USD",
            float(rag_yaml.get("embedding_price_per_1m_usd",
                               RAGConfig.embedding_price_per_1m_usd)),
        ),
        top_k=int(rag_yaml.get("top_k", RAGConfig.top_k)),
        candidate_k=int(rag_yaml.get("candidate_k", RAGConfig.candidate_k)),
        dense_weight=float(rag_yaml.get("dense_weight", RAGConfig.dense_weight)),
        llm_rerank_enabled=_env_bool(
            "RAG_LLM_RERANK_ENABLED",
            bool(rag_yaml.get("llm_rerank_enabled", RAGConfig.llm_rerank_enabled)),
        ),
        llm_rerank_k=int(rag_yaml.get("llm_rerank_k", RAGConfig.llm_rerank_k)),
    )

    cfg.taxonomy_proportions = dict(raw.get("taxonomy", {}).get("proportions", {}))
    cfg.mix = dict(raw.get("mix", {}))
    cfg.mix["include_legacy_legal"] = _env_bool(
        "INCLUDE_LEGACY_LEGAL", bool(cfg.mix.get("include_legacy_legal", False)))
    cfg.mix["include_identity_data"] = _env_bool(
        "INCLUDE_IDENTITY_DATA", bool(cfg.mix.get("include_identity_data", True)))
    cfg.split = dict(raw.get("split", {}))
    cfg.hf_dataset_repo_id = _env(
        "HF_DATASET_REPO_ID",
        default=raw.get("hf", {}).get("dataset_repo_id", "")) or ""

    for key, val in (overrides or {}).items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg
