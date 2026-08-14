# -*- coding: utf-8 -*-
"""Reconstruit l'index sémantique CCQ/CPC de la démonstration.

L'index livré (``data/agentic/rag_index/``) est exclu du dépôt : un clone
neuf n'en reçoit aucun, et ``/health`` répond alors ``rag.loaded: false``
avec un repli lexical. ``build_index`` existait déjà en bibliothèque mais
n'avait plus de point d'entrée depuis le resserrement du dépôt sur la démo,
ce qui rendait la démonstration irreproductible ailleurs que sur la machine
d'origine.

Usage ::

    python -m lexior.agentic.build_rag_index
    python -m lexior.agentic.build_rag_index --force --limit 200

Prérequis : ``pip install -e ".[index]"`` (le corpus vient de Hugging Face)
et ``RAG_EMBEDDING_API_KEY`` (ou ``OPENAI_API_KEY``) dans ``../.env``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lexior.agentic.build_rag_index",
        description="Construit l'index sémantique CCQ/CPC utilisé en live.")
    parser.add_argument(
        "--config", default="configs/agentic_generation.yaml",
        help="fichier de configuration (défaut : %(default)s)")
    parser.add_argument(
        "--force", action="store_true",
        help="reconstruire même si un index est déjà présent")
    parser.add_argument(
        "--limit", type=int, default=-1,
        help="n'indexer que les N premiers articles (essai rapide)")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        parser.error(f"configuration introuvable : {config_path}")

    # Les clés vivent à la racine du dépôt parent, comme pour l'API.
    phase1 = config_path.parent.parent
    load_dotenv(phase1.parent / ".env")

    from .config import load_config
    from .legal_rag import RAGError, build_embedder, build_index, index_exists

    cfg = load_config(str(config_path))
    if not cfg.rag.enabled:
        print("[rag] rag.enabled est faux dans la configuration : rien à faire.")
        return 1

    print(f"[rag] index      : {cfg.rag.index_dir}")
    print(f"[rag] corpus     : {cfg.rag.dataset_name} ({cfg.rag.dataset_split})")
    print(f"[rag] embeddings : {cfg.rag.embedding_model}")
    if index_exists(cfg.rag.index_dir) and not args.force:
        print("[rag] un index existe déjà; ajouter --force pour le remplacer.")
        return 1

    try:
        embedder = build_embedder(cfg.rag, allow_remote_calls=True)
        manifest = build_index(cfg.rag, embedder,
                               force=args.force, limit=args.limit)
    except RAGError as exc:
        print(f"[rag] échec : {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"[rag] dépendance manquante : {exc}\n"
              f"      installer avec : pip install -e \".[index]\"",
              file=sys.stderr)
        return 1

    usage = manifest.get("usage", {}).get("total", {})
    print(f"[rag] terminé : {manifest.get('documents')} articles, "
          f"{manifest.get('dimensions')} dimensions, "
          f"coût {usage.get('cost_usd', 0):.4f} USD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
