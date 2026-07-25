# -*- coding: utf-8 -*-
"""Chargement du fichier ``.env`` du dépôt — un seul endroit.

Les clés vivent dans ``LexiorGpt1/.env``, jamais dans le code ni dans les
YAML. Ce module les charge une fois pour toutes; ``config.py`` et les
scripts se contentent ensuite de lire ``os.environ``.

Deux règles :

1. **l'environnement réel gagne toujours.** Une variable déjà définie dans
   le shell (ou par un orchestrateur, un CI, un pod RunPod) n'est jamais
   écrasée par le fichier. Sans quoi un ``OPENAI_API_KEY=… python …``
   ponctuel serait silencieusement ignoré.
2. **aucune dépendance obligatoire.** ``python-dotenv`` est utilisé s'il
   est installé, sinon un analyseur minimal prend le relais : importer
   ``lexior`` ne doit pas exiger un paquet de plus.

``LEXIOR_ENV_FILE`` force un chemin précis si nécessaire.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# src/lexior/env.py -> src/lexior -> src -> Phase1_Data_Preparation -> dépôt
_PACKAGE_DIR = Path(__file__).resolve().parent
PHASE1_DIR = _PACKAGE_DIR.parents[1]
REPO_ROOT = PHASE1_DIR.parent

# Ordre de recherche : la racine du dépôt d'abord, c'est là que vit le
# fichier de référence.
CANDIDATE_PATHS: tuple[Path, ...] = (
    REPO_ROOT / ".env",
    PHASE1_DIR / ".env",
)

_loaded: Optional[Path] = None


def _parse(text: str) -> dict[str, str]:
    """Analyseur minimal : ``CLÉ=valeur``, ``export`` et guillemets."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def env_file() -> Optional[Path]:
    """Fichier ``.env`` retenu, ou ``None`` s'il n'y en a aucun."""
    override = os.environ.get("LEXIOR_ENV_FILE")
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    for candidate in CANDIDATE_PATHS:
        if candidate.is_file():
            return candidate
    return None


def load_project_env(force: bool = False) -> Optional[Path]:
    """Charge le ``.env`` du dépôt. Idempotent.

    Retourne le chemin chargé, ou ``None`` si aucun fichier n'existe —
    absence parfaitement normale en production, où les variables viennent
    de l'orchestrateur.
    """
    global _loaded
    if _loaded is not None and not force:
        return _loaded

    path = env_file()
    if path is None:
        return None

    try:
        from dotenv import load_dotenv
    except ImportError:
        for key, value in _parse(path.read_text(encoding="utf-8")).items():
            os.environ.setdefault(key, value)
    else:
        # override=False : l'environnement réel reste prioritaire.
        load_dotenv(path, override=False)

    _loaded = path
    return path


def describe() -> dict[str, object]:
    """État du chargement, sans jamais exposer une valeur."""
    path = env_file()
    return {
        "env_file": str(path) if path else "",
        "loaded": _loaded is not None,
        "keys_present": sorted(
            _parse(path.read_text(encoding="utf-8")) if path else {}),
    }
