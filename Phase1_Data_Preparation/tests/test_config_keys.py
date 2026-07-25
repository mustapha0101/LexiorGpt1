# -*- coding: utf-8 -*-
"""Aucune clé de configuration ne doit être documentée sans être appliquée.

Le manifeste d'un run enregistre la configuration. Si une clé y figure sans
être lue nulle part, le manifeste décrit des paramètres qui ne sont pas ceux
réellement appliqués — un problème de reproductibilité, pas de style.

Ce test échoue dès qu'une clé est ajoutée au YAML ou à ``redacted()`` sans
site de consommation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from lexior.agentic.config import RAGConfig, load_config

PHASE1 = Path(__file__).resolve().parents[1]
CONFIG_DIR = PHASE1 / "configs"
SOURCE_DIR = PHASE1 / "src" / "lexior"

# Clés dont le seul rôle légitime est d'apparaître dans le manifeste.
MANIFEST_ONLY = {
    # Calculé dans config.py, publié pour que le repli critic→teacher soit
    # visible sans lire le code (LOT 4.2).
    "critic_is_teacher",
}

# Sous-objets sérialisés à part.
NESTED = {"teacher", "critic", "rag"}


@pytest.fixture(scope="module")
def sources() -> str:
    """Tout le code du paquet, sauf le module de configuration lui-même."""
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SOURCE_DIR.rglob("*.py"))
        if path.name != "config.py"
    )


def _is_read(name: str, sources: str) -> bool:
    """Attribut lu quelque part : accès direct ou via ``getattr``."""
    return bool(
        re.search(rf"\.{re.escape(name)}\b", sources)
        or re.search(rf"getattr\([^)]*[\"']{re.escape(name)}[\"']", sources)
    )


def test_every_published_config_key_is_consumed(sources):
    config = load_config(str(CONFIG_DIR / "agentic_generation.yaml"))
    dead = [
        key for key in config.redacted()
        if key not in NESTED and key not in MANIFEST_ONLY
        and not _is_read(key, sources)
    ]

    assert not dead, (
        f"clés publiées dans le manifeste mais lues nulle part : {dead}. "
        f"Les câbler ou les retirer — pas d'entre-deux.")


def test_every_rag_key_is_consumed(sources):
    dead = [
        name for name in RAGConfig.__dataclass_fields__
        if not _is_read(name, sources)
    ]

    assert not dead, f"clés RAG mortes : {dead}"


@pytest.mark.parametrize(
    "config_name", sorted(path.name for path in CONFIG_DIR.glob("*.yaml")))
def test_every_yaml_section_is_consumed(config_name, sources):
    """Les sections imbriquées sont lues par nom de clé littéral."""
    raw = yaml.safe_load(
        (CONFIG_DIR / config_name).read_text(encoding="utf-8")) or {}

    dead: list[str] = []
    for section in ("split",):
        for key in (raw.get(section) or {}):
            if f'"{key}"' not in sources and f"'{key}'" not in sources:
                dead.append(f"{section}.{key}")

    assert not dead, (
        f"{config_name} : clés déclarées mais jamais lues : {dead}")


def test_the_grouping_dimensions_are_all_implemented():
    """``split.group_by`` ne doit lister que des dimensions réelles."""
    from lexior.agentic.publisher import GROUP_DIMENSIONS

    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        declared = (raw.get("split") or {}).get("group_by") or []
        unknown = [name for name in declared if name not in GROUP_DIMENSIONS]
        assert not unknown, (
            f"{path.name} : dimensions inexistantes {unknown}; "
            f"implémentées : {sorted(GROUP_DIMENSIONS)}")


def test_the_split_ratios_add_up(sources):
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        split = raw.get("split") or {}
        if not split:
            continue
        total = sum(float(split.get(name, 0.0))
                    for name in ("train", "validation", "test"))
        assert abs(total - 1.0) < 1e-6, f"{path.name} : somme {total}"
