# -*- coding: utf-8 -*-
"""Modes d'exécution du système central.

Deux modes, un seul graphe :

    dataset — un ``ScenarioSpec`` entre, une trajectoire d'entraînement
              validée sort (JSONL intermédiaire; la conversion ChatML
              reste un pas déterministe séparé).
    live    — un message utilisateur réel entre, une réponse sourcée
              sort; les clarifications passent par ``interrupt()``.

« chat » est accepté comme alias historique de « live ».
"""

from __future__ import annotations

DATASET = "dataset"
LIVE = "live"

_ALIASES = {"chat": LIVE, "live": LIVE, "dataset": DATASET, "": DATASET}


def normalize_mode(mode: str) -> str:
    try:
        return _ALIASES[(mode or "").strip().lower()]
    except KeyError:
        raise ValueError(f"mode inconnu : {mode!r} (attendu dataset|live)")


def is_live(mode: str) -> bool:
    return normalize_mode(mode) == LIVE
