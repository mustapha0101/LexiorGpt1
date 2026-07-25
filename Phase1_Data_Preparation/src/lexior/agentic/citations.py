# -*- coding: utf-8 -*-
"""Source unique des identifiants de tribunaux et des motifs de citation.

Neuf déclarations réparties sur huit fichiers portaient cinq listes
DIFFÉRENTES de tribunaux québécois. ``QCTAL`` manquait dans
``result_classifier``, ``validators`` et ``case_law_gate`` : toute décision
du Tribunal administratif du logement — c'est-à-dire la quasi-totalité du
contentieux locatif — était classée ``wrong_document_type`` et n'entrait
jamais dans une trajectoire.

Une seule liste, ici. Tout module qui reconnaît une citation consomme ces
constantes; ``tests/test_citation_consistency.py`` échoue si une liste
divergente réapparaît quelque part.
"""

from __future__ import annotations

import re

# ── Tribunaux ────────────────────────────────────────────────────────────

QUEBEC_COURT_SCOPES: tuple[str, ...] = (
    "QCCA",    # Cour d'appel du Québec
    "QCCS",    # Cour supérieure
    "QCCQ",    # Cour du Québec
    "QCTAL",   # Tribunal administratif du logement (depuis 2020)
    "QCRDL",   # Régie du logement (avant 2020) — QCRDE était une coquille
    "QCTAT",   # Tribunal administratif du travail
    "QCTDP",   # Tribunal des droits de la personne
    "QCCAI",   # Commission d'accès à l'information
)

FEDERAL_COURT_SCOPES: tuple[str, ...] = (
    "SCC", "CSC",   # Cour suprême du Canada
    "FCA", "CAF",   # Cour d'appel fédérale
    "FC", "CF",     # Cour fédérale
    "TCC",          # Cour canadienne de l'impôt
)

ALL_COURT_SCOPES: tuple[str, ...] = QUEBEC_COURT_SCOPES + FEDERAL_COURT_SCOPES


def _alternation(scopes: tuple[str, ...]) -> str:
    """Alternance regex, les identifiants longs d'abord.

    Sans ce tri, ``FC`` capturerait le préfixe de ``FCA`` et tronquerait la
    citation.
    """
    return "|".join(sorted(scopes, key=len, reverse=True))


QUEBEC_SCOPE_PATTERN = _alternation(QUEBEC_COURT_SCOPES)
FEDERAL_SCOPE_PATTERN = _alternation(FEDERAL_COURT_SCOPES)
COURT_SCOPE_PATTERN = _alternation(ALL_COURT_SCOPES)

# ── Motifs de citation ───────────────────────────────────────────────────

# « 2021 QCTAL 7020 », tous tribunaux confondus.
CASE_CITATION_RE = re.compile(
    rf"\b\d{{4}}\s+(?:{COURT_SCOPE_PATTERN})\s+\d+\b")

# Variantes à groupes capturants (année, tribunal, numéro).
QUEBEC_CITATION_RE = re.compile(
    rf"\b(\d{{4}})\s+({QUEBEC_SCOPE_PATTERN})\s+(\d+)\b")
FEDERAL_CITATION_RE = re.compile(
    rf"\b(\d{{4}})\s+({FEDERAL_SCOPE_PATTERN})\s+(\d+)\b")

# « Untel c. Unetelle » — exclut les désignations de lois (« RLRQ c. P-40 »),
# qui ont la même forme superficielle.
_STATUTE_PREFIXES = ("RLRQ", "LRQ", "CQLR", "RSC", "LRC", "SC", "LC", "RSQ")
CASE_NAME_RE = re.compile(
    r"(?<![\w.])(?!(?:" + "|".join(_STATUTE_PREFIXES) + r")\b)"
    r"[A-ZÀ-Ÿ][\w'-]+\s+c\.\s+[A-ZÀ-Ÿ][\w'-]+")


def is_quebec_scope(scope: str) -> bool:
    return (scope or "").upper() in QUEBEC_COURT_SCOPES


def is_federal_scope(scope: str) -> bool:
    return (scope or "").upper() in FEDERAL_COURT_SCOPES


def find_case_citation(text: str) -> str:
    """Première citation de décision du texte, chaîne vide si aucune."""
    match = CASE_CITATION_RE.search(text or "")
    return match.group(0) if match else ""
