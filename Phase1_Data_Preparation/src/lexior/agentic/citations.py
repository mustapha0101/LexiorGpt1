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


# ── Citations d'articles ─────────────────────────────────────────────────
#
# UNE seule extraction pour tout le projet. Les variantes locales ne
# reconnaissaient que « article N » au singulier : « les articles 1457 et
# 1465 » ne livrait aucun numéro, ou seulement le premier, et « 1465
# C.c.Q. » aucun. Un numéro non extrait n'est jamais confronté aux sources
# récupérées — c'est exactement par là qu'un article inventé passe.

_CODE_MARKER = (r"c\.?\s?c\.?\s?q|c\.?\s?p\.?\s?c|ccq|cpc|"
                r"code\s+civil|code\s+de\s+proc[ée]dure(?:\s+civile)?")

# « article 1457 », « articles 1457, 1465 et 1470 », « art. 1457 à 1460 »
_ARTICLE_LIST_RE = re.compile(
    r"\b(?:articles?|art\.?)\s*"
    r"(\d{1,4}(?:\.\d+)?"
    r"(?:\s*(?:,|;|\bet\b|\bou\b|\bà\b|\ba\b|-|–)\s*\d{1,4}(?:\.\d+)?)*)",
    re.I,
)

# « 1465 C.c.Q. », « 1465 du Code civil » — numéro nu qualifié par son code.
_BARE_NUMBER_WITH_CODE_RE = re.compile(
    rf"\b(\d{{1,4}}(?:\.\d+)?)\s*(?:,\s*)?(?:du\s+|de\s+la\s+)?(?:{_CODE_MARKER})\b",
    re.I,
)

_NUMBER_RE = re.compile(r"\d{1,4}(?:\.\d+)?")


def extract_article_citations(text: str) -> list[str]:
    """Tous les numéros d'article cités, dans l'ordre d'apparition.

    Couvre le singulier, le pluriel, les énumérations, les intervalles et
    le numéro nu suivi de son code. Les bornes d'un intervalle sont
    retournées telles quelles : élargir « 1457 à 1460 » à tous les numéros
    intermédiaires inventerait des citations que l'auteur n'a pas faites.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(number: str) -> None:
        value = number.strip()
        if value and value not in seen:
            seen.add(value)
            found.append(value)

    for match in _ARTICLE_LIST_RE.finditer(text or ""):
        for number in _NUMBER_RE.findall(match.group(1)):
            add(number)
    for match in _BARE_NUMBER_WITH_CODE_RE.finditer(text or ""):
        add(match.group(1))
    return found


def mentions_article(text: str, number: str) -> bool:
    """Le texte officiel récupéré porte-t-il CE numéro d'article ?

    Contrepartie de ``extract_article_citations`` du côté des sources :
    « Article 1466 », « art. 1466 » et « 1466 C.c.Q. » désignent la même
    disposition et doivent tous compter comme une preuve de récupération.
    """
    if not text or not number:
        return False
    escaped = re.escape(str(number))
    pattern = (rf"\b(?:articles?|art\.?)\s*{escaped}\b"
               rf"|\b{escaped}\s*(?:,\s*)?(?:du\s+|de\s+la\s+)?"
               rf"(?:{_CODE_MARKER})\b")
    return re.search(pattern, text, re.I) is not None


def is_quebec_scope(scope: str) -> bool:
    return (scope or "").upper() in QUEBEC_COURT_SCOPES


def is_federal_scope(scope: str) -> bool:
    return (scope or "").upper() in FEDERAL_COURT_SCOPES


def find_case_citation(text: str) -> str:
    """Première citation de décision du texte, chaîne vide si aucune."""
    match = CASE_CITATION_RE.search(text or "")
    return match.group(0) if match else ""
