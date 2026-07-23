# -*- coding: utf-8 -*-
"""
Vérificateur déterministe des réponses MCP.

Contrôle silencieux appliqué APRÈS l'exécution MCP et AVANT l'ajout au
tool_history. Les vérifications sont :

  1. Droit fédéral uniquement dans les résultats A2AJ (pas de provincial).
  2. Aucun article abrogé, omis ou épuisé (syntaxe fédérale et québécoise).
  3. Aucune section vide ou trop courte pour être normative.

Le vérificateur ne produit aucune sortie dans les données d'entraînement.
C'est un garde-barrière interne, au même titre que les critics.
"""

from __future__ import annotations

import json
import re

from .schemas import ToolObservation

# ---------------------------------------------------------------------------
# Patterns — sous-ensemble de a2aj_cleaner / ccq_cleaner
# ---------------------------------------------------------------------------

# Fédéral : souche entre crochets  (a2aj_cleaner._STUB_RE / _REPEAL_RE)
_FED_STUB_RE = re.compile(r"^\s*\[(?P<body>[^\]]{0,160})\]\s*$")
_FED_REPEAL_RE = re.compile(r"\[\s*(?:Abrog|Repealed)", re.IGNORECASE)

# Québécois : souche entre parenthèses  (ccq_cleaner._STUB_RE)
_QC_STUB_RE = re.compile(
    r"^\s*\(\s*(?P<body>[^)]*?)\s*\)\s*\.?\s*$", re.IGNORECASE)

# Juridictions fédérales acceptées dans A2AJ
FEDERAL_DATASETS = {
    "LEGISLATION-FED", "REGULATIONS-FED",
    "SCC", "FCA", "FC", "TAX",
}

MIN_CONTENT_CHARS = 30


# ---------------------------------------------------------------------------
# Classification légère
# ---------------------------------------------------------------------------

def _is_federal_stub_repealed(line: str) -> bool:
    m = _FED_STUB_RE.match((line or "").strip())
    if not m:
        return False
    body = m.group("body").strip().lower()
    return bool(re.match(r"(?:abrog[ée]e?s?|repealed)\b", body))


def _is_qc_unusable(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) < MIN_CONTENT_CHARS:
        return True
    m = _QC_STUB_RE.match(t)
    if not m:
        return False
    body = m.group("body").strip().lower()
    return body.startswith(("abrog", "omis", "modification"))


# ---------------------------------------------------------------------------
# Vérifications par outil
# ---------------------------------------------------------------------------

def _check_search_legal_documents(obs: ToolObservation) -> tuple[ToolObservation, list[str]]:
    issues: list[str] = []
    try:
        data = json.loads(obs.normalized_response)
    except (ValueError, TypeError):
        return obs, issues
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return obs, issues

    federal, provincial_names = [], []
    for result in data["results"]:
        if not isinstance(result, dict):
            continue
        dataset = result.get("dataset", "")
        if not dataset or dataset in FEDERAL_DATASETS:
            federal.append(result)
        else:
            provincial_names.append(dataset)

    if not provincial_names:
        return obs, issues

    issues.append(
        f"filtré {len(provincial_names)} résultat(s) non fédéral(aux) : "
        f"{', '.join(sorted(set(provincial_names)))}")

    if not federal:
        obs = obs.model_copy(update={
            "ok": False,
            "error": "aucun résultat fédéral — tous les résultats étaient provinciaux",
            "normalized_response": json.dumps(
                {"error": "aucun résultat fédéral",
                 "message": "La recherche n'a retourné que des résultats provinciaux, "
                            "filtrés car seul le droit fédéral est attendu."},
                ensure_ascii=False),
            "content_hash": "",
        })
        obs.finalize_hash()
        issues.append("FATAL : aucun résultat fédéral restant")
        return obs, issues

    obs = obs.model_copy(update={
        "normalized_response": json.dumps(
            {"results": federal}, ensure_ascii=False, sort_keys=True),
        "content_hash": "",
    })
    obs.finalize_hash()
    return obs, issues


def _check_fetch_document(obs: ToolObservation) -> tuple[ToolObservation, list[str]]:
    issues: list[str] = []
    text = obs.normalized_response.strip()

    if len(text) < MIN_CONTENT_CHARS:
        obs = obs.model_copy(update={
            "ok": False,
            "error": "document fédéral vide",
            "normalized_response": json.dumps(
                {"error": "document vide",
                 "message": "Le document récupéré est vide ou trop court."},
                ensure_ascii=False),
            "content_hash": "",
        })
        obs.finalize_hash()
        issues.append("FATAL : document fédéral vide ou trop court")
        return obs, issues

    if _FED_REPEAL_RE.search(text[:400]):
        lines = [line for line in text.split("\n") if line.strip()]
        repealed = sum(1 for line in lines if _is_federal_stub_repealed(line))
        if repealed and repealed >= len(lines):
            obs = obs.model_copy(update={
                "ok": False,
                "error": "document fédéral entièrement abrogé",
                "normalized_response": json.dumps(
                    {"error": "document abrogé",
                     "message": "Le document récupéré est entièrement abrogé."},
                    ensure_ascii=False),
                "content_hash": "",
            })
            obs.finalize_hash()
            issues.append("FATAL : document fédéral entièrement abrogé")
        elif repealed:
            issues.append(f"{repealed} section(s) abrogée(s) dans le document")

    return obs, issues


def _check_qc_article(obs: ToolObservation) -> tuple[ToolObservation, list[str]]:
    issues: list[str] = []
    text = obs.normalized_response.strip()

    if not text:
        obs = obs.model_copy(update={
            "ok": False,
            "error": "article vide",
            "normalized_response": json.dumps(
                {"error": "article vide",
                 "message": "L'article récupéré est vide."},
                ensure_ascii=False),
            "content_hash": "",
        })
        obs.finalize_hash()
        issues.append("FATAL : article québécois vide")
        return obs, issues

    if _is_qc_unusable(text):
        m = _QC_STUB_RE.match(text)
        kind = "abrogé"
        if m:
            body = m.group("body").strip().lower()
            if body.startswith("omis"):
                kind = "omis"
            elif body.startswith("modification"):
                kind = "épuisé (modification intégrée)"
        obs = obs.model_copy(update={
            "ok": False,
            "error": f"article {kind}",
            "normalized_response": json.dumps(
                {"error": f"article {kind}",
                 "message": f"L'article récupéré est {kind} et ne contient "
                            "aucun contenu normatif exploitable."},
                ensure_ascii=False),
            "content_hash": "",
        })
        obs.finalize_hash()
        issues.append(f"FATAL : article québécois {kind}")

    return obs, issues


def _check_qc_search(obs: ToolObservation) -> tuple[ToolObservation, list[str]]:
    issues: list[str] = []
    text = obs.normalized_response.strip()
    if not text or len(text) < MIN_CONTENT_CHARS:
        issues.append("résultat de recherche québécois vide ou trop court")
    return obs, issues


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_CHECKERS = {
    "search_legal_documents": _check_search_legal_documents,
    "fetch_document": _check_fetch_document,
    "get_ccq_articles": _check_qc_article,
    "get_cpc_articles": _check_qc_article,
    "search_ccq_keywords": _check_qc_search,
    "search_cpc_keywords": _check_qc_search,
    "get_quebec_regulation": _check_qc_search,
    "search_quebec_regulations": _check_qc_search,
    "search_quebec_jurisprudence": _check_qc_search,
}


def verify_observation(observation: ToolObservation) -> tuple[ToolObservation, list[str]]:
    """Vérifie une réponse MCP.

    Retourne (observation éventuellement nettoyée, liste de problèmes).
    Les problèmes préfixés « FATAL » indiquent que l'observation a été
    convertie en erreur (ok=False).
    """
    if not observation.ok:
        return observation, []
    checker = _CHECKERS.get(observation.tool_name)
    if checker is None:
        return observation, []
    return checker(observation)
