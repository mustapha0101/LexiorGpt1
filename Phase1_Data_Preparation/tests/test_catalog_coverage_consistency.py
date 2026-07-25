# -*- coding: utf-8 -*-
"""Le catalogue et la couverture déclarée doivent raconter la même histoire.

``representativeResponse`` enregistre ce qu'un outil a RÉELLEMENT retourné
au moment du catalogage; ``TOOL_COVERAGE.document_types`` déclare ce qu'il
est censé couvrir. Jusqu'ici, rien ne lisait le premier : la divergence de
``search_quebec_jurisprudence`` (déclaré « décisions », retournant des
lois) est restée invisible pendant tout ce temps.

Règle appliquée ici :

    outil disponible   → tout type observé doit être déclaré ;
    outil indisponible → la contradiction est tolérée, mais
                         ``availability_reason`` doit porter une
                         constatation DATÉE.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lexior.services.tool_coverage import TOOL_COVERAGE

CATALOG_PATH = (Path(__file__).resolve().parents[2] / "docs"
                / "mcp_tools_catalog.json")

_DECISION_CITATION = re.compile(
    r"\b\d{4}\s+(?:QCCA|QCCS|QCCQ|QCTAL|QCTAT|QCTDP|SCC|CSC|FCA|CAF|FC|CF|TCC)"
    r"\s+\d+\b")
_DECISION_URL = re.compile(
    r"canlii\.org/[^\s\"]*?/(?:qcca|qccs|qccq|qctal|scc|fca|fc)/doc/"
    r"|citoyens\.soquij\.qc\.ca", re.IGNORECASE)
_ARTICLE = re.compile(r"^\s*(?:###\s*)?Article\s+\d", re.MULTILINE)
# Sensible à la casse et exigeant : « règlement des différends » désigne un
# mode de résolution, pas un texte réglementaire.
_REGULATION = re.compile(
    r"\bRèglement sur\b|\b[A-Z]{1,3}-\d+(?:\.\d+)?,\s*r\.\s*\d")
_STATUTE = re.compile(r"\bLoi sur\b|\bRLRQ c\b|\bCode (?:civil|de procédure)\b")

_DATED_FINDING = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def observed_document_types(text: str) -> set[str]:
    """Types de documents que la réponse représentative contient vraiment."""
    observed: set[str] = set()
    if _DECISION_CITATION.search(text) or _DECISION_URL.search(text):
        observed.add("court_decision")
    if _REGULATION.search(text):
        observed.add("regulation")
    elif _ARTICLE.search(text):
        # Un article de règlement reste du règlement : on ne compte
        # « statute_article » que hors contexte réglementaire.
        observed.add("statute_article")
    if _STATUTE.search(text) and not observed & {"statute_article",
                                                 "regulation"}:
        observed.add("statute")
    return observed


def catalog_entries() -> list[tuple[str, str]]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = []
    for tool in catalog["tools"]:
        response = (tool.get("responseStructure") or {}).get(
            "representativeResponse")
        if isinstance(response, (dict, list)):
            response = json.dumps(response, ensure_ascii=False)
        if response and str(response).strip():
            entries.append((tool["canonicalName"], str(response)))
    return entries


def test_the_catalog_actually_carries_representative_responses():
    entries = catalog_entries()
    assert len(entries) >= 10, (
        "sans réponse représentative, la cohérence n'est pas vérifiable")


@pytest.mark.parametrize("tool_name,response", catalog_entries(),
                         ids=[name for name, _ in catalog_entries()])
def test_representative_response_matches_declared_coverage(
        tool_name, response):
    entry = TOOL_COVERAGE.get(tool_name)
    if entry is None:
        pytest.skip(f"{tool_name} n'a pas d'entrée de couverture")

    observed = observed_document_types(response)
    if not observed:
        pytest.skip(f"{tool_name} : aucun type de document identifiable")

    declared = set(entry.document_types)
    # « statute_article » satisfait une déclaration « statute » et
    # inversement : c'est la même famille documentaire.
    equivalents = {"statute_article": {"statute"}, "statute": {"statute_article"}}
    unexpected = {
        kind for kind in observed
        if kind not in declared and not (equivalents.get(kind, set()) & declared)
    }
    if not unexpected:
        return

    assert entry.availability_status != "available", (
        f"{tool_name} est déclaré disponible mais sa réponse représentative "
        f"contient {sorted(unexpected)}, absent de document_types "
        f"{sorted(declared)}")
    assert _DATED_FINDING.search(entry.availability_reason), (
        f"{tool_name} contredit sa couverture déclarée sans constatation "
        f"datée dans availability_reason — la prochaine personne refera "
        f"l'enquête. Reçu : {entry.availability_reason!r}")


def test_the_jurisprudence_tool_is_active_and_documented():
    """Verrou sur l'état d'un outil dont la disponibilité a déjà basculé.

    Il a été désactivé sur un diagnostic erroné — l'échec venait de QCTAL
    manquant côté client, pas du serveur. La constatation datée dans
    ``availability_reason`` existe pour que ce diagnostic ne se reperde pas.
    """
    entry = TOOL_COVERAGE["search_quebec_jurisprudence"]

    assert entry.availability_status == "available"
    assert entry.is_available("live") and entry.is_available("dataset")
    assert _DATED_FINDING.search(entry.availability_reason), (
        "toute bascule de disponibilité doit porter une constatation datée")
    assert "court_decision" in entry.document_types
    assert set(entry.court_scopes) >= {"QCTAL", "QCCA", "QCCS", "QCCQ"}
