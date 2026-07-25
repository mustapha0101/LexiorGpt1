# -*- coding: utf-8 -*-
"""Une seule liste de tribunaux, partout.

Neuf déclarations sur huit fichiers portaient cinq listes différentes.
``QCTAL`` manquait dans ``result_classifier``, ``validators`` et
``case_law_gate`` : toute décision du Tribunal administratif du logement
était classée ``wrong_document_type``, donc écartée — alors que c'est
précisément la juridiction du contentieux locatif.

Ce test échoue si une liste divergente réapparaît.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lexior.agentic.case_law_gate import _RE_QC_CITATION
from lexior.agentic.citations import (
    ALL_COURT_SCOPES,
    CASE_CITATION_RE,
    CASE_NAME_RE,
    FEDERAL_COURT_SCOPES,
    QUEBEC_COURT_SCOPES,
    find_case_citation,
)
from lexior.agentic.mcp_executor import CITATION_RE
from lexior.agentic.validators import _CASE_CITATION_RE as VALIDATORS_RE
from lexior.agent_graph.nodes._common import detect_case_reference
from lexior.agent_graph.result_classifier import (
    _CASE_CITATION_RE as CLASSIFIER_RE,
)
from lexior.services.result_verification import _QC_COURT_RE
from lexior.services.tool_coverage import (
    QUEBEC_COURT_SCOPES as COVERAGE_QUEBEC_SCOPES,
)

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "lexior"

# Décisions réellement obtenues du serveur MCP le 2026-07-24.
REAL_DECISIONS = [
    "2021 QCTAL 7020",
    "2023 QCTAL 35455",
    "2019 QCRDL 27403",
    "2011 QCCS 2655",
    "2026 QCCA 866",
    "2020 SCC 5",
]


# ── Une seule source ─────────────────────────────────────────────────────


def test_no_module_declares_its_own_court_list():
    """Aucune alternance de tribunaux codée en dur hors de citations.py."""
    hardcoded = re.compile(r"QC(?:CA|CS|CQ|TAL|TAT|TDP|RDL|RDE|CAI)\s*\|")
    offenders = [
        str(path.relative_to(SOURCE_DIR))
        for path in sorted(SOURCE_DIR.rglob("*.py"))
        if path.name != "citations.py"
        and hardcoded.search(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        f"listes de tribunaux dupliquées dans {offenders} — consommer "
        f"lexior.agentic.citations")


def test_every_consumer_shares_the_same_pattern():
    assert CLASSIFIER_RE is CASE_CITATION_RE
    assert VALIDATORS_RE is CASE_CITATION_RE
    assert tuple(COVERAGE_QUEBEC_SCOPES) == QUEBEC_COURT_SCOPES


def test_the_typo_scope_is_gone():
    """QCRDE n'existe pas : la Régie du logement, c'est QCRDL."""
    assert "QCRDE" not in ALL_COURT_SCOPES
    assert "QCRDL" in QUEBEC_COURT_SCOPES


def test_the_housing_tribunal_is_covered():
    """Le TAL est la juridiction du contentieux locatif : il ne peut pas
    manquer."""
    assert "QCTAL" in QUEBEC_COURT_SCOPES
    assert "QCTAT" in QUEBEC_COURT_SCOPES
    assert "QCCAI" in QUEBEC_COURT_SCOPES


# ── Reconnaissance effective ─────────────────────────────────────────────


@pytest.mark.parametrize("citation", REAL_DECISIONS)
def test_every_real_decision_is_recognised(citation):
    text = f"Locateur c. Locataire, {citation} (CanLII)"

    assert CASE_CITATION_RE.search(text), citation
    assert CLASSIFIER_RE.search(text), citation
    assert find_case_citation(text) == citation
    assert detect_case_reference(text) == citation


@pytest.mark.parametrize("citation", ["2021 QCTAL 7020", "2019 QCRDL 27403"])
def test_a_housing_decision_is_no_longer_a_wrong_document_type(citation):
    """Le symptôme d'origine, verrouillé."""
    from lexior.agentic.schemas import SearchResultStatus
    from lexior.agent_graph.result_classifier import ResultClassifier

    response = (f"Cauchon c. Habitations du Faubourg, {citation} (CanLII)\n"
                "Le tribunal résilie le bail et ordonne l'éviction du "
                "locataire pour non-paiement du loyer.")

    status = ResultClassifier().classify(
        "search_quebec_jurisprudence", response, ok=True)

    assert status == SearchResultStatus.usable


def test_quebec_and_federal_scopes_do_not_overlap():
    assert not set(QUEBEC_COURT_SCOPES) & set(FEDERAL_COURT_SCOPES)


def test_longer_identifiers_win_over_their_prefixes():
    """Sans tri par longueur, « FC » tronquerait « FCA »."""
    match = CASE_CITATION_RE.search("2022 FCA 118")

    assert match.group(0) == "2022 FCA 118"


def test_a_statute_designation_is_not_a_case_name():
    """« RLRQ c. P-40.1 » a la forme d'un nom de cause sans en être un.

    14 des 20 observations classées `usable` à tort passaient par là.
    """
    assert not CASE_NAME_RE.search("Loi sur la protection, RLRQ c. P-40.1")
    assert not CASE_NAME_RE.search("Code de procédure civile, RLRQ c. C-25.01")
    assert CASE_NAME_RE.search("Barreau du Québec c. Richard")


def test_the_mcp_citation_extractor_sees_housing_decisions():
    citations = [m.group(0) for m in CITATION_RE.finditer(
        "Décision 2021 QCTAL 7020 rendue en vertu de RLRQ c CCQ-1991")]

    assert "2021 QCTAL 7020" in citations


def test_the_case_law_gate_accepts_a_housing_citation():
    assert _RE_QC_CITATION.search("2023 QCTAL 35455")
    assert _QC_COURT_RE.search("2023 QCTAL 35455")
