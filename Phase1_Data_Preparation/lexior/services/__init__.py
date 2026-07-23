# -*- coding: utf-8 -*-
"""Couche services de Lexior — logique métier réutilisable, sans routage.

Les services calculent et appellent les modèles; ils ne décident JAMAIS
de l'ordre d'exécution. Le routage appartient exclusivement au graphe
LangGraph (``lexior.agent_graph``).

Chargement PARESSEUX (PEP 562) : ``agentic_generation.planner_agent``
importe ``lexior.services.jurisdiction`` au chargement; un __init__
eager recréerait un cycle d'import. N'ajouter AUCUN import eager ici.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    # modes
    "DATASET": ".modes",
    "LIVE": ".modes",
    "is_live": ".modes",
    "normalize_mode": ".modes",
    # jurisdiction (feuille — importée par agentic_generation.planner_agent)
    "QC_ONLY_TOOLS": ".jurisdiction",
    "QuebecToolsBlocked": ".jurisdiction",
    "detect_jurisdiction_hint": ".jurisdiction",
    "JurisdictionResolution": ".jurisdiction",
    "JurisdictionService": ".jurisdiction",
    # conteneur
    "LexiorServices": ".container",
    "build_services": ".container",
    # services
    "PlannerService": ".planner",
    "ToolExecutionService": ".tool_execution",
    "build_mock_executor": ".tool_execution",
    "build_real_executor": ".tool_execution",
    "ResultVerificationService": ".result_verification",
    "ToolResultAssessment": ".result_verification",
    "LegalResearchService": ".legal_research",
    "ClarificationService": ".clarification",
    "AnswerGenerationService": ".answer_generation",
    "CriticsService": ".critics",
    "CriticsOutcome": ".critics",
    "ValidationService": ".validation",
    "RepairService": ".repair",
    "FailureReport": ".repair",
    "FAILURE_TARGETS": ".repair",
    "DatasetExportService": ".dataset_export",
    # evidence
    "EvidenceLevel": ".evidence",
    "DetailedResultStatus": ".evidence",
    "EvidenceEntry": ".evidence",
    "CoverageGap": ".evidence",
    "AcceptanceBlocker": ".evidence",
    "JurisdictionDimensions": ".evidence",
    # tool coverage
    "TOOL_COVERAGE": ".tool_coverage",
    "ToolCoverageEntry": ".tool_coverage",
    "get_coverage": ".tool_coverage",
    "has_equivalent_coverage": ".tool_coverage",
    "tools_covering_court_scope": ".tool_coverage",
    "QUEBEC_COURT_SCOPES": ".tool_coverage",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
