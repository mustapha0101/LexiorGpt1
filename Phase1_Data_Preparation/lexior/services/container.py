# -*- coding: utf-8 -*-
"""Conteneur des services Lexior + fabrique unique.

Un seul jeu de services est construit par :func:`build_services` et
partagé par les deux modes (dataset et live).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agentic_generation.config import AgenticConfig
from agentic_generation.tool_catalog import ToolCatalog

from .answer_generation import AnswerGenerationService
from .clarification import ClarificationService
from .critics import CriticsService
from .dataset_export import DatasetExportService
from .jurisdiction import JurisdictionService
from .legal_research import LegalResearchService
from .planner import PlannerService
from .repair import RepairService
from .result_verification import ResultVerificationService
from .tool_execution import ToolExecutionService
from .validation import ValidationService


@dataclass
class LexiorServices:
    """Le jeu unique de services injecté dans le graphe central."""

    config: AgenticConfig
    catalog: ToolCatalog
    planner: PlannerService
    tools: ToolExecutionService
    verification: ResultVerificationService
    research: LegalResearchService
    jurisdiction: JurisdictionService
    clarification: ClarificationService
    answers: AnswerGenerationService
    critics: CriticsService
    validation: ValidationService
    repair: RepairService
    export: DatasetExportService = field(
        default_factory=DatasetExportService)


def build_services(
    config: AgenticConfig,
    catalog: ToolCatalog,
    *,
    executor: Any,
    teacher=None,
    critic_client=None,
    storage=None,
) -> LexiorServices:
    """Construit LE jeu de services partagé par les deux modes.

    ``teacher`` peut être ``None`` en mode offline (planner et rédacteur
    scriptés). ``critic_client`` retombe sur ``teacher`` si absent.
    """
    from agentic_generation.agentic_critic import AgenticCritic
    from agentic_generation.legal_critic import LegalCritic

    offline = bool(config.offline)
    critic_client = critic_client or teacher
    answers = AnswerGenerationService(client=teacher, offline=offline)
    return LexiorServices(
        config=config,
        catalog=catalog,
        planner=PlannerService(catalog, client=teacher, offline=offline),
        tools=ToolExecutionService(executor),
        verification=ResultVerificationService(),
        research=LegalResearchService(),
        jurisdiction=JurisdictionService(),
        clarification=ClarificationService(),
        answers=answers,
        critics=CriticsService(
            LegalCritic(critic_client, offline),
            AgenticCritic(critic_client, offline),
            no_critics=config.no_critics,
        ),
        validation=ValidationService(catalog),
        repair=RepairService(
            answers,
            legal_min_score=config.legal_min_score,
            agentic_min_score=config.agentic_min_score,
            max_repairs=config.max_repairs,
        ),
        export=DatasetExportService(storage),
    )
