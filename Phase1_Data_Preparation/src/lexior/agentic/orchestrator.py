# -*- coding: utf-8 -*-
"""Façade de compatibilité — DÉPRÉCIÉ.

L'ancienne machine à états (boucle ``while``) a été supprimée. Le
moteur unique est le graphe LangGraph central
(``lexior.agent_graph``) : ce module ne contient AUCUNE boucle
d'orchestration, aucun routage, aucune règle de critique, de
réparation ou d'acceptation — uniquement une délégation vers
``GraphRunner.run_dataset``.

Utiliser directement ::

    from lexior.agent_graph import GraphRunner
    runner.run_dataset(scenario)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional

from .agentic_critic import AgenticCritic
from .config import AgenticConfig
from .legal_critic import LegalCritic
from .mcp_executor import MCPExecutor
from .planner_agent import PlannerAgent
from .schemas import RejectionRecord, ScenarioSpec, TrainingTrajectory
from .tool_catalog import ToolCatalog
from .trajectory_agent import TrajectoryAgent
from .validators import ValidationResult


@dataclass
class OrchestrationResult:
    accepted: bool
    trajectory: Optional[TrainingTrajectory] = None
    rejection: Optional[RejectionRecord] = None
    validation: Optional[ValidationResult] = None


class AgenticOrchestrator:
    """Enveloppe mince au-dessus du graphe central (mode dataset).

    Conserve la signature historique; les agents injectés (y compris
    les doubles de test) sont enveloppés dans les services partagés du
    graphe. Aucune logique d'exécution ici.
    """

    def __init__(self, config: AgenticConfig, catalog: ToolCatalog,
                 planner: PlannerAgent, executor: MCPExecutor,
                 trajectory_agent: TrajectoryAgent, legal_critic: LegalCritic,
                 agentic_critic: AgenticCritic,
                 progress: Optional[Callable[[str], None]] = None):
        from lexior.agent_graph import GraphRunner, build_context
        from lexior.services import (
            AnswerGenerationService,
            ClarificationService,
            CriticsService,
            DatasetExportService,
            JurisdictionService,
            LegalResearchService,
            LexiorServices,
            PlannerService,
            RepairService,
            ResultVerificationService,
            ToolExecutionService,
            ValidationService,
        )

        warnings.warn(
            "AgenticOrchestrator est déprécié : utiliser "
            "lexior.agent_graph.GraphRunner.run_dataset",
            DeprecationWarning, stacklevel=2)

        self.config = config
        self.catalog = catalog
        self.progress = progress or (lambda _message: None)

        answers = AnswerGenerationService.from_writer(trajectory_agent)
        services = LexiorServices(
            config=config,
            catalog=catalog,
            planner=PlannerService.from_agent(planner, catalog),
            tools=ToolExecutionService(executor),
            verification=ResultVerificationService(),
            research=LegalResearchService(),
            jurisdiction=JurisdictionService(),
            clarification=ClarificationService(),
            answers=answers,
            critics=CriticsService(legal_critic, agentic_critic,
                                   no_critics=config.no_critics),
            validation=ValidationService(catalog),
            repair=RepairService(
                answers,
                legal_min_score=config.legal_min_score,
                agentic_min_score=config.agentic_min_score,
                max_repairs=config.max_repairs,
            ),
            export=DatasetExportService(None),
        )
        self.graph_runner = GraphRunner(
            build_context(config, catalog, services))

    @property
    def seen_fingerprints(self) -> set[str]:
        return self.graph_runner.context.services.validation.seen_fingerprints

    def run(self, scenario: ScenarioSpec) -> OrchestrationResult:
        result = self.graph_runner.run_dataset(
            scenario, progress=self.progress)
        return OrchestrationResult(
            accepted=result.accepted,
            trajectory=result.trajectory,
            rejection=result.rejection,
            validation=result.validation,
        )
