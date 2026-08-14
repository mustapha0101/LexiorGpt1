# -*- coding: utf-8 -*-
"""Conteneur des services utilisés par la démonstration live."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lexior.agentic.config import AgenticConfig
from lexior.agentic.tool_catalog import ToolCatalog

from .answer_generation import AnswerGenerationService
from .clarification import ClarificationService
from .critics import CriticsService
from .assertion_grounding import AssertionGroundingService
from .jurisdiction import JurisdictionService
from .legal_research import LegalResearchService
from .planner import PlannerService
from .repair import RepairService
from .request_classifier import RequestClassifierService
from .result_verification import ResultVerificationService
from .tool_execution import ToolExecutionService
from .validation import ValidationService


@dataclass
class LexiorServices:
    """Le jeu unique de services injecté dans le graphe central."""

    config: AgenticConfig
    catalog: ToolCatalog
    planner: PlannerService
    request_classifier: RequestClassifierService
    tools: ToolExecutionService
    verification: ResultVerificationService
    research: LegalResearchService
    jurisdiction: JurisdictionService
    clarification: ClarificationService
    answers: AnswerGenerationService
    critics: CriticsService
    validation: ValidationService
    repair: RepairService
    assertion_grounding: AssertionGroundingService = field(
        default_factory=AssertionGroundingService)


def build_services(
    config: AgenticConfig,
    catalog: ToolCatalog,
    *,
    executor: Any,
    teacher=None,
    critic_client=None,
) -> LexiorServices:
    """Construit LE jeu de services partagé par les deux modes.

    ``teacher`` peut être ``None`` en mode offline (planner et rédacteur
    scriptés). ``critic_client`` retombe sur ``teacher`` si absent.
    """
    from lexior.agentic.agentic_critic import AgenticCritic
    from lexior.agentic.legal_critic import LegalCritic

    offline = bool(config.offline)
    critic_client = critic_client or teacher
    answers = AnswerGenerationService(client=teacher, offline=offline)
    return LexiorServices(
        config=config,
        catalog=catalog,
        planner=PlannerService(
            catalog, client=teacher, offline=offline,
            initial_article_fetch_k=config.initial_article_fetch_k,
            article_fetch_batch_size=config.article_fetch_batch_size,
            max_articles_per_issue=config.max_articles_per_issue,
            evidence_first_enabled=config.evidence_first_enabled,
            evidence_first_initial_candidate_count=config.evidence_first_initial_candidate_count,
            evidence_first_initial_fetch_count=config.evidence_first_initial_fetch_count,
            evidence_first_maximum_article_batches=config.evidence_first_maximum_article_batches),
        request_classifier=RequestClassifierService(
            client=teacher, offline=offline,
            minimum_confidence=config.classification_min_confidence),
        tools=ToolExecutionService(executor),
        verification=ResultVerificationService(),
        research=LegalResearchService(),
        jurisdiction=JurisdictionService(),
        clarification=ClarificationService(),
        answers=answers,
        assertion_grounding=AssertionGroundingService(
            client=critic_client, offline=offline),
        critics=CriticsService(
            LegalCritic(critic_client, offline),
            AgenticCritic(critic_client, offline),
            no_critics=config.no_critics,
        ),
        validation=ValidationService(
            catalog,
            near_duplicate_jaccard=config.near_duplicate_jaccard,
            max_thinking_words=config.max_thinking_words,
        ),
        repair=RepairService(
            answers,
            legal_min_score=config.legal_min_score,
            agentic_min_score=config.agentic_min_score,
            max_repairs=config.max_repairs,
        ),
    )
