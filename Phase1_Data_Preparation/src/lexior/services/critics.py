# -*- coding: utf-8 -*-
"""Service critics — évaluation juridique et agentique, deux modes.

Regroupe ``LegalCritic`` + ``AgenticCritic`` avec les règles partagées :
``no_critics`` et le court-circuit déterministe des types « texte
officiel précis » (la fidélité mot à mot est vérifiée par les
validateurs; un juge LLM serait moins fiable et plus coûteux).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentic_generation.agentic_critic import AgenticCritic
from agentic_generation.legal_critic import LegalCritic
from agentic_generation.schemas import CriticResult, ResearchState
from agentic_generation.trajectory_agent import PRECISE_ARTICLE_TYPES


@dataclass
class CriticsOutcome:
    legal: Optional[CriticResult] = None
    agentic: Optional[CriticResult] = None
    skipped_reason: str = ""

    def failing(self, legal_min: float, agentic_min: float) -> list[str]:
        """Instructions de réparation issues des critiques en échec."""
        instructions: list[str] = []
        if self.legal and (not self.legal.accepted
                           or self.legal.score < legal_min):
            instructions.extend(
                self.legal.repair_instructions or self.legal.issues
                or ["Rendre la réponse fidèle et suffisante."])
        if self.agentic and (not self.agentic.accepted
                             or self.agentic.score < agentic_min):
            instructions.extend(
                self.agentic.repair_instructions or self.agentic.issues
                or ["Corriger sans ajouter de source absente."])
        return instructions


class CriticsService:
    def __init__(self, legal_critic: LegalCritic,
                 agentic_critic: AgenticCritic,
                 no_critics: bool = False):
        self.legal_critic = legal_critic
        self.agentic_critic = agentic_critic
        self.no_critics = no_critics

    def evaluate(self, state: ResearchState, answer: str) -> CriticsOutcome:
        if self.no_critics:
            return CriticsOutcome(skipped_reason="no_critics")
        if state.scenario.request_type in PRECISE_ARTICLE_TYPES:
            return CriticsOutcome(
                legal=CriticResult(critic="legal", accepted=True, score=1.0),
                agentic=CriticResult(critic="agentic", accepted=True,
                                     score=1.0),
                skipped_reason="controles deterministes (texte officiel)",
            )
        return CriticsOutcome(
            legal=self.legal_critic.evaluate(state, answer),
            agentic=self.agentic_critic.evaluate(state, answer),
        )
