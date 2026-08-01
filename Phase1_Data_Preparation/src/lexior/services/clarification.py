# -*- coding: utf-8 -*-
"""Generic clarification service shared by dataset and live modes."""

from __future__ import annotations

from typing import Optional

from lexior.agentic.schemas import PlannerDecision, ScenarioSpec

DEFAULT_QUESTION = "Pouvez-vous preciser?"


class ClarificationService:
    @staticmethod
    def build_question(decision: PlannerDecision,
                       missing_facts: Optional[list[str]] = None) -> str:
        if decision.clarification_question:
            return decision.clarification_question
        if missing_facts:
            return ("Pouvez-vous preciser le fait pertinent pour l'evenement "
                    "decrit? Cet element peut determiner la regle applicable.")
        return DEFAULT_QUESTION

    @staticmethod
    def synthetic_answer(scenario: ScenarioSpec) -> Optional[str]:
        return scenario.effective_clarification_answer
