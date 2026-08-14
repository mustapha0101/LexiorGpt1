# -*- coding: utf-8 -*-
"""Generic clarification service shared by dataset and live modes."""

from __future__ import annotations

from typing import Optional

from lexior.agentic.schemas import PlannerDecision, ScenarioSpec

DEFAULT_QUESTION = "Pouvez-vous préciser le point qui vous préoccupe?"


class ClarificationService:
    @staticmethod
    def build_question(decision: PlannerDecision,
                       missing_facts: Optional[list[str]] = None) -> str:
        if decision.clarification_question:
            return decision.clarification_question
        if decision.clarification_scope == "request_intent":
            return "Que souhaitez-vous savoir ou faire?"
        if decision.clarification_scope == "jurisdiction":
            return ("Dans quelle province, quel territoire ou quel pays la "
                    "situation se déroule-t-elle? Le droit applicable peut "
                    "changer selon ce lieu.")
        if decision.clarification_scope == "legal_regime":
            needs_location = "work_location" in (missing_facts or [])
            if needs_location:
                return ("Dans quelle province ou quel territoire travaillez-vous, "
                        "et quelle est l'activité principale de votre employeur?")
            return "Quelle est l'activité principale de votre employeur?"
        if missing_facts:
            return ("Pouvez-vous préciser ces faits? Ils ne changent pas "
                    "nécessairement la règle générale, mais ils peuvent "
                    "modifier son application à votre situation.")
        return DEFAULT_QUESTION

    @staticmethod
    def synthetic_answer(scenario: ScenarioSpec) -> Optional[str]:
        return scenario.effective_clarification_answer
