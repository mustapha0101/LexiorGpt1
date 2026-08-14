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
        if decision.clarification_scope == "jurisdiction":
            return ("Dans quelle province, quel territoire ou quel pays la "
                    "situation se déroule-t-elle? Le droit applicable peut "
                    "changer selon ce lieu.")
        if decision.clarification_scope == "legal_regime":
            return ("Quel est le domaine ou le secteur précis concerné? Par "
                    "exemple : emploi provincial, emploi fédéral, logement "
                    "ou vente.")
        if missing_facts:
            return ("Pouvez-vous préciser ces faits? Ils ne changent pas "
                    "nécessairement la règle générale, mais ils peuvent "
                    "modifier son application à votre situation.")
        return DEFAULT_QUESTION

    @staticmethod
    def synthetic_answer(scenario: ScenarioSpec) -> Optional[str]:
        return scenario.effective_clarification_answer
