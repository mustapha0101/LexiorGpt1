# -*- coding: utf-8 -*-
"""Service de clarification — même logique, deux livraisons.

La QUESTION est construite de la même façon dans les deux modes. Seule
la LIVRAISON diffère (différence de mode assumée, spec §3) :

    dataset — la réponse synthétique du scénario est consommée; sans
              réponse synthétique, la trajectoire se termine sur la
              question (exemple d'entraînement « clarification »).
    live    — le graphe s'interrompt (``interrupt()``); la vraie réponse
              arrive au tour suivant via ``Command(resume=...)``.
"""

from __future__ import annotations

from typing import Optional

from agentic_generation.schemas import PlannerDecision, ScenarioSpec

DEFAULT_QUESTION = "Pouvez-vous préciser?"


class ClarificationService:
    @staticmethod
    def build_question(decision: PlannerDecision,
                       missing_facts: Optional[list[str]] = None) -> str:
        if decision.clarification_question:
            return decision.clarification_question
        if missing_facts:
            facts = ", ".join(missing_facts[:3])
            return (f"Pouvez-vous préciser : {facts}? Ces éléments "
                    "déterminent la règle applicable.")
        return DEFAULT_QUESTION

    @staticmethod
    def synthetic_answer(scenario: ScenarioSpec) -> Optional[str]:
        return scenario.effective_clarification_answer
