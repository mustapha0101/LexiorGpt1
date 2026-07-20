# -*- coding: utf-8 -*-
"""Critique de la politique de recherche agentique."""

from __future__ import annotations

import json
import unicodedata

from .critic_context import bounded_tool_history
from .schemas import CriticResult, ResearchState


AGENTIC_CRITIC_SYSTEM = """Évalue le routage, l'ordre, les arguments, les
clarifications, les appels inutiles, les boucles et la condition d'arrêt.
Réponds UNIQUEMENT avec cet objet JSON exact, sans wrapper `CriticResult` et
sans markdown :
{
  "accepted": true,
  "score": 0.0,
  "issues": [],
  "unsupported_claims": [],
  "missing_sources": [],
  "repair_instructions": []
}
Le score est compris entre 0 et 1. `accepted` doit être true seulement si le
score est au moins 0.70 et qu'aucun problème bloquant n'existe. Si
`accepted` est false, explique au moins un motif dans `issues` et donne des
instructions de réparation concrètes."""

AGENTIC_CRITIC_SYSTEM += """

Politique de portée obligatoire :
- Évalue le comportement de l'assistant, jamais la qualité ou la richesse du
  scénario fabriqué par le Scenario Generator.
- Ne pénalise pas l'absence d'un fait si l'assistant a correctement demandé
  une clarification ou a formulé une limite prudente.
- Pour `question_non_juridique`, la bonne route est une réponse directe sans
  aucun outil.
- Pour une demande d'article précis, récupérer exactement cet article puis
  arrêter est optimal; toute recherche supplémentaire est inutile.
- Les étapes marquées optionnelles dans `expected_route` ne sont pas
  obligatoires. N'exige aucune source hors de cette route.
- Compare l'ordre réellement suivi à `expected_route`. N'invente jamais un
  ordre différent de celui déclaré dans ce scénario et ne pénalise pas une
  route qui respecte déjà l'ordre déclaré.
- Le champ `messages` contient la conversation complète. Si
  `clarification_asked=true`, il est interdit d'affirmer qu'aucune
  clarification n'a été demandée. Si la route n'exige pas de clarification,
  il est interdit d'en réclamer une après coup.
- N'évalue ni le style, ni la concision, ni le niveau de détail de la réponse
  finale : ces critères appartiennent au critique juridique. Une préférence
  rédactionnelle ne peut jamais faire échouer le comportement agentique.
- Les résultats des outils `semantic_search_*` sont des candidats, pas une
  liste de sources obligatoires. N'exige jamais de récupérer tous les
  candidats; évalue seulement la route déclarée et les textes officiels
  effectivement retenus.
"""


class AgenticCritic:
    def __init__(self, client=None, offline: bool = False):
        self.client = client
        self.offline = offline

    def evaluate(self, state: ResearchState, answer: str) -> CriticResult:
        if self.offline:
            issues = []
            sequence = [o.tool_name for o in state.tool_history]
            required = state.scenario.expected_route.required_tools()
            if any(tool not in sequence for tool in required):
                issues.append("outil requis manquant")
            if len(sequence) > state.max_tool_calls:
                issues.append("max_tool_calls dépassé")
            if len(sequence) != len(set((o.tool_name, json.dumps(o.arguments, sort_keys=True)) for o in state.tool_history)):
                issues.append("boucle d’appels")
            score = 1.0 if not issues else 0.0
            return CriticResult(critic="agentic", accepted=not issues, score=score,
                                issues=issues, repair_instructions=issues)
        clarification_asked = any(
            message.role.value == "assistant" and message.content.rstrip().endswith("?")
            for message in state.messages
        )
        raw = self.client.complete_json("agentic_critic", [
            {"role": "system", "content": AGENTIC_CRITIC_SYSTEM},
            {"role": "user", "content": json.dumps({"scenario": state.scenario.model_dump(mode="json"),
                                                     "messages": [
                                                         message.model_dump(mode="json")
                                                         for message in state.messages
                                                     ],
                                                     "clarification_asked": clarification_asked,
                                                     "tool_history": bounded_tool_history(state),
                                                     "answer": answer}, ensure_ascii=False)},
        ], temperature=0.0)
        raw["critic"] = "agentic"
        result = CriticResult.model_validate(raw)
        return self._apply_scope_policy(state, clarification_asked, result)

    @staticmethod
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        return "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        ).casefold()

    @classmethod
    def _apply_scope_policy(cls, state: ResearchState, clarification_asked: bool,
                            result: CriticResult) -> CriticResult:
        clarification_required = state.scenario.expected_route.requires_clarification
        clarification_missing = clarification_required and not clarification_asked

        def invalid_clarification_complaint(value: str) -> bool:
            folded = cls._fold(value)
            mentions_clarification = "clarif" in folded or "demander des precisions" in folded
            return mentions_clarification and not clarification_missing

        def answer_style_complaint(value: str) -> bool:
            folded = cls._fold(value)
            style_markers = (
                "reponse plus directe", "reponse plus concise",
                "directement repondu", "actions immediates", "style de reponse",
            )
            return any(marker in folded for marker in style_markers)

        def exhaustive_candidate_complaint(value: str) -> bool:
            folded = cls._fold(value)
            markers = (
                "tous les articles pertinents", "recuperer les articles",
                "chaque article candidat", "tous les candidats",
            )
            return any(marker in folded for marker in markers)

        filtered_issues = [
            issue for issue in result.issues
            if not invalid_clarification_complaint(issue)
            and not answer_style_complaint(issue)
            and not exhaustive_candidate_complaint(issue)
        ]
        filtered_repairs = [
            instruction for instruction in result.repair_instructions
            if not invalid_clarification_complaint(instruction)
            and not answer_style_complaint(instruction)
            and not exhaustive_candidate_complaint(instruction)
        ]
        blockers = filtered_issues or result.unsupported_claims or result.missing_sources
        return result.model_copy(update={
            "issues": filtered_issues,
            "repair_instructions": filtered_repairs,
            "accepted": result.accepted if blockers else True,
            "score": result.score if blockers else max(result.score, 0.75),
        })
