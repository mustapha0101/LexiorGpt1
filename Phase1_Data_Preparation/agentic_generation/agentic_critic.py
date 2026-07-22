# -*- coding: utf-8 -*-
"""Critique de la politique de recherche agentique — scoring 9 dimensions."""

from __future__ import annotations

import json
import unicodedata

from .critic_context import bounded_tool_history
from .critic_profiles import AGENTIC_RUBRICS, get_profile
from .prompts import AGENTIC_CRITIC_SYSTEM
from .schemas import CriticResult, MultiDimensionalScore, ResearchState


AGENTIC_CRITIC_SCOPE_POLICY = """
Politique de portée obligatoire :
- Évalue le comportement de l'assistant, jamais la qualité ou la richesse du
  scénario fabriqué par le Scenario Generator.
- Ne pénalise pas l'absence d'un fait si l'assistant a correctement demandé
  une clarification ou a formulé une limite prudente.
- Pour `non_legal`, la bonne route est une réponse directe sans aucun outil.
- Pour exact_text_retrieval, récupérer exactement cet article puis arrêter
  est optimal; toute recherche supplémentaire est inutile.
- Les étapes marquées optionnelles dans `expected_route` ne sont pas
  obligatoires. L'absence d'un outil optionnel ne doit JAMAIS provoquer un
  rejet ou réduire le score.
- Compare l'ordre réellement suivi à `expected_route`. N'invente jamais un
  ordre différent de celui déclaré.
- N'évalue ni le style, ni la concision, ni le niveau de détail de la réponse
  finale. Le thinking trop long est un avertissement, pas un motif de rejet.
- Les résultats de `semantic_search_*` sont des candidats, pas une liste de
  sources obligatoires.
- L'absence de jurisprudence ne doit PAS provoquer un rejet si :
  (a) la jurisprudence était optionnelle pour ce type de demande ;
  (b) le modèle reconnaît explicitement qu'aucune décision applicable n'a été
      trouvée ;
  (c) la réponse finale reste fondée sur la législation récupérée.
- Ne rejette PAS pour des raisons de style (registre, concision, formulation).
  Ces critères sont des avertissements, pas des motifs de rejet.
"""


class AgenticCritic:
    def __init__(self, client=None, offline: bool = False):
        self.client = client
        self.offline = offline

    def evaluate(self, state: ResearchState, answer: str) -> CriticResult:
        profile = get_profile(state.scenario.request_type)
        if self.offline:
            hard_failures = []
            sequence = [o.tool_name for o in state.tool_history]
            required = state.scenario.expected_route.required_tools()
            _empty_exempt = set()
            for _st, _ft in [("semantic_search_ccq", "get_ccq_articles"),
                              ("semantic_search_cpc", "get_cpc_articles")]:
                _searches = [o for o in state.tool_history if o.tool_name == _st]
                if len(_searches) >= 2 and all(
                    not o.ok or (o.normalized_response or "").strip() in ("", "[]", "{}")
                    for o in _searches
                ):
                    _empty_exempt.add(_ft)
            if any(tool not in sequence and tool not in _empty_exempt
                   for tool in required):
                hard_failures.append("outil requis manquant")
            if len(sequence) > state.max_tool_calls:
                hard_failures.append("max_tool_calls dépassé")
            if len(sequence) != len(set(
                (o.tool_name, json.dumps(o.arguments, sort_keys=True))
                for o in state.tool_history
            )):
                hard_failures.append("boucle d'appels")
            issues = list(hard_failures)
            score = 1.0 if not issues else 0.0
            dim = MultiDimensionalScore(
                request_classification_score=score,
                jurisdiction_score=score,
                clarification_score=score,
                tool_selection_score=score,
                search_quality_score=score,
                result_validation_score=score,
                grounding_score=1.0,
                legal_accuracy_score=1.0,
                answer_quality_score=1.0,
            )
            return CriticResult(
                critic="agentic", accepted=not hard_failures, score=score,
                issues=issues, repair_instructions=issues,
                hard_failures=hard_failures, soft_issues=[],
                critic_profile=profile.value,
                dimensional_scores=dim,
            )

        rubric = AGENTIC_RUBRICS.get(profile, "")
        rubric_block = (f"\nProfil : {profile.value}. {rubric}\n"
                        if rubric else "")
        clarification_asked = any(
            m.role.value == "assistant" and m.content.rstrip().endswith("?")
            for m in state.messages
        )
        raw = self.client.complete_json("agentic_critic", [
            {"role": "system",
             "content": AGENTIC_CRITIC_SYSTEM + AGENTIC_CRITIC_SCOPE_POLICY + rubric_block},
            {"role": "user", "content": json.dumps({
                "scenario": state.scenario.model_dump(mode="json"),
                "messages": [m.model_dump(mode="json") for m in state.messages],
                "clarification_asked": clarification_asked,
                "tool_history": bounded_tool_history(state),
                "answer": answer,
            }, ensure_ascii=False)},
        ], temperature=0.0)

        dim = MultiDimensionalScore(
            request_classification_score=float(
                raw.get("request_classification", 0.0)),
            jurisdiction_score=float(raw.get("jurisdiction", 0.0)),
            clarification_score=float(raw.get("clarification", 0.0)),
            tool_selection_score=float(raw.get("tool_selection", 0.0)),
            search_quality_score=float(raw.get("search_quality", 0.0)),
            result_validation_score=float(raw.get("result_validation", 0.0)),
            labels=list(raw.get("labels", [])),
        )

        flat_score = dim.aggregate_score
        raw["critic"] = "agentic"
        raw["score"] = flat_score
        raw["accepted"] = dim.accepted
        raw.setdefault("hard_failures", [])
        raw.setdefault("soft_issues", [])
        raw.setdefault("critic_profile", profile.value)
        raw["dimensional_scores"] = dim.model_dump()

        result = CriticResult.model_validate(raw)
        return self._apply_scope_policy(state, clarification_asked, result)

    @staticmethod
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        return "".join(
            c for c in normalized if not unicodedata.combining(c)
        ).casefold()

    @classmethod
    def _apply_scope_policy(cls, state: ResearchState,
                            clarification_asked: bool,
                            result: CriticResult) -> CriticResult:
        clarification_required = state.scenario.expected_route.requires_clarification
        clarification_missing = clarification_required and not clarification_asked

        def invalid_clarification_complaint(value: str) -> bool:
            folded = cls._fold(value)
            mentions = ("clarif" in folded
                        or "demander des precisions" in folded)
            return mentions and not clarification_missing

        def answer_style_complaint(value: str) -> bool:
            folded = cls._fold(value)
            markers = (
                "reponse plus directe", "reponse plus concise",
                "directement repondu", "actions immediates",
                "style de reponse",
            )
            return any(m in folded for m in markers)

        def exhaustive_candidate_complaint(value: str) -> bool:
            folded = cls._fold(value)
            markers = (
                "tous les articles pertinents", "recuperer les articles",
                "chaque article candidat", "tous les candidats",
            )
            return any(m in folded for m in markers)

        def optional_jurisprudence_complaint(value: str) -> bool:
            folded = cls._fold(value)
            markers = (
                "jurisprudence manquante", "pas de jurisprudence",
                "devrait chercher des decisions", "aucune jurisprudence",
                "recherche de jurisprudence",
            )
            if not any(m in folded for m in markers):
                return False
            from .taxonomy import NO_JURISPRUDENCE, REQUEST_TYPES
            rt = REQUEST_TYPES.get(state.scenario.request_type)
            if state.scenario.request_type in NO_JURISPRUDENCE:
                return True
            if rt:
                for step in rt.expected_route.steps:
                    if (step.tool == "search_quebec_jurisprudence"
                            and step.optional):
                        return True
            return False

        def thinking_complaint(value: str) -> bool:
            folded = cls._fold(value)
            return "thinking" in folded and ("long" in folded or "trop" in folded)

        filtered_issues = [
            issue for issue in result.issues
            if not invalid_clarification_complaint(issue)
            and not answer_style_complaint(issue)
            and not exhaustive_candidate_complaint(issue)
            and not optional_jurisprudence_complaint(issue)
            and not thinking_complaint(issue)
        ]
        filtered_repairs = [
            i for i in result.repair_instructions
            if not invalid_clarification_complaint(i)
            and not answer_style_complaint(i)
            and not exhaustive_candidate_complaint(i)
            and not optional_jurisprudence_complaint(i)
            and not thinking_complaint(i)
        ]

        # Move filtered-out issues to soft_issues
        soft = [
            issue for issue in result.issues
            if issue not in filtered_issues
        ]

        blockers = (filtered_issues or result.unsupported_claims
                    or result.missing_sources)
        return result.model_copy(update={
            "issues": filtered_issues,
            "repair_instructions": filtered_repairs,
            "soft_issues": list(result.soft_issues) + soft,
            "accepted": result.accepted if blockers else True,
            "score": result.score if blockers else max(result.score, 0.75),
        })
