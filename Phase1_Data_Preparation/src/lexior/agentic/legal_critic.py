# -*- coding: utf-8 -*-
"""Critique juridique structurée — scoring 9 dimensions."""

from __future__ import annotations

import json
import unicodedata

from .critic_context import bounded_tool_history
from .critic_profiles import LEGAL_RUBRICS, get_profile
from .prompts import LEGAL_CRITIC_SYSTEM
from .schemas import CriticResult, MultiDimensionalScore, ResearchState


LEGAL_CRITIC_SCOPE_POLICY = """
Politique de portée obligatoire :
- Pour `non_legal`, une réponse polie, brève et sans outil DOIT
  être acceptée si elle n'invente aucune règle juridique.
- Pour `exact_text_retrieval`, le texte de l'article demandé est une source
  suffisante. N'exige JAMAIS d'autres articles ou jurisprudences.
  Accepte seulement si la réponse finale est exactement le texte officiel
  récupéré, mot pour mot et sans introduction ni paraphrase.
- Pour `article_explanation`, le texte officiel récupéré doit apparaître
  intégralement et mot pour mot avant une explication séparée.
- L'exigence de reproduction mot pour mot s'applique UNIQUEMENT à
  exact_text_retrieval. Ne l'impose jamais à un cas concret ou à une
  recherche thématique.
- Une réponse prudente indiquant qu'un outil a échoué peut être acceptée.
- N'évalue pas la qualité du scénario. Évalue seulement la fidélité de la
  réponse par rapport à la question et aux preuves récupérées.
- L'absence de jurisprudence ne doit PAS provoquer un rejet si :
  (a) la jurisprudence est optionnelle pour ce type de demande ;
  (b) la réponse mentionne explicitement qu'aucune décision applicable n'a
      été trouvée ;
  (c) la réponse est fondée sur la législation récupérée.
- Ne rejette PAS pour le style, la concision ou le registre. Évalue
  uniquement la fidélité aux sources.
"""


class LegalCritic:
    def __init__(self, client=None, offline: bool = False):
        self.client = client
        self.offline = offline

    def evaluate(self, state: ResearchState, answer: str) -> CriticResult:
        profile = get_profile(state.scenario.request_type)
        if self.offline:
            hard_failures = []
            if not answer.strip():
                hard_failures.append("réponse vide")
            if any(not o.ok for o in state.tool_history) and "ne vais pas fabriquer" not in answer:
                hard_failures.append("erreur MCP non reconnue")
            issues = list(hard_failures)
            score = 1.0 if not issues else 0.0
            dim = MultiDimensionalScore(
                grounding_score=score,
                legal_accuracy_score=score,
                answer_quality_score=score,
                request_classification_score=1.0,
                jurisdiction_score=1.0,
                clarification_score=1.0,
                tool_selection_score=1.0,
                search_quality_score=1.0,
                result_validation_score=1.0,
            )
            return CriticResult(
                critic="legal", accepted=not hard_failures, score=score,
                issues=issues, repair_instructions=issues,
                hard_failures=hard_failures, soft_issues=[],
                critic_profile=profile.value,
                dimensional_scores=dim,
            )

        rubric = LEGAL_RUBRICS.get(profile, "")
        rubric_block = f"\nProfil : {profile.value}. {rubric}\n" if rubric else ""
        user_messages = [m.content for m in state.messages
                         if m.role.value == "user"]
        raw = self.client.complete_json("legal_critic", [
            {"role": "system",
             "content": LEGAL_CRITIC_SYSTEM + LEGAL_CRITIC_SCOPE_POLICY + rubric_block},
            {"role": "user", "content": json.dumps({
                "request_type": state.scenario.request_type,
                "question": state.scenario.user_query,
                "messages_utilisateur": user_messages,
                "tool_history": bounded_tool_history(state),
                "answer": answer,
            }, ensure_ascii=False)},
        ], temperature=0.0)

        dim = MultiDimensionalScore(
            grounding_score=float(raw.get("grounding", 0.0)),
            legal_accuracy_score=float(raw.get("legal_accuracy", 0.0)),
            answer_quality_score=float(raw.get("answer_quality", 0.0)),
            labels=list(raw.get("labels", [])),
        )

        flat_score = (dim.grounding_score + dim.legal_accuracy_score
                      + dim.answer_quality_score) / 3.0
        raw["critic"] = "legal"
        raw["score"] = flat_score
        raw["accepted"] = flat_score >= 0.70 and not any(
            lbl in dim.labels for lbl in (
                "unsupported_claim", "fabricated_case_law_pattern",
                "unsupported_deadline", "wrong_jurisdiction",
            )
        )
        raw.setdefault("hard_failures", [])
        raw.setdefault("soft_issues", [])
        raw.setdefault("critic_profile", profile.value)
        raw["dimensional_scores"] = dim.model_dump()

        result = CriticResult.model_validate(raw)
        return self._apply_scope_policy(state, answer, result)

    @staticmethod
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        return "".join(
            c for c in normalized if not unicodedata.combining(c)
        ).casefold()

    @classmethod
    def _apply_scope_policy(cls, state: ResearchState, answer: str,
                            result: CriticResult) -> CriticResult:
        request_type = state.scenario.request_type
        if request_type == "exact_text_retrieval":
            return result
        exact_markers = (
            "mot pour mot", "reproduction exacte", "reproduction integrale",
            "reproduit integralement", "reproduits integralement",
            "pas reproduit", "ne sont pas reproduits",
            "ne respecte pas l'exigence de reproduction",
        )

        def invalid_exact_requirement(value: str) -> bool:
            folded = cls._fold(value)
            return any(marker in folded for marker in exact_markers)

        filtered_issues = [
            issue for issue in result.issues
            if not invalid_exact_requirement(issue)
        ]
        filtered_repairs = [
            i for i in result.repair_instructions
            if not invalid_exact_requirement(i)
        ]
        blockers = (filtered_issues or result.unsupported_claims
                    or result.missing_sources)
        return result.model_copy(update={
            "issues": filtered_issues,
            "repair_instructions": filtered_repairs,
            "accepted": result.accepted if blockers else True,
            "score": result.score if blockers else max(result.score, 0.75),
        })
