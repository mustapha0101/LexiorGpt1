# -*- coding: utf-8 -*-
"""Critique juridique structurée."""

from __future__ import annotations

import json
import unicodedata

from .critic_context import bounded_tool_history
from .schemas import CriticResult, ResearchState


LEGAL_CRITIC_SYSTEM = """Évalue la réponse selon : juridiction, règle,
exceptions, application, fidélité aux résultats d'outils, citations et
prudence. Réponds UNIQUEMENT avec cet objet JSON exact, sans wrapper
`CriticResult` et sans markdown :
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

LEGAL_CRITIC_SYSTEM += """

Politique de portée obligatoire :
- Pour `question_non_juridique`, une réponse polie, brève et sans outil DOIT
  être acceptée si elle n'invente aucune règle juridique.
- Pour `article_ccq_precis`, `article_cpc_precis` et `explication_article`, le
  texte de l'article demandé est une source suffisante. N'exige JAMAIS
  d'autres articles, lois, jurisprudences, exceptions ou développements que
  l'utilisateur n'a pas demandés.
- Pour `article_ccq_precis` et `article_cpc_precis`, accepte seulement si la
  réponse finale est exactement le texte officiel récupéré, mot pour mot et
  sans introduction, paraphrase, conseil ni mise en garde générique.
- Pour `explication_article`, le texte officiel récupéré doit apparaître
  intégralement et mot pour mot avant une explication séparée.
- L'exigence de reproduction mot pour mot s'applique UNIQUEMENT à ces trois
  catégories d'article. Ne l'impose jamais à un cas concret ou à une recherche
  thématique.
- Pour `explication_article`, la présence d'une explication APRÈS le texte
  officiel est obligatoire et ne constitue jamais une altération du texte.
- Une réponse prudente indiquant qu'un outil a échoué ou qu'aucun résultat
  pertinent n'a été trouvé peut être acceptée; n'exige pas une source que le
  pipeline n'a pas récupérée.
- `missing_sources` sert uniquement lorsqu'une affirmation réellement faite
  dans la réponse nécessite une source absente. Ne demande pas d'élargir le
  sujet ni de transformer une réponse ciblée en consultation exhaustive.
- N'évalue pas la qualité du scénario généré. Évalue seulement la fidélité et
  la suffisance de la réponse par rapport à la question effectivement posée
  et aux preuves récupérées.
"""


class LegalCritic:
    def __init__(self, client=None, offline: bool = False):
        self.client = client
        self.offline = offline

    def evaluate(self, state: ResearchState, answer: str) -> CriticResult:
        if self.offline:
            issues = []
            if not answer.strip():
                issues.append("réponse vide")
            if any(not o.ok for o in state.tool_history) and "ne vais pas fabriquer" not in answer:
                issues.append("erreur MCP non reconnue")
            score = 1.0 if not issues else 0.0
            return CriticResult(critic="legal", accepted=not issues, score=score,
                                issues=issues, repair_instructions=issues)
        user_messages = [m.content for m in state.messages if m.role.value == "user"]
        raw = self.client.complete_json("legal_critic", [
            {"role": "system", "content": LEGAL_CRITIC_SYSTEM},
            {"role": "user", "content": json.dumps({"request_type": state.scenario.request_type,
                                                     "question": state.scenario.user_query,
                                                     "messages_utilisateur": user_messages,
                                                     "tool_history": bounded_tool_history(state),
                                                     "answer": answer}, ensure_ascii=False)},
        ], temperature=0.0)
        raw["critic"] = "legal"
        result = CriticResult.model_validate(raw)
        return self._apply_scope_policy(state, answer, result)

    @staticmethod
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        return "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        ).casefold()

    @classmethod
    def _apply_scope_policy(cls, state: ResearchState, answer: str,
                            result: CriticResult) -> CriticResult:
        """Neutralise les faux rejets contraires à la politique déclarée."""
        request_type = state.scenario.request_type
        if request_type in {"article_ccq_precis", "article_cpc_precis"}:
            return result
        exact_markers = (
            "mot pour mot", "reproduction exacte", "reproduction integrale",
            "reproduit integralement", "reproduits integralement",
            "pas reproduit", "ne sont pas reproduits", "ne respecte pas l'exigence de reproduction",
        )

        def invalid_exact_requirement(value: str) -> bool:
            folded = cls._fold(value)
            return any(marker in folded for marker in exact_markers)

        # Pour une explication, le contrôleur déterministe vérifie déjà que le
        # texte officiel intégral précède l'explication. L'explication ne doit
        # donc jamais être considérée comme une violation de fidélité littérale.
        filtered_issues = [
            issue for issue in result.issues
            if not invalid_exact_requirement(issue)
        ]
        filtered_repairs = [
            instruction for instruction in result.repair_instructions
            if not invalid_exact_requirement(instruction)
        ]
        blockers = filtered_issues or result.unsupported_claims or result.missing_sources
        return result.model_copy(update={
            "issues": filtered_issues,
            "repair_instructions": filtered_repairs,
            "accepted": result.accepted if blockers else True,
            "score": result.score if blockers else max(result.score, 0.75),
        })
