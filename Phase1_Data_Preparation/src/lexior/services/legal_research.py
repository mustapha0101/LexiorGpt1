# -*- coding: utf-8 -*-
"""Service de recherche juridique — reformulation déterministe.

Quand un résultat est inutilisable (non pertinent, vide, mauvais type de
document), ce service construit le retour correctif transmis au planner
pour la tentative suivante. Aucune décision de routage ici : le graphe
route, ce service formule.
"""

from __future__ import annotations

from agentic_generation.schemas import SearchResultStatus

from .result_verification import ToolResultAssessment


class LegalResearchService:
    @staticmethod
    def build_reformulation_feedback(
            assessment: ToolResultAssessment, attempt: int) -> str:
        status = assessment.search_status
        if status == SearchResultStatus.empty.value:
            return (
                f"La recherche « {assessment.tool_name} » n'a retourné "
                "aucun résultat. Reformule avec des termes plus larges ou "
                "des synonymes, ou change d'outil; ne répète pas la même "
                "requête."
            )
        if status == SearchResultStatus.wrong_document_type.value:
            return (
                f"L'outil « {assessment.tool_name} » a retourné le mauvais "
                "type de document"
                + (f" ({assessment.returned_document})"
                   if assessment.returned_document else "")
                + ". Reformule pour viser le bon type (décision vs loi) "
                  "ou utilise un outil adapté."
            )
        expected = assessment.expected_document or "le document demandé"
        returned = (f" au lieu de « {assessment.returned_document} »"
                    if assessment.returned_document else "")
        return (
            f"Le résultat de « {assessment.tool_name} » ne contient pas "
            f"« {expected} »{returned}. Reformule la requête en visant "
            f"explicitement « {expected} » (titre exact), ou utilise "
            "fetch_document avec le bon identifiant. Ne fonde AUCUNE "
            "réponse sur ce résultat."
        )
