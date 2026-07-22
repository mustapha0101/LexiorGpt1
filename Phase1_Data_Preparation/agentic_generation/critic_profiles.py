# -*- coding: utf-8 -*-
"""Critic evaluation profiles by request type.

Different types of legal requests need different evaluation rubrics.
This module maps each request type to a CriticProfile and provides
French-language rubric strings for the legal and agentic critics.
"""

from __future__ import annotations

from .schemas import CriticProfile

# ---------------------------------------------------------------------------
# Request-type to profile mapping (12 types)
# ---------------------------------------------------------------------------

REQUEST_TYPE_TO_PROFILE: dict[str, CriticProfile] = {
    "case_analysis": CriticProfile.factual_legal_analysis,
    "procedure_guidance": CriticProfile.procedure,
    "topic_research": CriticProfile.factual_legal_analysis,
    "exact_text_retrieval": CriticProfile.exact_text,
    "article_explanation": CriticProfile.legal_explanation,
    "case_law_research": CriticProfile.case_law_request,
    "law_or_regulation_identification": CriticProfile.regulation_lookup,
    "legislative_status_verification": CriticProfile.regulation_lookup,
    "document_analysis": CriticProfile.factual_legal_analysis,
    "comparative_law": CriticProfile.comparative,
    "dataset_coverage": CriticProfile.general,
    "non_legal": CriticProfile.non_legal,
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def get_profile(request_type: str) -> CriticProfile:
    """Return the critic profile for *request_type*, defaulting to ``general``."""
    return REQUEST_TYPE_TO_PROFILE.get(request_type, CriticProfile.general)


# ---------------------------------------------------------------------------
# French rubric strings
# ---------------------------------------------------------------------------

LEGAL_RUBRICS: dict[CriticProfile, str] = {
    CriticProfile.exact_text: (
        "Vérifie uniquement que le texte officiel est reproduit mot pour mot, "
        "sans ajout."
    ),
    CriticProfile.legal_explanation: (
        "Vérifie que le texte officiel est reproduit intégralement, "
        "suivi d'une explication fondée sur les sources."
    ),
    CriticProfile.factual_legal_analysis: (
        "Vérifie la règle applicable, l'application aux faits, "
        "les exceptions signalées et la fidélité aux sources récupérées."
    ),
    CriticProfile.case_law_request: (
        "Vérifie que les décisions citées proviennent des résultats "
        "d'outils et que l'analyse est fidèle."
    ),
    CriticProfile.regulation_lookup: (
        "Vérifie que le règlement est correctement identifié "
        "et son contenu fidèlement rapporté."
    ),
    CriticProfile.procedure: (
        "Vérifie la disposition procédurale, les délais mentionnés "
        "et la fidélité au CPC récupéré."
    ),
    CriticProfile.clarification: (
        "Vérifie que la clarification est pertinente "
        "et qu'aucune recherche prématurée n'a été faite."
    ),
    CriticProfile.non_legal: (
        "Vérifie qu'aucune règle juridique n'est inventée "
        "et que la réponse est appropriée."
    ),
    CriticProfile.failure_mode: (
        "Vérifie que l'échec de l'outil est clairement signalé "
        "et qu'aucune information n'est fabriquée."
    ),
    CriticProfile.federal: (
        "Vérifie la juridiction fédérale, la loi applicable "
        "et la fidélité au document récupéré."
    ),
    CriticProfile.comparative: (
        "Vérifie que les deux juridictions sont correctement identifiées "
        "et les sources fidèlement comparées."
    ),
    CriticProfile.general: (
        "Évalue la fidélité générale aux sources "
        "et la prudence de la conclusion."
    ),
}

AGENTIC_RUBRICS: dict[CriticProfile, str] = {
    CriticProfile.exact_text: (
        "Route mono-outil : récupérer l'article demandé puis arrêter."
    ),
    CriticProfile.legal_explanation: (
        "Récupérer le texte officiel ; jurisprudence optionnelle."
    ),
    CriticProfile.factual_legal_analysis: (
        "Recherche sémantique → texte officiel → jurisprudence "
        "si les faits le justifient."
    ),
    CriticProfile.case_law_request: (
        "Jurisprudence obligatoire ; texte de loi optionnel en support."
    ),
    CriticProfile.regulation_lookup: (
        "Recherche puis récupération du règlement identifié."
    ),
    CriticProfile.procedure: (
        "Recherche CPC → texte officiel. Jurisprudence optionnelle "
        "si la procédure le justifie."
    ),
    CriticProfile.clarification: (
        "Clarification avant toute recherche ; pas d'outil prématuré."
    ),
    CriticProfile.non_legal: (
        "Aucun outil ne doit être appelé."
    ),
    CriticProfile.failure_mode: (
        "L'outil échoue ou retourne vide ; "
        "une reformulation max puis réponse prudente."
    ),
    CriticProfile.federal: (
        "Route fédérale : search_legal_documents → fetch_document."
    ),
    CriticProfile.comparative: (
        "Outils québécois ET fédéraux nécessaires pour la comparaison."
    ),
    CriticProfile.general: (
        "Vérifier le routage et l'ordre des outils."
    ),
}
