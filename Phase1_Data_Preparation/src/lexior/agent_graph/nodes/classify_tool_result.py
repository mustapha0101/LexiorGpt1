# -*- coding: utf-8 -*-
"""classify_tool_result — three-tier evidence classification.

Separates: technical success, official status, relevance to the legal
issue, court scope compatibility, and evidence level (candidate / citable
/ usable). An official source does NOT automatically become usable.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from lexior.services.evidence import EvidenceLevel
from lexior.services.provenance import (
    a_une_provenance, numero_demande, reponses_reussies,
)

from ..context import GraphContext
from ..state import LexiorState

NAME = "classify_tool_result"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    tool_history = state.get("tool_history", [])
    if not tool_history:
        return {}

    observation = tool_history[-1]
    prior = state.get("last_tool_assessment") or {}
    # Un numéro d'article dont la provenance est établie échappe au veto
    # lexical : sa pertinence a été jugée quand le numéro a été produit.
    numero = numero_demande(observation.tool_name, observation.arguments)
    provenance = numero is not None and a_une_provenance(
        numero,
        state.get("active_issue") or state.get("latest_user_message", ""),
        reponses_reussies(tool_history, avant=observation))
    assessment = ctx.services.verification.assess(
        observation,
        prior.get("verifier_issues"),
        resolved_jurisdiction=state.get("resolved_jurisdiction", ""),
        requested_court_scope=state.get("requested_court_scope", ""),
        requested_document_type="",
        # active_issue et NON latest_user_message : après une clarification,
        # handle_clarification écrase latest_user_message par la réponse de
        # l'usager (« au Québec », « préciser quoi? »). L'article récupéré
        # était alors comparé à cette réponse et classé « sans rapport
        # thématique » — l'article 1466, qui traite précisément des dommages
        # causés par un animal, est ressorti irrelevant sur une question de
        # morsure de chien. update_active_task conserve la question d'origine.
        user_query=(state.get("active_issue")
                    or state.get("latest_user_message", "")),
        provenance_verifiee=provenance,
    )

    index = len(tool_history) - 1
    evaluations = list(state.get("search_evaluations", []))
    evaluations.append(assessment.to_search_evaluation(index))

    # Legacy index list (backward compat).
    usable_evidence = list(state.get("usable_evidence", []))
    if assessment.usable_as_evidence:
        usable_evidence.append(index)

    # Three-tier evidence collections.
    entry = assessment.to_evidence_entry(index)
    entry_dict = entry.to_dict()

    candidate_sources = list(state.get("candidate_sources", []))
    citable_sources = list(state.get("citable_sources", []))
    usable_entries = list(state.get("usable_evidence_entries", []))
    alternative_sources = list(state.get("alternative_sources", []))
    invalidated_sources = list(state.get("invalidated_sources", []))

    level = EvidenceLevel(assessment.evidence_level)
    if level == EvidenceLevel.usable:
        usable_entries.append(entry_dict)
    elif level == EvidenceLevel.citable:
        if assessment.detailed_status == "alternative_only":
            alternative_sources.append(entry_dict)
        else:
            citable_sources.append(entry_dict)
    else:
        candidate_sources.append(entry_dict)

    if not assessment.usable_as_evidence and not assessment.relevant:
        invalidated_sources.append(entry_dict)

    updates: dict[str, Any] = {
        "search_evaluations": evaluations,
        "usable_evidence": usable_evidence,
        "last_tool_result_status": assessment.search_status,
        "last_tool_assessment": asdict(assessment),
        "candidate_sources": candidate_sources,
        "citable_sources": citable_sources,
        "usable_evidence_entries": usable_entries,
        "alternative_sources": alternative_sources,
        "invalidated_sources": invalidated_sources,
    }

    if assessment.jurisdiction_dims:
        dims = assessment.jurisdiction_dims
        if dims.get("returned_court_scope"):
            updates["returned_court_scope"] = dims["returned_court_scope"]
        if dims.get("returned_source_jurisdiction"):
            updates["returned_source_jurisdiction"] = (
                dims["returned_source_jurisdiction"])

    return updates
