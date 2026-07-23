# -*- coding: utf-8 -*-
"""build_answer_contract — le contrat que la réponse doit honorer.

Fige, AVANT rédaction : la question exacte à laquelle répondre (le
dernier suivi, pas la question initiale), la juridiction établie, les
preuves utilisables — et le mode de réponse.

The writer receives ONLY:
  - usable_evidence (three-tier classified, level == usable)
  - explicitly labelled alternative_sources (when the contract allows)
It must NOT receive irrelevant candidate results as legal evidence.

Without usable evidence, the graph chooses: clarification, reformulation,
equivalent tool, coverage limitation, or cannot_conclude.
"""

from __future__ import annotations

from typing import Any

from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState

NAME = "build_answer_contract"

_SUBSTANTIVE_TYPES_NEEDING_EVIDENCE = {
    "exact_text_retrieval", "article_explanation", "topic_research",
    "case_analysis", "law_or_regulation_identification",
}

_COVERAGE_LIMITATION_FR = (
    "Je n'ai pas pu récupérer et vérifier {source_desc} à partir des "
    "sources actuellement disponibles dans ce système.")


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    tool_history = state.get("tool_history", [])
    exempt = ctx.services.validation.compute_exempt_tools(tool_history)
    live = is_live(state.get("mode", ""))

    # ── Validation de route (dataset, avant rédaction) ───────────────────
    if not live and state.get("stop_reason") != "clarification_required":
        route_errors = ctx.services.validation.validate_tool_route(
            state["scenario"].request_type,
            [o.tool_name for o in tool_history],
            exempt_tools=exempt,
        )
        if route_errors:
            return {
                "status": "rejected",
                "stop_reason": "; ".join(route_errors),
                "deterministic_blockers": list(route_errors),
                "exempt_tools": exempt,
            }

    # ── Evidence from the three-tier collections ─────────────────────────
    usable_entries = state.get("usable_evidence_entries", [])
    alternative_entries = state.get("alternative_sources", [])
    coverage_gaps = state.get("coverage_gaps", [])

    usable_idx = state.get("usable_evidence", [])
    usable_tools = [
        tool_history[i].tool_name for i in usable_idx
        if 0 <= i < len(tool_history)
    ]

    unusable = [
        {"tool": e.tool_name, "status": e.result_status}
        for e in state.get("search_evaluations", [])
        if e.result_status not in ("usable", "exact_match", "truncated")
    ]

    attempted_research = bool(tool_history)
    has_usable = bool(usable_entries) or bool(usable_idx)
    needs_evidence = (
        state.get("request_type", "")
        in _SUBSTANTIVE_TYPES_NEEDING_EVIDENCE)

    if not attempted_research:
        answer_mode = "direct"
    elif has_usable:
        answer_mode = "grounded"
    elif needs_evidence:
        answer_mode = "no_evidence"
    else:
        answer_mode = "direct"

    directives: list[str] = [
        "Réponds à la DERNIÈRE question de l'utilisateur.",
    ]
    if state.get("refers_to_previous_answer"):
        directives.append(
            "C'est une question de suivi : ne répète pas la réponse "
            "précédente; fournis exactement ce qui est demandé "
            f"({state.get('requested_output_type', 'answer')}).")
    if answer_mode == "no_evidence":
        directives.append(
            "AUCUNE preuve utilisable n'a été récupérée : n'affirme "
            "aucune règle de fond; explique la limite de recherche et "
            "oriente vers les sources officielles (CanLII, SOQUIJ).")

    # Coverage gap directives.
    for gap in coverage_gaps:
        desc = gap.get("requested_court_scope", "") or gap.get(
            "requested_document_type", "cette source")
        directives.append(_COVERAGE_LIMITATION_FR.format(
            source_desc=f"une décision de {desc}"))

    # Alternative sources directive.
    alternatives_for_contract = []
    if alternative_entries and answer_mode in ("grounded", "no_evidence"):
        for alt in alternative_entries:
            alternatives_for_contract.append({
                "tool": alt.get("tool_name", ""),
                "status": alt.get("detailed_status", "alternative_only"),
                "reason": alt.get("reason", ""),
            })
        directives.append(
            "Des sources ALTERNATIVES (non-équivalentes) sont "
            "disponibles. Tu peux les mentionner en les identifiant "
            "explicitement comme alternatives, jamais comme réponses "
            "directes à la demande.")

    contract = {
        "question_courante": (state.get("latest_user_intent")
                              or state["scenario"].user_query),
        "enjeu_actif": state.get("active_issue", ""),
        "question_de_suivi": state.get("refers_to_previous_answer", False),
        "type_de_sortie": state.get("requested_output_type", "answer"),
        "juridiction_etablie": (state.get("resolved_jurisdiction")
                                or "unknown"),
        "juridiction_verrouillee": state.get("jurisdiction_locked", False),
        "mode_de_reponse": answer_mode,
        "preuves_utilisables": usable_tools,
        "preuves_inutilisables": unusable,
        "sources_alternatives": alternatives_for_contract,
        "lacunes_de_couverture": [g for g in coverage_gaps],
        "consignes": directives,
    }

    return {
        "answer_contract": contract,
        "exempt_tools": exempt,
        "status": "answering",
    }
