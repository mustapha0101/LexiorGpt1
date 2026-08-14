# -*- coding: utf-8 -*-
"""update_active_task — l'enjeu actif et l'objectif courant du tour.

L'enjeu actif (la question de fond ouverte en début d'échange) survit
aux suivis; l'objectif courant est TOUJOURS la dernière demande.
"""

from __future__ import annotations

from typing import Any
import uuid

from lexior.agentic.schemas import AcceptanceResult, ClaimLedger, RepairReport

from ..context import GraphContext
from ..state import LexiorState
from ._common import detect_case_reference, first_user_content

NAME = "update_active_task"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    messages = state.get("messages", [])
    latest = state.get("latest_user_message", "")
    opening = first_user_content(messages) or state["scenario"].user_query

    context = dict(state.get("case_context") or {})
    active_issue = (state.get("active_issue") or context.get("active_issue")
                    or opening)

    # A substantive message that is not classified as a follow-up starts a
    # new task inside the same thread. The conversation remains available to
    # the UI, but research evidence and repair state do not cross the task
    # boundary.
    prior_issue = str(context.get("active_issue") or state.get("active_issue") or "").strip()
    new_task = bool(
        prior_issue
        and latest.strip() and latest.strip() != prior_issue.strip()
        and not state.get("refers_to_previous_answer", False)
    )
    if new_task:
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        context = {"active_issue": latest.strip(), "facts": {}}
        active_issue = latest.strip()
        return {
            "task_id": task_id,
            "previous_task_id": state.get("task_id", ""),
            "active_task_reset": True,
            "active_issue": active_issue,
            "current_user_goal": latest,
            "facts": {},
            "missing_facts_before_search": [],
            "missing_facts_before_application": [],
            "missing_critical_facts": [],
            "legal_regime": "unknown",
            "legal_regime_basis": "",
            "legal_regime_verified": False,
            "employment_sector": "",
            "substantive_law": "",
            "case_context": context,
            "prior_evidence": [],
            "article_reviews": {},
            "clarification_history": [],
            "tool_history": [],
            "sources": [],
            "usable_evidence": [],
            "search_evaluations": [],
            "candidate_sources": [],
            "citable_sources": [],
            "usable_evidence_entries": [],
            "alternative_sources": [],
            "invalidated_sources": [],
            "coverage_gaps": [],
            "official_rule_retrieved": False,
            "official_rule_sources": [],
            "case_law_verified": [],
            "case_law_search_status": "not_required",
            "usable_case_sources": [],
            "primary_authority_selection": {},
            "rule_contract": {},
            "source_sufficiency_decision": {},
            "remedy_intent": {},
            "normative_references": [],
            "regulation_verified": False,
            "answer_contract": None,
            "information_gap": "",
            "fact_analysis": {
                "jurisdiction": "", "user_goal": latest,
                "asserted_material_facts": [],
                "uncertain_material_facts": [],
                "raw_clarification_answers": [], "legal_elements": [],
            },
            "latest_decision": None,
            "planner_feedback": "",
            "pending_clarification": {},
            "clarification_answer": "",
            "claim_ledger": ClaimLedger(task_id=task_id),
            "failure_history": [],
            "failure_reports": [],
            "grounding_failures": [],
            "repair_history": [],
            "repair_count": 0,
            "repair": RepairReport(),
            "repair_from_node": "",
            "clarification_count": 0,
            "reformulation_count": 0,
            "case_law_retry_count": 0,
            "last_tool_call": None,
            "last_tool_normalization": {},
            "last_tool_result_status": "",
            "last_tool_assessment": None,
            "deterministic_blockers": [],
            "acceptance_blockers": [],
            "validation_issues": [],
            "validation_result": None,
            "acceptance_result": AcceptanceResult(),
            "delivered_to_user": False,
            "trajectory_accepted": False,
            "quality_accepted": False,
            "first_invalid_step": None,
        }

    # Le dossier distingue l'enjeu initial de la dernière demande. On garde
    # les énoncés de faits tels que formulés par la personne, sans leur
    # attribuer une qualification juridique ni inférer les éléments manquants.
    facts = dict(context.get("facts") or state.get("facts") or {})
    statements = list(facts.get("user_statements") or [])
    candidate = (latest or "").strip()
    if candidate and candidate != opening and candidate not in statements:
        statements.append(candidate)
    facts["user_statements"] = statements
    context.update({
        "active_issue": active_issue,
        "facts": facts,
        "task_id": state.get("task_id", ""),
    })

    case_ref = None
    for message in messages:
        found = detect_case_reference(message.content)
        if found:
            case_ref = found  # la mention la plus récente l'emporte

    return {
        "active_task_reset": False,
        "task_id": state.get("task_id", ""),
        "active_issue": active_issue,
        "current_user_goal": latest or active_issue,
        "active_case_or_document": (
            case_ref or state.get("active_case_or_document", "")),
        "facts": facts,
        "case_context": context,
    }
