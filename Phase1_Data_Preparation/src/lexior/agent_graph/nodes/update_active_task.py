# -*- coding: utf-8 -*-
"""update_active_task — l'enjeu actif et l'objectif courant du tour.

L'enjeu actif (la question de fond ouverte en début d'échange) survit
aux suivis; l'objectif courant est TOUJOURS la dernière demande.
"""

from __future__ import annotations

from typing import Any
import uuid

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
    has_prior_research = bool(context.get("prior_evidence") or state.get("prior_evidence"))
    new_task = bool(
        prior_issue and has_prior_research
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
            "primary_authority_selection": {},
            "rule_contract": {},
            "source_sufficiency_decision": {},
            "normative_references": [],
            "answer_contract": None,
            "information_gap": "",
            "claim_ledger": {},
            "failure_history": [],
            "failure_reports": [],
            "grounding_failures": [],
            "repair_history": [],
            "repair_count": 0,
            "clarification_count": 0,
            "reformulation_count": 0,
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
