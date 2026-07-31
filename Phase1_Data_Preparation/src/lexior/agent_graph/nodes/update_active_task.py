# -*- coding: utf-8 -*-
"""update_active_task — l'enjeu actif et l'objectif courant du tour.

L'enjeu actif (la question de fond ouverte en début d'échange) survit
aux suivis; l'objectif courant est TOUJOURS la dernière demande.
"""

from __future__ import annotations

from typing import Any

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
    })

    case_ref = None
    for message in messages:
        found = detect_case_reference(message.content)
        if found:
            case_ref = found  # la mention la plus récente l'emporte

    return {
        "active_issue": active_issue,
        "current_user_goal": latest or active_issue,
        "active_case_or_document": (
            case_ref or state.get("active_case_or_document", "")),
        "facts": facts,
        "case_context": context,
    }
