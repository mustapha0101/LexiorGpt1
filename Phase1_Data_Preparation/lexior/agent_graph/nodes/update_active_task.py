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

    active_issue = state.get("active_issue") or opening

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
    }
