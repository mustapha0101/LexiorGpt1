# -*- coding: utf-8 -*-
"""classify_follow_up — le dernier message prolonge-t-il l'échange ?

Détection déterministe : un suivi court (« donne-moi le site », « cite
l'article exact ») doit être traité comme LA question courante, jamais
comme une répétition de la question initiale.
"""

from __future__ import annotations

from typing import Any

from ..context import GraphContext
from ..state import LexiorState
from ._common import (
    last_assistant_content,
    looks_like_follow_up,
    user_turn_count,
)

NAME = "classify_follow_up"

_PREVIEW = 240


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    messages = state.get("messages", [])
    latest = state.get("latest_user_message", "")
    previous_answer = last_assistant_content(messages)

    is_follow_up = looks_like_follow_up(latest, bool(previous_answer))
    multi_turn = user_turn_count(messages) > 1

    return {
        "latest_user_intent": latest,
        "refers_to_previous_answer": is_follow_up,
        "already_answered": (previous_answer[:_PREVIEW]
                             if multi_turn else ""),
        "still_needed": latest if (is_follow_up or multi_turn) else "",
    }
