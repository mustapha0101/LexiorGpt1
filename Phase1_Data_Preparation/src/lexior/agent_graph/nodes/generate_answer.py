# -*- coding: utf-8 -*-
"""generate_answer — rédaction finale sous contrat."""

from __future__ import annotations

from typing import Any

from lexior.agentic.schemas import Message, Role

from ..context import GraphContext
from ..state import LexiorState, to_research_state

NAME = "generate_answer"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    messages = state.get("messages", [])
    stop = state.get("stop_reason", "")

    # Trajectoire « clarification » (dataset sans réponse synthétique) :
    # la question déjà posée EST la réponse finale.
    if (stop == "clarification_required" and messages
            and messages[-1].role == Role.assistant):
        return {
            "final_answer": messages[-1].content,
            "final_reasoning_summary": "",
            "status": "answering",
        }

    research_state = to_research_state(state)
    thinking, answer = ctx.services.answers.generate(
        research_state,
        state.get("mode", "dataset"),
        contract=state.get("answer_contract"),
    )

    new_messages = list(messages)
    new_messages.append(Message(
        role=Role.assistant,
        thinking=thinking or None,
        content=answer,
    ))

    return {
        "messages": new_messages,
        "final_answer": answer,
        "final_reasoning_summary": thinking or "",
        "status": "answering",
    }
