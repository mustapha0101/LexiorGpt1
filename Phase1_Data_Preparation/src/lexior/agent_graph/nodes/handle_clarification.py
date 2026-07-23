# -*- coding: utf-8 -*-
"""handle_clarification — même logique, deux livraisons.

Dataset : consomme la réponse synthétique du scénario; sans réponse
synthétique, la trajectoire d'entraînement se termine sur la question.
Live : ``interrupt()`` suspend le run; la vraie réponse arrive via
``Command(resume=...)`` et l'exécution reprend ICI, dans le même nœud.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from lexior.agentic.schemas import Message, PlannerDecision, Role
from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState

NAME = "handle_clarification"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    decision = PlannerDecision.model_validate(state["latest_decision"])
    question = ctx.services.clarification.build_question(
        decision, state.get("missing_critical_facts", []))

    messages = list(state.get("messages", []))
    messages.append(Message(role=Role.assistant, content=question))
    count = state.get("clarification_count", 0) + 1

    if is_live(state.get("mode", "")):
        # Suspension du graphe — la question part vers l'utilisateur réel.
        # À la reprise, interrupt() retourne sa réponse et le nœud
        # rejoue depuis le début (la construction ci-dessus est pure).
        answer = interrupt({
            "question": question,
            "missing_facts": list(state.get("missing_critical_facts", [])),
        })
        answer_text = str(answer or "").strip()
        messages.append(Message(role=Role.user, content=answer_text))
        return {
            "messages": messages,
            "clarification_count": count,
            "pending_clarification": "",
            "clarification_answer": answer_text,
            "latest_user_message": answer_text,
            "latest_user_intent": answer_text,
            "status": "planning",
        }

    synthetic = ctx.services.clarification.synthetic_answer(
        state["scenario"])
    if synthetic:
        messages.append(Message(role=Role.user, content=synthetic))
        return {
            "messages": messages,
            "clarification_count": count,
            "pending_clarification": "",
            "clarification_answer": synthetic,
            "status": "planning",
        }

    # Pas de réponse synthétique : l'exemple d'entraînement est la
    # question elle-même (trajectoire « clarification »).
    return {
        "messages": messages,
        "clarification_count": count,
        "pending_clarification": question,
        "final_answer": question,
        "status": "answering",
        "stop_reason": "clarification_required",
    }
