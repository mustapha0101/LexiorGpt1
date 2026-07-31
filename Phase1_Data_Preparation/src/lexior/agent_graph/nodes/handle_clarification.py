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


def _clarification_category(question: str, missing_facts: list[str]) -> str:
    """Catégorie explicite pour éviter de redemander la même information."""
    corpus = " ".join([question, *map(str, missing_facts)]).casefold()
    return "jurisdiction" if any(token in corpus for token in (
        "province", "juridiction", "fédéral", "federal")) else "fact"


def _history_entry(question: str, missing_facts: list[str], answer: str,
                   category: str, clarification: dict[str, Any] | None = None,
                   interpretation: str = "") -> dict[str, Any]:
    return {
        "clarification_id": (clarification or {}).get("clarification_id", ""),
        "category": category,
        "question": question,
        "fact_keys": list((clarification or {}).get(
            "fact_keys", missing_facts)),
        "missing_facts": list(missing_facts),
        "answer": answer,
        "answered": bool(answer),
        "answer_interpretation": interpretation,
        "status": "answered" if answer else "unanswered",
    }


def _interpret_answer(answer: str) -> str:
    folded = " ".join((answer or "").casefold().split())
    if folded in {"oui", "yes", "je pense que oui", "probablement",
                  "je crois que oui", "certainement"}:
        return "affirmative"
    if folded in {"non", "no", "je ne pense pas", "certainement pas"}:
        return "negative"
    if folded in {"je ne sais pas", "ne sais pas", "incertain", "incertaine",
                  "je ne suis pas certain", "je ne suis pas certaine"}:
        return "unresolved"
    return "explanation"


def _apply_fact_answer(facts: dict[str, Any], clarification: dict[str, Any],
                       answer: str) -> tuple[dict[str, Any], str]:
    interpretation = _interpret_answer(answer)
    keys = [str(key) for key in clarification.get("fact_keys", [])]
    if clarification.get("category") != "fact":
        return facts, interpretation
    for key in keys:
        value: Any
        if interpretation == "affirmative":
            value = True
        elif interpretation == "negative":
            value = False
        elif interpretation == "unresolved":
            value = None
        else:
            value = {"answer": answer, "interpretation": interpretation}
        facts[key] = {
            "value": value,
            "source": "user_clarification",
            "confidence": "asserted_by_user",
            "status": interpretation,
        }
    return facts, interpretation


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    decision = PlannerDecision.model_validate(state["latest_decision"])
    pending = dict(state.get("pending_clarification") or {})
    question = str(pending.get("question") or "").strip()
    if not question:
        question = ctx.services.clarification.build_question(
            decision, state.get("missing_critical_facts", []))
        pending = {
            "clarification_id": "fact-runtime",
            "category": _clarification_category(
                question, state.get("missing_critical_facts", [])),
            "fact_keys": list(state.get("missing_critical_facts", [])),
            "question": question,
            "answer_type": "yes_no_or_explanation",
            "source_articles": [],
            "status": "pending",
        }

    messages = list(state.get("messages", []))
    messages.append(Message(role=Role.assistant, content=question))
    count = state.get("clarification_count", 0) + 1
    missing_facts = list(state.get("missing_critical_facts", []))
    category = _clarification_category(question, missing_facts)

    if is_live(state.get("mode", "")):
        # Suspension du graphe — la question part vers l'utilisateur réel.
        # À la reprise, interrupt() retourne sa réponse et le nœud
        # rejoue depuis le début (la construction ci-dessus est pure).
        answer = interrupt({
            "question": question,
            "missing_facts": list(pending.get(
                "fact_keys", state.get("missing_critical_facts", []))),
            "clarification": pending,
        })
        answer_text = str(answer or "").strip()
        messages.append(Message(role=Role.user, content=answer_text))
        history = list(state.get("clarification_history", []))
        context = dict(state.get("case_context") or {})
        facts = dict(context.get("facts") or state.get("facts") or {})
        facts, interpretation = _apply_fact_answer(
            facts, pending, answer_text)
        history.append(_history_entry(
            question, missing_facts, answer_text, category, pending,
            interpretation))
        context.update({
            "facts": facts,
            "clarification_history": history,
            "pending_clarification": {},
        })
        return {
            "messages": messages,
            "clarification_count": count,
            "clarification_answer": answer_text,
            "latest_user_message": answer_text,
            "latest_user_intent": answer_text,
            "facts": facts,
            "clarification_history": history,
            "case_context": context,
            "pending_clarification": {},
            "status": "planning",
        }

    synthetic = ctx.services.clarification.synthetic_answer(
        state["scenario"])
    if synthetic:
        messages.append(Message(role=Role.user, content=synthetic))
        history = list(state.get("clarification_history", []))
        history.append(_history_entry(
            question, missing_facts, synthetic, category, pending,
            _interpret_answer(synthetic)))
        return {
            "messages": messages,
            "clarification_count": count,
            "clarification_answer": synthetic,
            "clarification_history": history,
            "pending_clarification": {},
            "status": "planning",
        }

    # Pas de réponse synthétique : l'exemple d'entraînement est la
    # question elle-même (trajectoire « clarification »).
    return {
        "messages": messages,
        "clarification_count": count,
        "pending_clarification": pending,
        "final_answer": question,
        "status": "answering",
        "stop_reason": "clarification_required",
    }
