# -*- coding: utf-8 -*-
"""repair_answer — la rédaction est fautive, les preuves sont bonnes.

Réécrit la réponse selon les instructions des critiques, sans ajouter
de fait ni de source. Retourne ensuite à ``run_critics`` pour une
réévaluation (une réparation = un cycle, borné par ``max_repairs``).
"""

from __future__ import annotations

from typing import Any

from agentic_generation.schemas import Message, RepairReport, Role
from lexior.services.critics import CriticsOutcome

from ..context import GraphContext
from ..state import LexiorState, to_research_state

NAME = "repair_answer"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    answer = state.get("final_answer", "")
    thinking = state.get("final_reasoning_summary", "")

    instructions: list[str] = []
    for report in state.get("failure_reports", []):
        if report.get("target_node") == NAME:
            instructions.extend(report.get("instructions", []))
    if not instructions:
        critics = state.get("critic_results", {}) or {}
        outcome = CriticsOutcome(
            legal=critics.get("legal"), agentic=critics.get("agentic"))
        instructions = outcome.failing(
            ctx.config.legal_min_score, ctx.config.agentic_min_score)

    updates: dict[str, Any] = {
        "repair_count": state.get("repair_count", 0) + 1,
        "repair_from_node": "",
    }

    if not instructions:
        return updates

    repaired_thinking, repaired_answer = ctx.services.repair.repair_answer(
        to_research_state(state), state.get("mode", "dataset"),
        answer, thinking, instructions)

    history_entry = {
        "from_node": NAME,
        "category": "writing",
        "instructions": list(instructions),
    }

    if repaired_answer != answer:
        messages = list(state.get("messages", []))
        if messages and messages[-1].role == Role.assistant:
            messages[-1] = Message(
                role=Role.assistant,
                thinking=repaired_thinking or None,
                content=repaired_answer,
            )
        updates.update({
            "messages": messages,
            "final_answer": repaired_answer,
            "final_reasoning_summary": repaired_thinking or "",
            "repair": RepairReport(
                attempted=True, status="successful",
                changes=list(instructions)),
        })
        history_entry["status"] = "successful"
    else:
        updates["repair"] = RepairReport(
            attempted=True, status="failed",
            reason="la réparation n'a pas modifié la réponse")
        history_entry["status"] = "failed"

    updates["repair_history"] = list(
        state.get("repair_history", [])) + [history_entry]
    return updates
