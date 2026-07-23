# -*- coding: utf-8 -*-
"""plan — le planner PROPOSE une action; il ne route jamais.

La proposition part ensuite vers ``validate_plan`` (vérificateur
déterministe) qui l'autorise, la modifie ou la rejette; l'arête
conditionnelle de LangGraph choisit le nœud suivant.
"""

from __future__ import annotations

import json
from typing import Any

from ..context import GraphContext
from ..state import LexiorState, to_research_state

NAME = "plan"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    research_state = to_research_state(state)
    feedback = state.get("planner_feedback", "")

    decision = ctx.services.planner.propose(
        research_state, state.get("mode", "dataset"),
        feedback=feedback or None,
    )

    log_line = (
        f"[planner] etape {state.get('step', 0) + 1}: "
        f"{decision.decision.value}"
        + (f" -> {decision.next_tool} "
           f"{json.dumps(decision.arguments)[:120]}"
           if decision.next_tool else "")
        + (f" | juridiction: {decision.jurisdiction}"
           if decision.jurisdiction else "")
    )
    # Console Windows cp1252 : rester en ASCII pour ne jamais faire
    # échouer le nœud sur un simple log.
    print(log_line.encode("ascii", "backslashreplace").decode("ascii"),
          flush=True)

    return {
        "latest_decision": decision.model_dump(mode="json"),
        "step": state.get("step", 0) + 1,
        "missing_critical_facts": decision.missing_critical_facts,
        "planner_feedback": "",  # correctif consommé
    }
