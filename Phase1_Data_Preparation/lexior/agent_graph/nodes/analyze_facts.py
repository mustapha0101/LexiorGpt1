# -*- coding: utf-8 -*-
"""analyze_facts — faits disponibles et faits manquants avant recherche.

Dataset : les faits viennent du scénario (déjà dans l'état initial).
Live : la juridiction inconnue est LE fait manquant structurel quand la
réponse dépend de la province — c'est ce qui pousse le planner à poser
la question avant de conclure.
"""

from __future__ import annotations

from typing import Any

from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState

NAME = "analyze_facts"

_JURISDICTION_FACT = "juridiction (province ou fédéral)"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    facts = dict(state.get("facts", {}))
    missing_before_search = list(state.get("missing_facts_before_search", []))

    if is_live(state.get("mode", "")):
        facts["question_courante"] = state.get("latest_user_intent", "")
        resolved = state.get("resolved_jurisdiction", "")
        if resolved:
            facts["juridiction"] = resolved
            missing_before_search = [
                f for f in missing_before_search if f != _JURISDICTION_FACT]
        elif (state.get("request_type") != "non_legal"
                and _JURISDICTION_FACT not in missing_before_search):
            missing_before_search.append(_JURISDICTION_FACT)

    return {
        "facts": facts,
        "missing_facts_before_search": missing_before_search,
    }
