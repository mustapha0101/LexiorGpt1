# -*- coding: utf-8 -*-
"""analyze_facts — faits disponibles et faits manquants avant recherche.

Dataset : les faits viennent du scénario (déjà dans l'état initial).
Live : la juridiction inconnue est LE fait manquant structurel quand la
réponse dépend de la province — c'est ce qui pousse le planner à poser
la question avant de conclure.

À cela s'ajoutent les ``required_facts`` du type de demande : pour une
question locateur/locataire, il faut au minimum la juridiction et les
faits déterminants. Ce que ce nœud écrit dans
``missing_facts_before_search`` est LU par ``validate_plan``, qui force
alors une clarification — la décision ne dépend pas du bon vouloir du
planner.
"""

from __future__ import annotations

from typing import Any

from lexior.agentic.taxonomy import REQUEST_TYPES
from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState

NAME = "analyze_facts"

_JURISDICTION_FACT = "juridiction (province ou fédéral)"

# Un fait requis nommant la juridiction est satisfait par la juridiction
# résolue : inutile de le redemander.
_JURISDICTION_ALIASES = ("juridiction", "province")


def _is_known(fact: str, facts: dict[str, Any]) -> bool:
    folded = fact.casefold()
    for key, value in facts.items():
        if not str(value).strip():
            continue
        key_folded = str(key).casefold()
        if key_folded in folded or folded in key_folded:
            return True
    return False


def required_facts_for(request_type: str) -> list[str]:
    spec = REQUEST_TYPES.get(request_type)
    return list(spec.required_facts) if spec else []


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    facts = dict(state.get("facts", {}))
    missing_before_search = list(state.get("missing_facts_before_search", []))
    live = is_live(state.get("mode", ""))
    resolved = state.get("resolved_jurisdiction", "")

    if live:
        facts["question_courante"] = state.get("latest_user_intent", "")
        if resolved:
            facts["juridiction"] = resolved
            missing_before_search = [
                f for f in missing_before_search if f != _JURISDICTION_FACT]
        elif (state.get("request_type") != "non_legal"
                and _JURISDICTION_FACT not in missing_before_search):
            missing_before_search.append(_JURISDICTION_FACT)

    # Faits obligatoires du type de demande, en live uniquement (en dataset
    # le scénario porte ses propres faits requis) et UNE SEULE FOIS : après
    # une clarification, un fait qu'aucune règle déterministe ne sait
    # constater resterait « manquant » indéfiniment et rejouerait la
    # question sans fin.
    if live and not state.get("clarification_count", 0):
        for fact in required_facts_for(state.get("request_type", "")):
            folded = fact.casefold()
            if resolved and any(alias in folded
                                for alias in _JURISDICTION_ALIASES):
                continue
            if _is_known(fact, facts) or fact in missing_before_search:
                continue
            missing_before_search.append(fact)

    return {
        "facts": facts,
        "missing_facts_before_search": missing_before_search,
    }
