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
_LEGAL_REGIME_FACT = "régime juridique de l'emploi (secteur fédéral ou provincial)"

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
        elif (state.get("request_intent") == "legal"
                and state.get("jurisdiction_material", False)
                and _JURISDICTION_FACT not in missing_before_search):
            missing_before_search.append(_JURISDICTION_FACT)
        else:
            missing_before_search = [
                fact for fact in missing_before_search
                if fact != _JURISDICTION_FACT
            ]
        if (state.get("request_intent") == "legal"
                and state.get("employment_regime_material", False)
                and state.get("legal_regime", "unknown") == "unknown"
                and _LEGAL_REGIME_FACT not in missing_before_search):
            missing_before_search.append(_LEGAL_REGIME_FACT)
        else:
            if state.get("legal_regime", "unknown") != "unknown":
                facts["legal_regime"] = state.get("legal_regime")
            missing_before_search = [
                fact for fact in missing_before_search
                if fact != _LEGAL_REGIME_FACT
            ]

    # Faits obligatoires du type de demande, en live uniquement (en dataset
    # le scénario porte ses propres faits requis) et UNE SEULE FOIS : après
    # une clarification, un fait qu'aucune règle déterministe ne sait
    # constater resterait « manquant » indéfiniment et rejouerait la
    # question sans fin.
    if live and not state.get("clarification_count", 0) \
            and not ctx.config.evidence_first_enabled:
        for fact in required_facts_for(state.get("request_type", "")):
            folded = fact.casefold()
            if resolved and any(alias in folded
                                for alias in _JURISDICTION_ALIASES):
                continue
            if _is_known(fact, facts) or fact in missing_before_search:
                continue
            missing_before_search.append(fact)

    # This first pass is intentionally descriptive. Legal elements are left
    # empty until the retrieved primary authority has produced a RuleContract.
    statements = [str(value).strip() for value in facts.get(
        "user_statements", []) if str(value).strip()]
    asserted: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for key, value in facts.items():
        if key in {"juridiction", "question_courante", "user_statements"}:
            continue
        if isinstance(value, dict) and value.get("value") is None:
            uncertain.append({"fact_id": str(key), **value})
        elif value not in (None, "", [], {}):
            asserted.append({"fact_id": str(key), "value": value,
                             "certainty": "asserted"})
    for index, statement in enumerate(statements):
        asserted.append({"fact_id": f"statement_{index}", "value": statement,
                         "certainty": "asserted"})
    fact_analysis = {
        "jurisdiction": resolved,
        "legal_regime": state.get("legal_regime", "unknown"),
        "user_goal": state.get("latest_user_intent") or state.get(
            "latest_user_message", ""),
        "asserted_material_facts": asserted,
        "uncertain_material_facts": uncertain,
        "raw_clarification_answers": [
            entry.get("answer", "") for entry in state.get(
                "clarification_history", []) if entry.get("answer")
        ],
        "legal_elements": [],
    }
    return {
        "facts": facts,
        "missing_facts_before_search": missing_before_search,
        "fact_analysis": fact_analysis,
    }
