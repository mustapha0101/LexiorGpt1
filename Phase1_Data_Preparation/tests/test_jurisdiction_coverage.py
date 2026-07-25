# -*- coding: utf-8 -*-
"""Quatre cas de juridiction, quatre comportements.

Les catégories existaient dans le YAML des distributions; le comportement
associé manquait. Une juridiction inconnue était devinée, et le droit d'une
autre province recevait une réponse québécoise ou fédérale improvisée.
"""

from __future__ import annotations

import pytest

from test_central_graph import _scenario, offline_runner

from lexior.agentic.schemas import Decision, PlannerDecision
from lexior.agentic.taxonomy import REQUEST_TYPES
from lexior.services.jurisdiction import (
    JurisdictionCoverage,
    classify_coverage,
    coverage_action,
)


@pytest.fixture()
def runner(catalog):
    return offline_runner(catalog)


# ── Classification ───────────────────────────────────────────────────────


@pytest.mark.parametrize("value,expected", [
    ("Québec", JurisdictionCoverage.supported_quebec),
    ("quebec", JurisdictionCoverage.supported_quebec),
    ("Federal", JurisdictionCoverage.supported_federal),
    ("fédéral", JurisdictionCoverage.supported_federal),
    ("Ontario", JurisdictionCoverage.unsupported_other_canadian),
    ("Alberta", JurisdictionCoverage.unsupported_other_canadian),
    ("hors Québec (province non précisée)",
     JurisdictionCoverage.unsupported_other_canadian),
    ("France", JurisdictionCoverage.unsupported_foreign),
    ("", JurisdictionCoverage.unknown),
])
def test_each_jurisdiction_lands_in_the_right_category(value, expected):
    assert classify_coverage(value) is expected


def test_another_province_with_a_federal_matter_is_covered():
    """L'exception explicite de la table de décision."""
    assert classify_coverage("Ontario", federal_matter=True) is (
        JurisdictionCoverage.supported_federal)
    assert coverage_action("Ontario", federal_matter=True) == "proceed"


def test_another_province_with_a_provincial_matter_is_declined():
    assert coverage_action("Ontario") == "decline"


@pytest.mark.parametrize("value,action", [
    ("Québec", "proceed"),
    ("Federal", "proceed"),
    ("Ontario", "decline"),
    ("Belgique", "decline"),
    ("", "clarify"),
])
def test_each_category_has_its_behaviour(value, action):
    assert coverage_action(value) == action


def test_an_unknown_jurisdiction_is_never_guessed():
    """Le point qui comptait : inconnu ≠ Québec par défaut."""
    assert classify_coverage("") is JurisdictionCoverage.unknown
    assert coverage_action("") == "clarify"


# ── Faits obligatoires par type de demande ───────────────────────────────


def test_the_taxonomy_declares_required_facts():
    assert REQUEST_TYPES["case_analysis"].required_facts
    assert REQUEST_TYPES["procedure_guidance"].required_facts


def test_required_facts_are_deterministically_checkable():
    """Un fait qu'aucune règle ne sait constater manquerait toujours et
    rejouerait la clarification sans fin."""
    from lexior.agent_graph.nodes.analyze_facts import _JURISDICTION_ALIASES

    for name, spec in REQUEST_TYPES.items():
        for fact in spec.required_facts:
            folded = fact.casefold()
            assert any(alias in folded for alias in _JURISDICTION_ALIASES) or (
                name in ("comparative_law", "document_analysis")), (
                f"{name}: « {fact} » n'est pas constatable automatiquement")


# ── Clarification forcée ─────────────────────────────────────────────────


def test_missing_facts_reach_the_state(runner):
    """analyze_facts écrit; le champ n'était lu par personne auparavant."""
    from lexior.agent_graph.nodes import analyze_facts as node
    from lexior.agent_graph.state import initial_state
    state = initial_state(_scenario(), mode="live", system_prompt="t")
    state["request_type"] = "case_analysis"
    state["resolved_jurisdiction"] = ""

    update = node.run(state, runner.context)

    assert update["missing_facts_before_search"], (
        "la juridiction inconnue est un fait manquant structurel")


def test_an_unknown_jurisdiction_forces_a_clarification(runner):
    from lexior.agent_graph.nodes import validate_plan as node
    from lexior.agent_graph.state import initial_state
    state = initial_state(_scenario(), mode="live", system_prompt="t")
    state.update(
        resolved_jurisdiction="",
        latest_decision=PlannerDecision(
            request_type="case_analysis", jurisdiction="",
            decision=Decision.call_tool, next_tool="get_ccq_articles",
            arguments={"start_article": 1726},
        ).model_dump(mode="json"),
        step=1,
    )

    update = node.run(state, runner.context)

    assert update["latest_decision"]["decision"] == "ask_clarification"
    assert update["latest_decision"]["clarification_question"]


def test_the_planner_cannot_skip_the_clarification(runner):
    """La décision est forcée par le vérificateur, pas espérée du planner."""
    from lexior.agent_graph.nodes import validate_plan as node
    from lexior.agent_graph.state import initial_state
    state = initial_state(_scenario(), mode="live", system_prompt="t")
    state.update(
        resolved_jurisdiction="",
        latest_decision=PlannerDecision(
            request_type="case_analysis", jurisdiction="Québec",
            decision=Decision.final_answer,
        ).model_dump(mode="json"),
        step=1,
    )

    update = node.run(state, runner.context)

    assert update["latest_decision"]["decision"] == "ask_clarification"
