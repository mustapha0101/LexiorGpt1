# -*- coding: utf-8 -*-
"""Chat-mode guards: deterministic jurisdiction gating + small-model parsing."""

import pytest

from agentic_generation.planner_agent import (
    PlannerAgent,
    QC_ONLY_TOOLS,
    QuebecToolsBlocked,
)
from agentic_generation.schemas import (
    Decision,
    Message,
    PlannerDecision,
    ResearchState,
    Role,
    ScenarioSpec,
)
from agentic_generation.trajectory_agent import TrajectoryAgent


def _state(*turns: tuple[str, str]) -> ResearchState:
    scenario = ScenarioSpec(
        scenario_id="t", scenario_family_id="chat",
        request_type="case_analysis", language="fr",
        user_query=turns[-1][1] if turns else "q",
        jurisdiction="quebec",
    )
    roles = {"user": Role.user, "assistant": Role.assistant}
    return ResearchState(
        scenario=scenario,
        messages=[Message(role=roles[r], content=c) for r, c in turns],
    )


# ── _chat_jurisdiction_hint ──────────────────────────────────────────────


def test_hint_non_after_quebec_question():
    state = _state(
        ("user", "mon employeur refuse de payer mes heures supplémentaires"),
        ("assistant", "Habitez-vous au Québec?"),
        ("user", "non"),
    )
    hint = PlannerAgent._chat_jurisdiction_hint(state)
    assert hint == "hors Québec (province non précisée)"


def test_hint_oui_after_quebec_question():
    state = _state(
        ("assistant", "Habitez-vous au Québec?"),
        ("user", "oui"),
    )
    assert PlannerAgent._chat_jurisdiction_hint(state) == "Québec"


def test_hint_province_mention():
    state = _state(("user", "je vis en ontario et mon patron ne paie pas"))
    assert PlannerAgent._chat_jurisdiction_hint(state) == "Ontario"


def test_hint_no_signal():
    state = _state(("user", "question de bail"))
    assert PlannerAgent._chat_jurisdiction_hint(state) is None


def test_hint_latest_signal_wins():
    state = _state(
        ("user", "je vivais en ontario"),
        ("assistant", "D'accord."),
        ("user", "mais maintenant je suis à montréal"),
    )
    assert PlannerAgent._chat_jurisdiction_hint(state) == "Québec"


# ── _guard_chat_jurisdiction ─────────────────────────────────────────────


def _qc_tool_decision() -> PlannerDecision:
    return PlannerDecision(
        request_type="case_analysis", jurisdiction="",
        decision=Decision.call_tool,
        next_tool="get_ccq_articles",
        arguments={"start_article": 1726},
    )


def test_guard_blocks_qc_tool_outside_quebec(catalog):
    planner = PlannerAgent(catalog, chat_mode=True)
    state = _state(
        ("assistant", "Habitez-vous au Québec?"),
        ("user", "non"),
    )
    with pytest.raises(QuebecToolsBlocked):
        planner._guard_chat_jurisdiction(state, _qc_tool_decision(),
                                         retried=False)


def test_guard_converts_to_final_answer_after_retry(catalog):
    planner = PlannerAgent(catalog, chat_mode=True)
    state = _state(
        ("assistant", "Habitez-vous au Québec?"),
        ("user", "non"),
    )
    decision = planner._guard_chat_jurisdiction(
        state, _qc_tool_decision(), retried=True)
    assert decision.decision == Decision.final_answer
    assert decision.next_tool is None
    assert "hors Québec" in decision.jurisdiction


def test_guard_allows_qc_tool_in_quebec(catalog):
    planner = PlannerAgent(catalog, chat_mode=True)
    state = _state(
        ("assistant", "Habitez-vous au Québec?"),
        ("user", "oui"),
    )
    decision = planner._guard_chat_jurisdiction(
        state, _qc_tool_decision(), retried=False)
    assert decision.decision == Decision.call_tool
    assert decision.next_tool == "get_ccq_articles"
    assert decision.jurisdiction == "Québec"


def test_guard_allows_federal_tool_outside_quebec(catalog):
    planner = PlannerAgent(catalog, chat_mode=True)
    state = _state(("user", "je travaille en alberta"))
    decision = PlannerDecision(
        request_type="case_analysis", jurisdiction="",
        decision=Decision.call_tool,
        next_tool="search_legal_documents",
        arguments={"query": "overtime"},
    )
    out = planner._guard_chat_jurisdiction(state, decision, retried=False)
    assert out.next_tool == "search_legal_documents"
    assert out.jurisdiction == "Alberta"


def test_qc_only_tools_all_in_catalog(catalog):
    known = set(catalog.tools) | {"search_quebec_jurisprudence"}
    assert QC_ONLY_TOOLS <= known


# ── _split_thinking_answer : formats de petits modèles ──────────────────


def test_split_degenerate_duplicated_raisonnement():
    text = (
        "RAISONNEMENT : La loi X ne s'applique pas.\n\n---\n\n"
        "RAISONNEMENT : La loi X ne s'applique pas.\n\n"
        "Selon le Code canadien du travail, les heures supplémentaires "
        "sont payées à 1,5 fois le taux normal."
    )
    thinking, answer = TrajectoryAgent._split_thinking_answer(text)
    assert thinking == "La loi X ne s'applique pas."
    assert answer.startswith("Selon le Code canadien du travail")
    assert "RAISONNEMENT" not in answer


def test_split_official_separator_still_works():
    thinking, answer = TrajectoryAgent._split_thinking_answer(
        "analyse ici---ANSWER---la réponse")
    assert thinking == "analyse ici"
    assert answer == "la réponse"


def test_split_separator_with_inner_newlines():
    """Un petit modèle insère parfois des sauts de ligne dans le
    séparateur : « ---\\n\\nANSWER--- » doit séparer quand même."""
    thinking, answer = TrajectoryAgent._split_thinking_answer(
        "RAISONNEMENT : analyse détaillée.\n\n---\n\nANSWER---\n\n"
        "La réponse finale.")
    assert thinking == "analyse détaillée."
    assert answer == "La réponse finale."


def test_split_reponse_label_still_works():
    thinking, answer = TrajectoryAgent._split_thinking_answer(
        "RAISONNEMENT : analyse.\n\n---\n\nRÉPONSE : la réponse finale.")
    assert thinking == "analyse."
    assert answer == "la réponse finale."


def test_split_plain_text_unchanged():
    thinking, answer = TrajectoryAgent._split_thinking_answer(
        "Bonjour, comment puis-je aider?")
    assert thinking == ""
    assert answer == "Bonjour, comment puis-je aider?"
