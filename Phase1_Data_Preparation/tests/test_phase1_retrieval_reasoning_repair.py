# -*- coding: utf-8 -*-
"""Contrats déterministes du flux live de Phase 1."""

from __future__ import annotations

from lexior.agentic.case_law_gate import gate_search_results
from lexior.agentic.schemas import (
    CaseRelevanceResult,
    Decision,
    Message,
    PlannerDecision,
    ResearchState,
    Role,
    ScenarioSpec,
    ToolObservation,
)
from lexior.agentic.planner_agent import PlannerAgent
from lexior.agent_graph.events import StreamTranslator, result_metadata
from lexior.services.article_review import (
    assess_legislative_sufficiency,
    build_clarification,
    enrich_article_review,
)
from lexior.agent_graph.nodes.handle_clarification import (
    _apply_fact_answer,
)


TREE = (
    "Un arbre pourri du voisin est tombé sur mon garage. "
    "Le dommage matériel est lié à la chute."
)


def _state(catalog, **updates):
    scenario = ScenarioSpec(
        scenario_id="phase1-tree",
        scenario_family_id="live",
        request_type="case_analysis",
        user_query=TREE,
    )
    base = dict(
        scenario=scenario,
        messages=[Message(role=Role.user, content=TREE)],
        case_description=TREE,
        current_turn_tool_count=0,
        max_search_reformulations=1,
    )
    base.update(updates)
    return ResearchState(**base)


def _review(number, text, *, status="conditionally_applicable", rank=1,
            facts=TREE):
    return enrich_article_review(
        article_number=number,
        status=status,
        reason="motif interne à ne jamais afficher",
        text=text,
        facts=facts,
        rank=rank,
    )


def test_clarification_uses_primary_missing_fact_not_contextual_reason():
    reviews = {
        "contextual": {
            "status": "conditionally_applicable",
            "retrieval_group": "contextual",
            "reason": "faute lourde et clause limitative",
            "missing_fact_keys": ["exclusion_clause"],
            "clarification_priority": 999,
        },
        "primary": _review(
            "primary",
            "Toute personne doit respecter les règles de conduite et réparer "
            "le préjudice causé par sa faute.",
            rank=11,
        ),
    }
    selected = build_clarification(reviews, {}, [], [TREE])
    assert selected is not None
    assert selected["source_articles"] == ["primary"]
    assert selected["fact_keys"]
    assert "faute lourde" not in selected["question"].casefold()
    assert "que pouvez-vous confirmer" not in selected["question"].casefold()


def test_known_fact_is_not_asked_again_and_no_exclusion_question_is_created():
    review = _review(
        "primary",
        "Toute personne doit respecter les règles et réparer le dommage "
        "causé par sa faute.",
        facts=(TREE + " Le voisin savait que l'arbre risquait de tomber et "
               "n'a pris aucune mesure."),
    )
    assert build_clarification(
        {"primary": review},
        {"prior_knowledge": {"value": True}},
        [],
        [TREE + " Le voisin savait que l'arbre risquait de tomber et n'a rien fait."],
    ) is None
    exclusion = _review(
        "contextual",
        "Une clause limitative peut exclure la responsabilité en cas de faute lourde.",
    )
    assert build_clarification({"x": exclusion}, {}, [], [TREE]) is None


def test_short_answers_bind_to_pending_clarification():
    pending = {
        "category": "fact",
        "fact_keys": ["prior_knowledge", "failure_to_take_reasonable_action"],
    }
    facts, interpretation = _apply_fact_answer({}, pending, "oui")
    assert interpretation == "affirmative"
    assert facts["prior_knowledge"]["value"] is True
    assert facts["failure_to_take_reasonable_action"]["value"] is True
    facts, interpretation = _apply_fact_answer({}, pending, "non")
    assert interpretation == "negative"
    assert facts["prior_knowledge"]["value"] is False
    facts, interpretation = _apply_fact_answer({}, pending, "je ne sais pas")
    assert interpretation == "unresolved"
    assert facts["prior_knowledge"]["value"] is None
    facts, interpretation = _apply_fact_answer({}, {}, "oui")
    assert facts == {}


def test_sufficiency_requires_roles_and_does_not_count_contextual_text():
    general = _review(
        "general",
        "Toute personne doit respecter les règles et réparer le préjudice "
        "causé par sa faute.",
        status="applicable",
        rank=1,
    )
    contextual = _review(
        "contextual",
        "Si un arbre menace de tomber, le propriétaire peut demander de le "
        "redresser.",
        status="applicable",
        rank=2,
    )
    result = assess_legislative_sufficiency(
        {"general": general, "contextual": contextual}, TREE,
        remaining_candidates=True,
    )
    assert not result.sufficient
    assert result.should_fetch_next_batch
    assert "contextual" in result.contextual_articles


def test_progressive_fetch_can_reach_rank_eleven_and_stops_at_twenty(catalog):
    candidates = "\n".join(
        f"{i}. Article {1000 + i} — confiance 0.{90 - i:02d}"
        for i in range(1, 21)
    )
    state = _state(
        catalog,
        tool_history=[ToolObservation(
            tool_name="semantic_search_ccq",
            arguments={"query": TREE},
            normalized_response=candidates,
            ok=True,
        )],
        article_reviews={
            "1001": _review(
                "1001",
                "Toute personne doit respecter les règles et réparer le préjudice.",
                status="applicable",
            )
        },
    )
    planner = PlannerAgent(catalog, chat_mode=True, initial_article_fetch_k=6,
                           article_fetch_batch_size=6)
    args = planner._article_fetch_arguments(state, "get_ccq_articles")
    assert args == {"articles": [1001, 1002, 1003, 1004, 1005, 1006]}
    fetched = ToolObservation(
        tool_name="get_ccq_articles", arguments=args,
        normalized_response="Article 1001\nTexte officiel.", ok=True,
    )
    state.tool_history.append(fetched)
    next_args = planner._article_fetch_arguments(state, "get_ccq_articles")
    assert next_args["articles"][0] == 1007
    assert len(next_args["articles"]) <= 6


def test_case_law_query_uses_full_dossier_and_only_primary_articles(catalog):
    state = _state(
        catalog,
        case_facts={
            "prior_knowledge": {"value": True},
            "failure_to_take_reasonable_action": {"value": True},
            "damage_assessment": "garage endommagé",
        },
        article_reviews={
            "primary": _review(
                "primary",
                "Toute personne doit respecter les règles et réparer le préjudice.",
                status="conditionally_applicable",
            ),
            "contextual": _review(
                "contextual",
                "Une clause limitative exclut la responsabilité en cas de faute lourde.",
                status="conditionally_applicable",
            ),
        },
    )
    query = PlannerAgent(catalog, chat_mode=True)._build_quebec_case_law_query(state)
    assert "garage" in query
    assert "prior knowledge" in query
    assert "article primary" in query
    assert "article contextual" not in query


def test_irrelevant_case_law_never_fetches_and_one_reformulation_is_bounded(catalog):
    search = ToolObservation(
        tool_name="search_quebec_jurisprudence",
        arguments={"query": TREE},
        normalized_response="Aucun résultat pertinent.",
        source_urls=["https://rejected.example/decision"],
        ok=True,
    )
    state = _state(
        catalog,
        tool_history=[search],
        article_reviews={"primary": _review(
            "primary",
            "Toute personne doit respecter les règles et réparer le préjudice.",
            status="applicable",
        )},
        case_law_search_status="irrelevant",
        reformulation_count=1,
    )
    planner = PlannerAgent(catalog, chat_mode=True)
    final = PlannerDecision(
        request_type="case_analysis", decision=Decision.final_answer,
    )
    planned = planner._guard_live_source_completeness(state, final)
    assert planned.decision == Decision.final_answer
    assert planner._arguments("get_quebec_regulation", state) is None


def test_accepted_case_candidate_uses_quebec_fetch_and_not_federal_fetch(catalog):
    candidate = CaseRelevanceResult(
        usable=True, source_url="https://accepted.example/decision",
        citation="2024 QCCQ 1",
    )
    state = _state(
        catalog,
        tool_history=[ToolObservation(
            tool_name="search_quebec_jurisprudence", arguments={"query": TREE},
            normalized_response="2024 QCCQ 1 Article 1457", ok=True,
        )],
        usable_case_sources=[candidate],
        case_law_search_status="candidates_pending_fetch",
        article_reviews={"primary": _review(
            "primary",
            "Toute personne doit respecter les règles et réparer le préjudice.",
            status="applicable",
        )},
    )
    planner = PlannerAgent(catalog, chat_mode=True)
    assert planner._arguments("get_quebec_regulation", state) == {
        "url": "https://accepted.example/decision"
    }
    proposed = PlannerDecision(
        request_type="case_analysis", decision=Decision.call_tool,
        next_tool="fetch_document", arguments={"url": candidate.source_url},
    )
    corrected = planner._guard_live_tool_chain_compatibility(state, proposed)
    assert corrected.next_tool == "get_quebec_regulation"


def test_stream_metadata_and_resume_do_not_replay_old_tools():
    long_text = "\n".join(f"Article {i}" for i in range(1, 21)) + "x" * 500
    metadata = result_metadata(
        "semantic_search_ccq", {"query": "dossier"}, long_text)
    assert metadata["preview_truncated"] is True
    assert metadata["candidate_count"] == 20
    assert result_metadata(
        "get_ccq_articles", {"articles": [985, 1457]}, "Article 985"
    )["article_count"] == 2

    translator = StreamTranslator(initial_tool_count=1)
    events = list(translator.translate_chunk({
        "execute_tool": {"tool_history": [
            type("Obs", (), {"tool_name": "old", "arguments": {},
                              "normalized_response": "old", "ok": True})(),
            type("Obs", (), {"tool_name": "new", "arguments": {"query": "x"},
                              "normalized_response": long_text, "ok": True})(),
        ]}
    }))
    calls = [event for event in events if event["type"] == "tool_call"]
    assert [event["tool"] for event in calls] == ["new"]


def test_gate_rejected_url_is_not_an_accepted_case_candidate():
    results, status = gate_search_results(
        "2024 QCCA 1\nArticle 900\nSujet différent",
        ["1457"], TREE,
        source_urls=["https://rejected.example/decision"],
    )
    assert not any(item.usable for item in results)
    assert status.value == "irrelevant"
