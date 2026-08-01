# -*- coding: utf-8 -*-
"""Regression tests added before fixing the dog-bite smoke-test failures."""

from lexior.agentic.config import load_config
from lexior.agentic.schemas import AcceptanceResult, ScenarioSpec, ToolObservation
from lexior.agent_graph.events import StreamTranslator
from lexior.agentic.tool_catalog import load_catalog
from lexior.agent_graph import build_context
from lexior.agent_graph.graph import _wrap
from lexior.agent_graph.nodes import (
    compute_acceptance, select_primary_authorities, update_research_state,
    validate_final,
)
from lexior.agent_graph.state import initial_state
from lexior.services import build_mock_executor, build_services
from lexior.services.evidence_first import build_claim_ledger
from lexior.agentic.schemas import PrimaryAuthoritySelection
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from typing import TypedDict


class MiniState(TypedDict, total=False):
    task_id: str
    status: str
    stop_reason: str
    deterministic_blockers: list[str]
    node_failed: str
    error_type: str
    thread_id: str
    ok: bool


ARTICLE_1466 = (
    "Le propriétaire d’un animal est tenu de réparer le préjudice que l’animal "
    "a causé, soit qu’il fût sous sa garde ou sous celle d’un tiers, soit qu’il "
    "fût égaré ou échappé.\nLa personne qui se sert de l’animal en est aussi, "
    "pendant ce temps, responsable avec le propriétaire."
)


def _dog_state_and_context():
    config = load_config()
    config.evidence_first_enabled = True
    config.offline = True
    config.dry_run = True
    catalog = load_catalog(config.catalog_path)
    services = build_services(
        config, catalog, executor=build_mock_executor(catalog, {}))
    context = build_context(config, catalog, services)
    scenario = ScenarioSpec(
        scenario_id="dog-bite-regression",
        scenario_family_id="dog-bite",
        request_type="legal_question",
        user_query=("Le chien de ma voisine m'a mordu alors que je marchais "
                    "sur le trottoir. Est-ce que je peux porter plainte?"),
        jurisdiction="Québec",
    )
    observation = ToolObservation(
        tool_name="get_ccq_articles",
        arguments={"articles": [1466, 1457, 2498]},
        normalized_response=(
            f"Article 1466\n{ARTICLE_1466}\n"
            "Article 1457\nToute personne a le devoir de respecter les règles.\n"
            "Article 2498\nL’assureur peut être tenu à certaines obligations."),
        ok=True,
    ).finalize_hash()
    state = initial_state(scenario, mode="dataset", thread_id="thread-dog")
    state["tool_history"] = [observation]
    state["active_issue"] = scenario.user_query
    state["case_context"] = {"facts": {}}
    return state, context


def test_update_research_state_accepts_three_article_response_without_typeerror():
    state, context = _dog_state_and_context()
    updates = update_research_state.run(state, context)

    assert updates["status"] == "planning"
    assert updates["official_rule_retrieved"] is True
    assert updates["prior_evidence"]
    assert set(updates["article_reviews"]) == {"1466", "1457", "2498"}
    assert all(not review["rule_roles"]
               for review in updates["article_reviews"].values())

    next_state = {**state, **updates}
    selection_update = select_primary_authorities.run(next_state, context)
    assert selection_update["status"] == "planning"


def test_faithful_animal_owner_paraphrase_is_verified():
    selection = PrimaryAuthoritySelection(
        task_id="task-dog", primary_sources=["ccq:1466"])
    claim = ("Selon l'article 1466, le propriétaire d'un animal est tenu de "
             "réparer le préjudice causé par l'animal, même s'il était sous "
             "la garde d'un tiers.")
    ledger = build_claim_ledger(
        claim, selection, {"ccq:1466": ARTICLE_1466}, task_id="task-dog")

    assert ledger.claims[0].verification_status == "verified"
    assert ledger.claims[0].support_type in {"direct", "reasonable_inference"}
    assert ARTICLE_1466.split("\n")[0] in ledger.claims[0].premises or ledger.claims[0].support_type == "direct"


def _ledger_for_claim(claim: str, source: str = "Le propriétaire peut demander une réparation."):
    return build_claim_ledger(
        claim,
        PrimaryAuthoritySelection(task_id="task-claims", primary_sources=["ccq:1"]),
        {"ccq:1": source},
        task_id="task-claims",
    ).claims[0]


def test_changed_modality_is_rejected_with_reason():
    claim = _ledger_for_claim("Selon l'article 1, le propriétaire doit obtenir une réparation.")
    assert claim.verification_status == "failed"
    assert claim.modality_changed is True
    assert "modalité" in (claim.failure_reason or "")


def test_invented_prior_knowledge_condition_is_rejected():
    claim = _ledger_for_claim(
        "Selon l'article 1, le propriétaire peut demander une réparation "
        "seulement s'il connaissait le danger."
    )
    assert claim.verification_status == "failed"
    assert claim.added_conditions


def test_correct_article_with_wrong_exception_is_rejected():
    claim = _ledger_for_claim(
        "Selon l'article 1, le propriétaire peut demander une réparation "
        "sauf si la victime est un voisin.")
    assert claim.verification_status == "failed"
    assert claim.added_conditions or claim.failure_reason


def test_wrapped_node_exception_routes_to_reject_without_following_static_edge():
    visited = []
    graph = StateGraph(MiniState)
    graph.add_node("bad", _wrap(
        "update_research_state",
        lambda state, ctx: (_ for _ in ()).throw(TypeError("boom")), None))
    graph.add_node("select_primary_authorities",
                   lambda state: visited.append("select") or {})
    graph.add_node("extract_rule_contract",
                   lambda state: visited.append("extract") or {})
    graph.add_node("reject", lambda state: visited.append("reject") or {})
    graph.set_entry_point("bad")
    graph.add_conditional_edges(
        "bad", lambda state: "reject" if state.get("status") == "rejected" else "select",
        {"reject": "reject", "select": "select_primary_authorities"},
    )
    graph.add_edge("select_primary_authorities", "extract_rule_contract")
    graph.add_edge("extract_rule_contract", END)
    graph.add_edge("reject", END)

    result = graph.compile().invoke({"task_id": "task-error"})

    assert visited == ["reject"]
    assert result["status"] == "rejected"
    assert "update_research_state: TypeError" in result["stop_reason"]


def test_wrapped_normal_node_keeps_its_habitual_route():
    visited = []
    graph = StateGraph(MiniState)
    graph.add_node("first", _wrap("first", lambda state, ctx: {"ok": True}, None))
    graph.add_node("next", lambda state: visited.append("next") or {"ok": state["ok"]})
    graph.set_entry_point("first")
    graph.add_conditional_edges(
        "first", lambda state: "next" if state.get("ok") else "reject",
        {"next": "next", "reject": "next"},
    )
    graph.add_edge("next", END)
    assert graph.compile().invoke({})["ok"] is True
    assert visited == ["next"]


def test_graph_interrupt_is_propagated_by_the_common_wrapper():
    wrapped = _wrap("clarification", lambda state, ctx: (_ for _ in ()).throw(GraphInterrupt()), None)
    try:
        wrapped({})
    except GraphInterrupt:
        pass
    else:
        raise AssertionError("GraphInterrupt must not become a rejected update")


def test_live_failed_claim_is_repaired_and_old_failure_is_resolved():
    state, context = _dog_state_and_context()
    state.update({
        "mode": "live",
        "final_answer": "Selon l'article 1466, le propriétaire doit obtenir une amende automatique.",
        "answer_contract": {
            "filtre_articles_officiels": True,
            "articles_retenus": ["1466"],
        },
        "failure_history": [{
            "claim": "Selon l'article 1466, le propriétaire doit obtenir une amende automatique.",
            "failure_type": "unsupported_legal_claim",
            "status": "open",
        }],
    })
    update = validate_final.run(state, context)
    merged = {**state, **update}
    acceptance = compute_acceptance.run(merged, None)

    assert update["final_answer"]
    assert not any(claim.verification_status == "failed"
                   for claim in update["claim_ledger"].claims)
    assert any(item.get("status") == "resolved"
               for item in update["failure_history"])
    assert acceptance["quality_accepted"] is True
    assert acceptance["open_grounding_failures_total"] == 0


def test_live_invalid_fallback_is_safe_and_not_a_technical_error():
    state, context = _dog_state_and_context()
    state.update({
        "mode": "live",
        "tool_history": [],
        "final_answer": "Selon l'article 1466, le propriétaire est toujours coupable.",
        "answer_contract": {"filtre_articles_officiels": True, "articles_retenus": []},
    })
    update = validate_final.run(state, context)

    assert update["final_answer"].startswith("Je ne peux pas")
    assert "unsupported_legal_claim" not in update["final_answer"]
    assert not any(claim.verification_status == "failed"
                   for claim in update["claim_ledger"].claims)


def test_observability_does_not_turn_absent_failures_into_empty_list():
    events = list(StreamTranslator(thread_id="thread-observe").translate_chunk({
        "compute_acceptance": {
            "task_id": "task-observe",
            "acceptance_result": AcceptanceResult(accepted=True),
        }
    }))
    statuses = [item for item in events if item.get("type") == "status"]
    assert statuses
    assert "grounding_failures" not in statuses[0]
    observations = [item["event"] for item in events
                    if item.get("type") == "observability"]
    assert observations[0]["task_id"] == "task-observe"
