# -*- coding: utf-8 -*-
"""Preuves de la migration : UN graphe LangGraph central, deux modes.

Couvre les 20 exigences de la spécification de refonte :
graphe partagé, services partagés, clarification synthétique vs
interrupt(), absence de moteur legacy, classification des résultats,
réparation ciblée, juridiction verrouillée, suivis conversationnels,
compatibilité JSONL/ChatML, streaming depuis le graphe.
"""

from __future__ import annotations

import inspect

import pytest

from agentic_generation.config import AgenticConfig
from agentic_generation.fixtures import MOCK_MCP_FIXTURES
from agentic_generation.scenario_generator import ScenarioGenerator
from agentic_generation.schemas import (
    CriticResult,
    Decision,
    Message,
    PlannerDecision,
    Role,
    ScenarioSpec,
    SearchEvaluation,
    ToolObservation,
    TrainingTrajectory,
)
from agentic_generation.storage import RunStorage
from agentic_generation.trajectory_agent import TrajectoryAgent
from lexior.agent_graph import GraphRunner, build_context
from lexior.agent_graph.nodes import NODE_NAMES
from lexior.agent_graph.nodes import (
    build_answer_contract as node_build_answer_contract,
    classify_failures as node_classify_failures,
    classify_follow_up as node_classify_follow_up,
    classify_request as node_classify_request,
    validate_plan as node_validate_plan,
)
from lexior.agent_graph.state import initial_state
from lexior.services import (
    AnswerGenerationService,
    DatasetExportService,
    JurisdictionResolution,
    JurisdictionService,
    PlannerService,
    ResultVerificationService,
    build_mock_executor,
    build_services,
)


# ── Aides ────────────────────────────────────────────────────────────────


def offline_runner(catalog, planner=None, writer=None, *,
                   max_tool_calls=4, no_critics=False, storage=None):
    config = AgenticConfig(seed=3407, offline=True, dry_run=True,
                           max_tool_calls=max_tool_calls,
                           no_critics=no_critics)
    executor = build_mock_executor(catalog, MOCK_MCP_FIXTURES)
    services = build_services(config, catalog, executor=executor,
                              storage=storage)
    if planner is not None:
        services.planner = PlannerService.from_agent(planner, catalog)
    if writer is not None:
        services.answers = AnswerGenerationService.from_writer(writer)
        services.repair.answer_service = services.answers
    return GraphRunner(build_context(config, catalog, services))


class ScriptedPlanner:
    """Planner factice : rejoue une liste de décisions."""

    offline = True
    client = None

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, state):
        self.calls += 1
        if len(self.decisions) > 1:
            return self.decisions.pop(0)
        return self.decisions[0]


def _ask(question="Habitez-vous au Québec?"):
    return PlannerDecision(
        request_type="case_analysis", jurisdiction="",
        decision=Decision.ask_clarification,
        clarification_question=question,
    )


def _final():
    return PlannerDecision(
        request_type="case_analysis", jurisdiction="Québec",
        decision=Decision.final_answer,
        thinking_text="Je réponds avec les éléments connus.",
    )


def _scenario(**kw):
    defaults = dict(
        scenario_id="central-001", scenario_family_id="test",
        request_type="case_analysis",
        user_query="Mon employeur refuse de payer mes heures supplémentaires.",
    )
    defaults.update(kw)
    return ScenarioSpec(**defaults)


@pytest.fixture()
def runner(catalog):
    return offline_runner(catalog)


# ── 1-2. Un seul graphe compilé, mêmes nœuds pour les deux modes ─────────


def test_both_modes_share_the_same_compiled_graph(catalog):
    planner = ScriptedPlanner([_final()])
    shared = offline_runner(catalog, planner=planner,
                            writer=TrajectoryAgent(offline=True))
    graph_before = shared.graph

    dataset_result = shared.run_dataset(_scenario())
    live_result = shared.run_live("Question de test?")

    assert shared.graph is graph_before  # jamais recompilé par mode
    assert dataset_result.final_state.get("mode") == "dataset"
    assert live_result.final_state.get("mode") == "live"


def test_compiled_graph_has_all_central_nodes(runner):
    compiled_nodes = set(runner.graph.get_graph().nodes)
    compiled_nodes -= {"__start__", "__end__"}
    assert compiled_nodes == set(NODE_NAMES)
    # Les deux sorties de mode appartiennent au MÊME graphe.
    assert {"export_dataset", "return_live_answer"} <= compiled_nodes


# ── 3-4. Clarification : synthétique (dataset) vs interrupt (live) ───────


def test_dataset_clarification_uses_synthetic_answer(catalog):
    runner = offline_runner(catalog)
    scenario = ScenarioGenerator(seed=3407, offline=True).generate(
        "case_analysis", clarification_stage="before_search")
    assert scenario.effective_clarification_answer

    result = runner.run_dataset(scenario)
    assert result.accepted, result.rejection
    messages = result.trajectory.messages
    question_idx = next(
        i for i, m in enumerate(messages)
        if m.role == Role.assistant and m.content.rstrip().endswith("?"))
    follow = messages[question_idx + 1]
    assert follow.role == Role.user
    assert follow.content == scenario.effective_clarification_answer


def test_live_clarification_uses_interrupt_and_resume(catalog):
    planner = ScriptedPlanner([_ask(), _final()])
    runner = offline_runner(catalog, planner=planner,
                            writer=TrajectoryAgent(offline=True))

    first = runner.run_live("Mon employeur ne paie pas mes heures.",
                            thread_id="t-interrupt")
    assert first.pending_question == "Habitez-vous au Québec?"
    assert first.final_answer == ""

    second = runner.resume_live("t-interrupt", "non")
    assert second.pending_question is None
    assert second.final_answer
    assert planner.calls == 2
    contents = [m.content for m in second.final_state["messages"]]
    assert "non" in contents  # la réponse reprise est dans la conversation


# ── 5-7. Mêmes services dans les deux modes ──────────────────────────────


def test_both_modes_use_the_same_planner_service(catalog):
    planner = ScriptedPlanner([_final()])
    runner = offline_runner(catalog, planner=planner,
                            writer=TrajectoryAgent(offline=True))
    service = runner.context.services.planner
    seen_modes = []
    original = service.propose

    def spy(state, mode, feedback=None):
        seen_modes.append(mode)
        return original(state, mode, feedback)

    service.propose = spy
    try:
        runner.run_dataset(_scenario())
        runner.run_live("Question?")
    finally:
        service.propose = original

    assert "dataset" in seen_modes and "live" in seen_modes
    assert runner.context.services.planner is service


def test_both_modes_use_the_same_result_verifier(catalog):
    call_tool = PlannerDecision(
        request_type="case_analysis", jurisdiction="Québec",
        decision=Decision.call_tool,
        next_tool="get_ccq_articles",
        arguments={"start_article": 1726},
    )
    planner = ScriptedPlanner([call_tool, _final()])
    runner = offline_runner(catalog, planner=planner,
                            writer=TrajectoryAgent(offline=True))
    verifier = runner.context.services.verification
    assert isinstance(verifier, ResultVerificationService)

    seen_modes = []
    original = verifier.assess

    def spy(observation, issues=None, **kwargs):
        return original(observation, issues, **kwargs)

    def spy_with_mode(observation, issues=None, **kwargs):
        seen_modes.append(True)
        return original(observation, issues, **kwargs)

    verifier.assess = spy_with_mode
    try:
        runner.context.services.planner = PlannerService.from_agent(
            ScriptedPlanner([call_tool, _final()]), catalog)
        runner.run_dataset(_scenario())
        dataset_calls = len(seen_modes)
        assert dataset_calls >= 1
        runner.context.services.planner = PlannerService.from_agent(
            ScriptedPlanner([call_tool, _final()]), catalog)
        runner.run_live("Donne-moi l'article 1726 du CCQ, je suis à Québec.")
        assert len(seen_modes) > dataset_calls
    finally:
        verifier.assess = original


def test_both_modes_use_the_same_critics_and_blockers(catalog):
    planner = ScriptedPlanner([_final()])
    runner = offline_runner(catalog, planner=planner,
                            writer=TrajectoryAgent(offline=True))
    critics = runner.context.services.critics
    validation = runner.context.services.validation
    calls = []
    original = critics.evaluate

    def spy(state, answer):
        calls.append(state.scenario.scenario_id)
        return original(state, answer)

    critics.evaluate = spy
    try:
        runner.run_dataset(_scenario())
        runner.run_live("Question?")
    finally:
        critics.evaluate = original

    assert len(calls) >= 2  # dataset ET live passent par les mêmes critics
    assert runner.context.services.validation is validation


# ── 8-9. Plus aucun moteur legacy ────────────────────────────────────────


def test_legacy_orchestrator_contains_no_orchestration_loop():
    from agentic_generation.orchestrator import AgenticOrchestrator

    source = inspect.getsource(AgenticOrchestrator)
    assert "while" not in source
    assert "for " not in inspect.getsource(AgenticOrchestrator.run)
    assert "graph_runner.run_dataset" in inspect.getsource(
        AgenticOrchestrator.run)
    assert "StateStatus.planning" not in source


def test_cli_does_not_run_the_old_engine():
    import agentic_generation.cli as cli

    source = inspect.getsource(cli)
    assert "AgenticOrchestrator" not in source
    assert "run_dataset" in source


# ── 10. Mauvaise loi ⇒ irrelevant (régression heures supplémentaires) ────


def test_wrong_statute_is_classified_irrelevant():
    verifier = ResultVerificationService()
    observation = ToolObservation(
        tool_name="search_legal_documents",
        arguments={"query": "heures supplémentaires Code canadien du travail"},
        normalized_response=(
            "Loi sur l'équité salariale — L.C. 2018, ch. 27, art. 416. "
            "Cette loi vise à atteindre l'équité salariale par des moyens "
            "proactifs en corrigeant la discrimination systémique fondée "
            "sur le sexe."),
        ok=True,
    )
    assessment = verifier.assess(observation)
    assert assessment.tool_call_succeeded is True
    assert assessment.search_status == "irrelevant"
    assert assessment.expected_document == "Canada Labour Code"
    assert "équité salariale" in assessment.returned_document.lower()
    assert assessment.usable_as_evidence is False


def test_wrong_statute_routes_to_reformulation():
    from lexior.agent_graph.routing import route_after_classification

    state = initial_state(_scenario(), system_prompt="t")
    state.update(last_tool_result_status="irrelevant",
                 reformulation_count=0, max_reformulations=1)
    assert route_after_classification(state) == "reformulate_search"


# ── 11. Aucune preuve utilisable ⇒ pas de réponse de fond ────────────────


def test_no_usable_evidence_blocks_substantive_answer(runner):
    state = initial_state(_scenario(), mode="live", system_prompt="t")
    state.update(
        tool_history=[ToolObservation(
            tool_name="search_legal_documents",
            arguments={"query": "heures supplémentaires"},
            normalized_response="Loi sur l'équité salariale ...",
            ok=True)],
        search_evaluations=[SearchEvaluation(
            tool_call_index=0, tool_name="search_legal_documents",
            result_status="irrelevant")],
        usable_evidence=[],
        request_type="case_analysis",
    )
    update = node_build_answer_contract.run(state, runner.context)
    contract = update["answer_contract"]
    assert contract["mode_de_reponse"] == "no_evidence"
    assert any("n'affirme aucune règle de fond" in c.lower()
               or "aucune preuve utilisable" in c.lower()
               for c in contract["consignes"])


# ── 12-13. La réparation retourne au premier nœud invalide ───────────────


def test_retrieval_failure_returns_to_retrieval(runner):
    graph_edges = {(e.source, e.target)
                   for e in runner.graph.get_graph().edges}
    assert ("reformulate_search", "plan") in graph_edges
    assert ("repair_trajectory", "plan") in graph_edges


def test_writing_failure_returns_only_to_answer_repair(runner):
    state = initial_state(_scenario(), system_prompt="t")
    state.update(
        critic_results={
            "legal": CriticResult(
                critic="legal", accepted=False, score=0.4,
                issues=["Formulation non étayée par les sources."]),
            "agentic": CriticResult(
                critic="agentic", accepted=True, score=0.9),
        },
        repair_count=0, max_repairs=1,
    )
    update = node_classify_failures.run(state, runner.context)
    assert update["repair_from_node"] == "repair_answer"

    graph_edges = {(e.source, e.target)
                   for e in runner.graph.get_graph().edges}
    assert ("repair_answer", "run_critics") in graph_edges


def test_jurisdiction_failure_returns_to_jurisdiction(runner):
    state = initial_state(_scenario(), system_prompt="t")
    state.update(
        critic_results={
            "legal": CriticResult(
                critic="legal", accepted=False, score=0.3,
                issues=["La juridiction appliquée est incorrecte "
                        "(hors Québec)."]),
        },
        repair_count=0, max_repairs=1,
    )
    update = node_classify_failures.run(state, runner.context)
    assert update["repair_from_node"] == "resolve_jurisdiction"


# ── 14. La juridiction verrouillée ne change jamais silencieusement ──────


def test_locked_jurisdiction_survives_planner_inference():
    service = JurisdictionService()
    locked = JurisdictionResolution(
        value="Ontario", basis="explicit_user_statement",
        locked=True, verified=True)
    result = service.resolve_dataset(_scenario(), "Québec", locked)
    assert result.value == "Ontario"

    live = service.resolve_live(
        [Message(role=Role.user, content="question sans indice")], locked)
    assert live.value == "Ontario" and live.locked


def test_validate_plan_overrides_decision_with_locked_jurisdiction(runner):
    state = initial_state(_scenario(), mode="live", system_prompt="t")
    state.update(
        resolved_jurisdiction="Ontario",
        jurisdiction_locked=True,
        latest_decision=PlannerDecision(
            request_type="case_analysis", jurisdiction="Québec",
            decision=Decision.call_tool,
            next_tool="search_legal_documents",
            arguments={"query": "overtime"},
        ).model_dump(mode="json"),
        step=1,
    )
    update = node_validate_plan.run(state, runner.context)
    assert update["latest_decision"]["jurisdiction"] == "Ontario"


def test_validate_plan_blocks_quebec_tools_outside_quebec(runner):
    state = initial_state(_scenario(), mode="live", system_prompt="t")
    state.update(
        resolved_jurisdiction="hors Québec (province non précisée)",
        jurisdiction_locked=True,
        latest_decision=PlannerDecision(
            request_type="case_analysis", jurisdiction="",
            decision=Decision.call_tool,
            next_tool="get_ccq_articles",
            arguments={"start_article": 1726},
        ).model_dump(mode="json"),
        step=1,
    )
    update = node_validate_plan.run(state, runner.context)
    assert update["latest_decision"]["decision"] == "final_answer"
    assert update["latest_decision"]["next_tool"] is None


# ── 15. « donne-moi le site » : le suivi est LA question courante ────────


def test_follow_up_site_request_targets_current_question(runner):
    state = initial_state(
        _scenario(user_query="donne-moi le site"),
        mode="live", system_prompt="t")
    state["messages"] = [
        Message(role=Role.system, content="t"),
        Message(role=Role.user,
                content="Quelles sont les règles sur les heures "
                        "supplémentaires au fédéral?"),
        Message(role=Role.assistant,
                content="Le Code canadien du travail prévoit ... "
                        "(réponse complète)."),
        Message(role=Role.user, content="donne-moi le site"),
    ]
    state["latest_user_message"] = "donne-moi le site"

    for module in (node_classify_request, node_classify_follow_up):
        state.update(module.run(state, runner.context))
    from lexior.agent_graph.nodes import update_active_task
    state.update(update_active_task.run(state, runner.context))
    update = node_build_answer_contract.run(state, runner.context)

    contract = update["answer_contract"]
    assert state["refers_to_previous_answer"] is True
    assert state["requested_output_type"] == "source_url"
    assert contract["question_courante"] == "donne-moi le site"
    assert contract["question_de_suivi"] is True
    assert any("ne répète pas" in c for c in contract["consignes"])


# ── 16-17. JSONL intermédiaire compatible; ChatML reste un pas séparé ────


def test_jsonl_intermediate_output_remains_compatible(catalog, tmp_path):
    storage = RunStorage(tmp_path, "graph-run")
    runner = offline_runner(catalog, storage=storage)
    result = runner.run_dataset(
        ScenarioGenerator(seed=3407, offline=True).generate(
            "exact_text_retrieval"))
    assert result.accepted

    lines = storage.accepted_path.read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json as _json
    restored = TrainingTrajectory.model_validate(_json.loads(lines[0]))
    assert restored.quality.accepted_for_intermediate is True
    assert restored.tool_trace and restored.messages


def test_chatml_export_is_a_separate_step(catalog):
    from agentic_generation.training_formatter import (
        export_trajectory_for_training,
    )
    from lexior.agent_graph.nodes import export_dataset

    # Le nœud d'export n'importe ni n'appelle le formateur ChatML
    # (sa docstring peut le MENTIONNER comme pas séparé).
    source = inspect.getsource(export_dataset)
    assert "import training_formatter" not in source
    assert "from agentic_generation.training_formatter" not in source
    assert "format_for_finetuning" not in source
    assert "export_trajectory_for_training(" not in source

    runner = offline_runner(catalog)
    result = runner.run_dataset(
        ScenarioGenerator(seed=3407, offline=True).generate(
            "exact_text_retrieval"))
    assert result.accepted
    formatted = export_trajectory_for_training(result.trajectory)
    assert formatted  # conversion déterministe séparée, toujours possible


# ── 18-19. Streaming depuis le graphe; traces comparables ────────────────


def test_streaming_events_come_from_the_central_graph(catalog):
    planner = ScriptedPlanner([_final()])
    runner = offline_runner(catalog, planner=planner,
                            writer=TrajectoryAgent(offline=True))
    events = list(runner.stream_live("Question de test?"))

    kinds = {e["type"] for e in events}
    assert {"status", "token", "done"} <= kinds
    status_nodes = {e["node"] for e in events if e["type"] == "status"}
    assert status_nodes <= set(NODE_NAMES)
    assert "plan" in status_nodes
    assert "return_live_answer" in status_nodes
    assert events[-1]["type"] == "done" and events[-1]["accepted"] is True


def test_dataset_and_live_traces_are_comparable(catalog):
    planner = ScriptedPlanner([_final()])
    runner = offline_runner(catalog, planner=planner,
                            writer=TrajectoryAgent(offline=True))
    dataset_result = runner.run_dataset(_scenario())
    live_result = runner.run_live("Question de test?")

    dataset_traj = dataset_result.final_state.get("trajectory")
    live_traj = live_result.final_state.get("trajectory")
    assert dataset_traj and live_traj
    assert set(dataset_traj.keys()) == set(live_traj.keys())
    assert live_traj["resolved_jurisdiction"] == "Québec"
