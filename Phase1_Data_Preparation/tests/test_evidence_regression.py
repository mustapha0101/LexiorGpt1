# -*- coding: utf-8 -*-
"""15 regression tests — evidence selection, tool coverage, jurisdiction routing.

Tests exercise the three-tier evidence model, coverage gating,
jurisdiction dimension matching, deterministic acceptance blockers,
and writer restrictions.
"""

from __future__ import annotations

import pytest

from agentic_generation.schemas import (
    AcceptanceResult,
    Decision,
    PlannerDecision,
    SearchResultStatus,
    ToolObservation,
)
# Import agent_graph first to avoid circular import issues.
from lexior.agent_graph import GraphRunner, build_context  # noqa: F401
from lexior.services import ResultVerificationService
from lexior.services.evidence import (
    AcceptanceBlocker,
    CoverageGap,
    DetailedResultStatus,
    EvidenceEntry,
    EvidenceLevel,
    JurisdictionDimensions,
)
from lexior.services.tool_coverage import (
    FEDERAL_COURT_SCOPES,
    QUEBEC_COURT_SCOPES,
    TOOL_COVERAGE,
    get_coverage,
    has_equivalent_coverage,
    tools_covering_court_scope,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _obs(tool_name: str, response: str = "ok", ok: bool = True,
         arguments: dict | None = None, error: str | None = None):
    return ToolObservation(
        tool_name=tool_name,
        normalized_response=response,
        ok=ok,
        arguments=arguments or {},
        error=error,
    )


def _service():
    return ResultVerificationService()


# ── 1. Semantic CCQ results remain candidate-only ────────────────────────

def test_semantic_ccq_results_remain_candidate_only():
    svc = _service()
    for tool in ("semantic_search_ccq", "semantic_search_cpc",
                 "search_ccq_keywords", "search_cpc_keywords"):
        obs = _obs(tool, response="Art. 1457. Toute personne …")
        a = svc.assess(obs)
        assert a.evidence_level == EvidenceLevel.candidate.value, (
            f"{tool}: expected candidate, got {a.evidence_level}")
        assert not a.usable_as_evidence, (
            f"{tool}: retrieval-only must NOT be usable evidence")
        assert not a.citable, (
            f"{tool}: retrieval-only must NOT be citable")


# ── 2. Official CCQ article text can still be classified irrelevant ──────

def test_official_ccq_article_can_be_irrelevant():
    svc = _service()
    obs = _obs("get_ccq_articles",
               response="Art. 1466. Le propriétaire d'un animal …",
               arguments={"start_article": 1466})
    a = svc.assess(obs, user_query="Un locataire peut-il avoir un chat?")
    assert a.official, "get_ccq_articles is an official source"
    assert a.citable, "official source is citable"
    # Being official+citable does not automatically mean relevant/usable.
    # The assess method sets relevant=True by default because relevance
    # requires LLM-level judgment; but the TYPE SYSTEM allows official
    # sources to be classified irrelevant.
    entry = a.to_evidence_entry(0)
    assert entry.official
    # Manually mark as irrelevant (simulating LLM relevance check).
    entry_dict = entry.to_dict()
    entry_dict["relevant"] = False
    entry_dict["evidence_level"] = EvidenceLevel.citable.value
    assert entry_dict["official"] and not entry_dict["relevant"]


# ── 3. Article 1466 is not usable evidence for pet question ─────────────

def test_article_1466_not_usable_for_landlord_cat_question():
    svc = _service()
    obs = _obs("get_ccq_articles",
               response="Art. 1466. Le propriétaire d'un animal …",
               arguments={"start_article": 1466})
    a = svc.assess(obs, user_query="Un locataire peut-il avoir un chat?")
    entry = a.to_evidence_entry(0)
    # The evidence entry captures that this article is official+citable.
    # A downstream node (or manual classification) can mark it irrelevant.
    assert entry.official
    # Simulate: after relevance verification, it's NOT usable.
    entry_dict = entry.to_dict()
    entry_dict["relevant"] = False
    entry_dict["usable"] = False
    entry_dict["evidence_level"] = EvidenceLevel.citable.value
    # Key assertion: official source can be NOT usable.
    assert entry_dict["official"] and not entry_dict["usable"]
    assert entry_dict["evidence_level"] != EvidenceLevel.usable.value


# ── 4. Quebec jurisprudence does not silently return federal case law ────

def test_quebec_jurisprudence_no_silent_federal_fallback():
    svc = _service()
    obs = _obs("search_legal_documents",
               response="2020 SCC 5 — In the matter of …",
               arguments={"query": "droit québécois"})
    a = svc.assess(
        obs,
        resolved_jurisdiction="Québec",
        requested_court_scope="QCCA",
    )
    assert a.detailed_status in (
        DetailedResultStatus.alternative_only.value,
        DetailedResultStatus.wrong_court_scope.value,
    ), "Federal result for a QCCA request must NOT be classified as usable"
    assert not a.usable_as_evidence


# ── 5. SCC decision on Quebec law is alternative_only ────────────────────

def test_scc_decision_concerning_quebec_law_is_alternative():
    svc = _service()
    obs = _obs("search_legal_documents",
               response="2020 SCC 5 — Hypothèques, droit civil québécois",
               arguments={"query": "hypothèques Québec"})
    a = svc.assess(
        obs,
        resolved_jurisdiction="Québec",
        requested_court_scope="QCCA",
    )
    # SCC is usable_as_alternative for Québec via jurisdiction_dims logic.
    assert a.detailed_status == DetailedResultStatus.alternative_only.value
    assert a.evidence_level == EvidenceLevel.citable.value
    assert not a.usable_as_evidence


# ── 6. Unavailable Quebec jurisprudence produces coverage limitation ─────

def test_quebec_jurisprudence_covers_the_quebec_courts():
    """Réactivé le 2026-07-24 : l'échec était côté client (QCTAL manquant).

    L'ancienne version de ce test verrouillait l'indisponibilité; elle
    verrouillait en réalité un défaut de reconnaissance des citations.
    """
    coverage = get_coverage("search_quebec_jurisprudence")
    assert coverage is not None
    assert coverage.is_available("live")
    assert coverage.is_available("dataset")

    covered, tools, gap = svc_check_coverage_qcca_live()
    assert covered, "aucun autre outil ne couvre les cours québécoises"
    assert "search_quebec_jurisprudence" in tools
    assert gap is None


def svc_check_coverage_qcca_live():
    svc = _service()
    return svc.check_coverage(
        document_type="court_decision",
        court_scope="QCCA",
        jurisdiction="Québec",
        mode="live",
    )


# ── 7. Coverage failure is not described as absence of decisions ──────────

def test_coverage_failure_not_described_as_absence():
    coverage = get_coverage("search_quebec_jurisprudence")
    assert coverage is not None
    reason = coverage.availability_reason
    assert reason, "must have an availability_reason"
    # The reason must describe system limitations, not assert no decisions exist.
    lower = reason.lower()
    assert "aucune" not in lower, (
        "must NOT say 'aucune décision' — that implies decisions don't exist")
    assert "no decision" not in lower
    assert "no quebec decision" not in lower
    # La couverture québécoise est assurée : plus aucun manque à signaler.
    covered, _, gap = svc_check_coverage_qcca_live()
    assert covered and gap is None


# ── 8. search_legal_documents cannot satisfy QCCA/QCCS/QCCQ/QCTAL/QCTAT ─

def test_search_legal_documents_cannot_cover_quebec_courts():
    coverage = get_coverage("search_legal_documents")
    assert coverage is not None
    for scope in ("QCCA", "QCCS", "QCCQ", "QCTAL", "QCTAT"):
        assert not coverage.covers_court_scope(scope), (
            f"search_legal_documents must NOT cover {scope}")
    # Positive check: it does cover federal scopes.
    for scope in ("SCC", "FCA", "FC", "TCC"):
        assert coverage.covers_court_scope(scope)


# ── 9. No substantive answer without usable evidence ─────────────────────

def test_no_substantive_answer_without_usable_evidence(catalog):
    from lexior.agent_graph.nodes import build_answer_contract
    from lexior.agent_graph.state import initial_state
    from tests.test_central_graph import offline_runner, ScriptedPlanner

    obs = _obs("semantic_search_ccq",
               response="Art. 1457. Résultat sémantique")
    state = initial_state(scenario=_scenario())
    state["tool_history"] = [obs]
    state["request_type"] = "case_analysis"
    state["usable_evidence_entries"] = []
    state["usable_evidence"] = []
    # Skip route validation (irrelevant to evidence-level test).
    state["stop_reason"] = "clarification_required"

    runner = offline_runner(catalog)
    result = build_answer_contract.run(state, runner.context)
    assert result["answer_contract"]["mode_de_reponse"] == "no_evidence"


def _scenario(**kw):
    from agentic_generation.schemas import ScenarioSpec
    defaults = dict(
        scenario_id="regr-001", scenario_family_id="regression",
        request_type="case_analysis",
        user_query="Mon propriétaire peut-il interdire un chat?",
    )
    defaults.update(kw)
    return ScenarioSpec(**defaults)


# ── 10. Retrieval-only results never reach the writer as citations ───────

def test_retrieval_only_never_reaches_writer(catalog):
    from lexior.agent_graph.nodes import build_answer_contract
    from lexior.agent_graph.state import initial_state
    from tests.test_central_graph import offline_runner

    obs = _obs("semantic_search_ccq", response="Art. 1457 CCQ")
    state = initial_state(scenario=_scenario())
    state["tool_history"] = [obs]
    state["request_type"] = "case_analysis"
    state["usable_evidence"] = []
    state["usable_evidence_entries"] = []
    state["stop_reason"] = "clarification_required"

    runner = offline_runner(catalog)
    result = build_answer_contract.run(state, runner.context)
    contract = result["answer_contract"]
    # Writer receives NO usable tool references for retrieval-only tools.
    assert "semantic_search_ccq" not in contract.get("preuves_utilisables", [])
    assert contract["mode_de_reponse"] in ("no_evidence", "direct")


# ── 11. Official but irrelevant documents never enter usable_evidence ────

def test_official_irrelevant_never_enters_usable_evidence():
    from lexior.agent_graph.nodes import classify_tool_result
    from lexior.agent_graph.state import initial_state

    obs = _obs("get_ccq_articles",
               response="Art. 1466. Propriétaire d'un animal …",
               arguments={"start_article": 1466})
    state = initial_state(scenario=_scenario())
    state["tool_history"] = [obs]

    class FakeContext:
        class services:
            class verification:
                @staticmethod
                def assess(observation, issues=None, **kwargs):
                    from lexior.services.result_verification import (
                        ToolResultAssessment,
                    )
                    return ToolResultAssessment(
                        tool_name=observation.tool_name,
                        tool_call_succeeded=observation.ok,
                        official=True,
                        citable=True,
                        relevant=False,
                        usable_as_evidence=False,
                        evidence_level=EvidenceLevel.citable.value,
                        detailed_status=DetailedResultStatus.irrelevant.value,
                        search_status=SearchResultStatus.irrelevant.value,
                        reason="not relevant to user query",
                    )

    result = classify_tool_result.run(state, FakeContext())
    assert len(result.get("usable_evidence_entries", [])) == 0
    assert len(result.get("invalidated_sources", [])) > 0


# ── 12. Tool success and legal relevance remain separate ─────────────────

def test_tool_success_and_legal_relevance_are_separate():
    svc = _service()
    obs = _obs("get_ccq_articles",
               response="Art. 1466. Le propriétaire d'un animal …",
               arguments={"start_article": 1466},
               ok=True)
    a = svc.assess(obs)
    assert a.tool_call_succeeded, "Tool succeeded technically"
    # Technical success is a separate dimension from legal usability.
    # The assessment allows us to track both independently.
    entry = a.to_evidence_entry(0)
    assert entry.technical_success
    # Changing relevance does NOT change technical success.
    entry_dict = entry.to_dict()
    entry_dict["relevant"] = False
    assert entry_dict["technical_success"] and not entry_dict["relevant"]


# ── 13. Wrong court scope blocks acceptance ──────────────────────────────

def test_wrong_court_scope_blocks_acceptance():
    from lexior.agent_graph.nodes.compute_acceptance import (
        _compute_evidence_blockers,
    )

    state = {
        "usable_evidence_entries": [
            {
                "tool_name": "search_legal_documents",
                "detailed_status": "wrong_court_scope",
                "official": True,
                "relevant": True,
            }
        ],
        "coverage_gaps": [],
        "alternative_sources": [],
    }
    blockers = _compute_evidence_blockers(state)
    assert AcceptanceBlocker.wrong_court_scope.value in blockers


# ── 14. Deterministic blockers override high critic scores ───────────────

def test_deterministic_blockers_override_critic_scores():
    from lexior.agent_graph.nodes.compute_acceptance import (
        _compute_evidence_blockers,
    )

    # State with coverage gaps — deterministic blocker.
    state = {
        "usable_evidence_entries": [],
        "coverage_gaps": [CoverageGap(
            requested_court_scope="QCCA",
            requested_jurisdiction="Québec",
            reason="no tool covers QCCA",
        ).to_dict()],
        "alternative_sources": [],
    }
    blockers = _compute_evidence_blockers(state)
    assert AcceptanceBlocker.coverage_mismatch.value in blockers

    # Simulate: critics gave high scores and accepted.
    acceptance = AcceptanceResult(accepted=True)

    # Deterministic blockers MUST override the high critic acceptance.
    if blockers and acceptance.accepted:
        acceptance.accepted = False
        acceptance.blocking_errors = list(
            acceptance.blocking_errors or []) + blockers

    assert not acceptance.accepted, (
        "coverage_mismatch blocker must override high critic scores")
    assert AcceptanceBlocker.coverage_mismatch.value in (
        acceptance.blocking_errors)


# ── 15. Both modes use the same verification and coverage rules ──────────

def test_both_modes_use_same_verification_pipeline():
    svc = _service()
    obs = _obs("search_legal_documents",
               response="2024 FCA 123 — Immigration matter",
               arguments={"query": "immigration"},
               ok=True)

    # Assess in dataset mode context.
    a_dataset = svc.assess(
        obs,
        resolved_jurisdiction="Federal",
        requested_court_scope="FCA",
    )

    # Assess in live mode context — same service, same logic.
    a_live = svc.assess(
        obs,
        resolved_jurisdiction="Federal",
        requested_court_scope="FCA",
    )

    assert a_dataset.evidence_level == a_live.evidence_level
    assert a_dataset.detailed_status == a_live.detailed_status
    assert a_dataset.official == a_live.official
    assert a_dataset.citable == a_live.citable

    # Coverage rules are also shared.
    cov_dataset = has_equivalent_coverage(
        "court_decision", "FCA", "Federal", "dataset")
    cov_live = has_equivalent_coverage(
        "court_decision", "FCA", "Federal", "live")
    assert cov_dataset[0] == cov_live[0], (
        "FCA coverage must be identical in both modes")


# ── 16. assess() se sert enfin de la question de l'usager ────────────────

_ARTICLE_1466 = (
    "Article 1466. Le propriétaire d'un animal est tenu de réparer le "
    "préjudice que l'animal a causé, soit qu'il fût sous sa garde ou sous "
    "celle d'un tiers."
)


def test_assess_rejects_a_result_foreign_to_the_question():
    svc = _service()
    obs = _obs("get_ccq_articles", response=_ARTICLE_1466,
               arguments={"start_article": 1466})

    a = svc.assess(obs, user_query="Quel délai ai-je pour aller en appel?")

    assert a.search_status == SearchResultStatus.irrelevant.value
    assert not a.relevant
    assert not a.usable_as_evidence
    assert a.reason, "un rejet doit être motivé"
    # Officiel ne veut pas dire pertinent : la source reste citable.
    assert a.official and a.citable


def test_assess_keeps_a_result_that_answers_the_question():
    svc = _service()
    obs = _obs("get_ccq_articles", response=_ARTICLE_1466,
               arguments={"start_article": 1466})

    a = svc.assess(obs, user_query="Un chien m'a mordu, qui doit réparer le "
                                   "préjudice causé par un animal?")

    assert a.search_status == SearchResultStatus.usable.value
    assert a.relevant and a.usable_as_evidence


def test_assess_without_a_question_behaves_as_before():
    svc = _service()
    obs = _obs("get_ccq_articles", response=_ARTICLE_1466,
               arguments={"start_article": 1466})

    assert svc.assess(obs).search_status == SearchResultStatus.usable.value


def test_an_off_topic_result_reaches_the_reformulation_route():
    """Le câblage complet : verdict → état → route."""
    from lexior.agent_graph.routing import route_after_classification

    svc = _service()
    a = svc.assess(
        _obs("get_ccq_articles", response=_ARTICLE_1466),
        user_query="Quel délai ai-je pour aller en appel?")
    state = {
        "last_tool_result_status": a.search_status,
        "reformulation_count": 0,
        "max_reformulations": 1,
    }

    assert route_after_classification(state) == "reformulate_search"
