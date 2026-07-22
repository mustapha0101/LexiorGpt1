# -*- coding: utf-8 -*-
"""Tests for the centralized acceptance logic and new validators."""

import pytest

from agentic_generation.acceptance import (
    _check_article_grounding,
    _check_clarification_consistency,
    _check_jurisdiction_consistency,
    _infer_jurisdiction_from_query,
    compute_acceptance_status,
)
from agentic_generation.schemas import (
    AcceptanceResult,
    CriticResult,
    GenerationMetadata,
    GroundingEntry,
    Message,
    MultiDimensionalScore,
    RepairReport,
    QualityReport,
    ResearchState,
    Role,
    ScenarioSpec,
    ToolObservation,
    TrainingTrajectory,
)
from agentic_generation.validators import (
    ValidationResult,
    classify_search_result,
    validate_jurisprudence_query,
    validate_tool_sequence_logic,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

def _scenario(**kw):
    defaults = dict(
        scenario_id="s1",
        scenario_family_id="f1",
        request_type="case_analysis",
        user_query="Quels sont mes recours pour un vice caché au Québec?",
        jurisdiction_status="supported_quebec",
        clarification_stage="none",
    )
    defaults.update(kw)
    return ScenarioSpec(**defaults)


def _trajectory(request_type="case_analysis", messages=None,
                tool_trace=None, grounding=None, **kw):
    return TrainingTrajectory(
        scenario_id="s1",
        scenario_family_id="f1",
        request_type=request_type,
        messages=messages or [
            Message(role=Role.user, content="Question?"),
            Message(role=Role.assistant, content="Réponse."),
        ],
        tool_trace=tool_trace or [],
        grounding=grounding or [],
        generation_metadata=GenerationMetadata(tool_catalog_hash="abc"),
        **kw,
    )


def _state(scenario=None, tool_history=None, messages=None):
    sc = scenario or _scenario()
    return ResearchState(
        scenario=sc,
        messages=messages or [
            Message(role=Role.user, content=sc.user_query),
        ],
        tool_history=tool_history or [],
    )


def _critic_ok(critic="legal", score=0.9):
    return CriticResult(
        critic=critic, accepted=True, score=score,
        dimensional_scores=MultiDimensionalScore(
            grounding_score=score,
            legal_accuracy_score=score,
            answer_quality_score=score,
        ),
    )


def _critic_fail(critic="legal", score=0.4, issues=None, labels=None):
    return CriticResult(
        critic=critic, accepted=False, score=score,
        issues=issues or ["problème détecté"],
        dimensional_scores=MultiDimensionalScore(
            labels=labels or [],
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# Test 1: CCQ query classified as Quebec
# ══════════════════════════════════════════════════════════════════════════

class TestJurisdictionInference:
    def test_ccq_is_quebec(self):
        assert _infer_jurisdiction_from_query(
            "Quel est le texte de l'article 1457 du Code civil du Québec?"
        ) == "supported_quebec"

    def test_federal_law_is_federal(self):
        assert _infer_jurisdiction_from_query(
            "Quelles sont les règles de la Loi sur la faillite fédérale?"
        ) == "supported_federal"

    def test_no_markers_is_undetermined(self):
        assert _infer_jurisdiction_from_query(
            "Bonjour, comment allez-vous?"
        ) == "undetermined"

    def test_other_province(self):
        assert _infer_jurisdiction_from_query(
            "Quels sont les droits en Ontario?"
        ) == "supported_other_canadian"

    def test_municipal(self):
        assert _infer_jurisdiction_from_query(
            "Quel règlement municipal s'applique?"
        ) == "municipal_coverage_uncertain"


# ══════════════════════════════════════════════════════════════════════════
# Test 2: Jurisdiction consistency validator
# ══════════════════════════════════════════════════════════════════════════

class TestJurisdictionConsistency:
    def test_ccq_marked_federal_is_mismatch(self):
        sc = _scenario(
            user_query="Article 1457 du Code civil du Québec",
            jurisdiction_status="supported_federal",
        )
        state = _state(scenario=sc)
        traj = _trajectory()
        issues = _check_jurisdiction_consistency(traj, state)
        assert any("jurisdiction_mismatch" in i for i in issues)

    def test_ccq_marked_quebec_is_ok(self):
        sc = _scenario(
            user_query="Article 1457 du Code civil du Québec",
            jurisdiction_status="supported_quebec",
        )
        state = _state(scenario=sc)
        traj = _trajectory()
        issues = _check_jurisdiction_consistency(traj, state)
        assert not issues


# ══════════════════════════════════════════════════════════════════════════
# Test 3: Clarification contradiction detected
# ══════════════════════════════════════════════════════════════════════════

class TestClarificationConsistency:
    def test_stage_none_with_answer_is_inconsistent(self):
        sc = _scenario(
            clarification_stage="none",
            synthetic_clarification_answer="Réponse synthétique",
        )
        state = _state(scenario=sc)
        traj = _trajectory()
        issues = _check_clarification_consistency(traj, state)
        assert any("clarification_inconsistency" in i for i in issues)

    def test_stage_none_without_answer_is_ok(self):
        sc = _scenario(clarification_stage="none")
        state = _state(scenario=sc)
        traj = _trajectory()
        issues = _check_clarification_consistency(traj, state)
        assert not issues

    def test_before_search_with_no_question_but_answer(self):
        sc = _scenario(
            clarification_stage="before_search",
            synthetic_clarification_answer="Ma réponse",
        )
        msgs = [Message(role=Role.user, content="Question?"),
                Message(role=Role.assistant, content="Réponse.")]
        state = _state(scenario=sc, messages=msgs)
        traj = _trajectory(messages=msgs)
        issues = _check_clarification_consistency(traj, state)
        assert any("clarification_inconsistency" in i for i in issues)


# ══════════════════════════════════════════════════════════════════════════
# Test 4: Article used but not retrieved
# ══════════════════════════════════════════════════════════════════════════

class TestArticleGrounding:
    def test_unretrieved_article_detected(self):
        traj = _trajectory(
            messages=[
                Message(role=Role.user, content="Question?"),
                Message(role=Role.assistant,
                        content="Selon l'article 2326, le délai est de 3 ans."),
            ],
            tool_trace=[],
        )
        issues = _check_article_grounding(traj)
        assert any("article 2326" in i for i in issues)

    def test_retrieved_article_is_ok(self):
        obs = ToolObservation(
            tool_name="get_ccq_articles", server="quebec",
            arguments={"start_article": 1457},
            raw_response="Article 1457. Toute personne...",
            normalized_response="Article 1457. Toute personne...",
            mock=True, ok=True,
        ).finalize_hash()
        traj = _trajectory(
            messages=[
                Message(role=Role.user, content="Question?"),
                Message(role=Role.assistant,
                        content="L'article 1457 prévoit que toute personne..."),
            ],
            tool_trace=[obs],
        )
        issues = _check_article_grounding(traj)
        assert not issues


# ══════════════════════════════════════════════════════════════════════════
# Test 5: Jurisprudence returning a law → wrong_document_type
# ══════════════════════════════════════════════════════════════════════════

class TestSearchResultClassification:
    def test_jurisprudence_with_law_is_wrong_type(self):
        result = classify_search_result(
            "search_quebec_jurisprudence",
            "Loi sur la protection du consommateur, RLRQ c P-40.1",
            ok=True,
        )
        assert result == "wrong_document_type"

    def test_jurisprudence_with_case_is_usable(self):
        result = classify_search_result(
            "search_quebec_jurisprudence",
            "Dupont c. Martin, 2023 QCCS 1234. Le tribunal a conclu...",
            ok=True,
        )
        assert result == "usable"

    def test_empty_result(self):
        assert classify_search_result("search_quebec_jurisprudence", "", True) == "empty"
        assert classify_search_result("search_quebec_jurisprudence", "[]", True) == "empty"

    def test_tool_error(self):
        assert classify_search_result(
            "search_quebec_jurisprudence", "", False, "timeout") == "tool_error"


# ══════════════════════════════════════════════════════════════════════════
# Test 6: Single reformulation allowed
# ══════════════════════════════════════════════════════════════════════════

class TestJurisprudenceQueryQuality:
    def test_good_query_passes(self):
        warnings = validate_jurisprudence_query(
            "article 1458 CCQ responsabilité contractuelle inexécution "
            "obligation dommages Québec")
        assert not warnings

    def test_generic_query_warned(self):
        warnings = validate_jurisprudence_query(
            "aimerais comprendre comment ça marche svp merci bonjour")
        assert any("conversationnelle" in w for w in warnings)

    def test_empty_query_warned(self):
        warnings = validate_jurisprudence_query("")
        assert any("courte" in w for w in warnings)


# ══════════════════════════════════════════════════════════════════════════
# Test 7: Optional jurisprudence absence → accepted
# ══════════════════════════════════════════════════════════════════════════

class TestOptionalJurisprudenceAccepted:
    def test_case_analysis_no_jurisprudence_accepted(self):
        obs = ToolObservation(
            tool_name="get_ccq_articles", server="quebec",
            arguments={"start_article": 1457},
            raw_response="Article 1457. Toute personne...",
            normalized_response="Article 1457. Toute personne...",
            mock=True, ok=True,
        ).finalize_hash()
        traj = _trajectory(
            request_type="case_analysis",
            messages=[
                Message(role=Role.user,
                        content="Vice caché maison Québec?"),
                Message(role=Role.assistant,
                        content="L'article 1457 prévoit que toute personne..."),
            ],
            tool_trace=[obs],
            grounding=[GroundingEntry(
                tool_name="get_ccq_articles",
                content_hash=obs.content_hash)],
        )
        # Validation has a jurisprudence warning but it's optional
        validation = ValidationResult(valid=True, errors=[], warnings=[])
        legal = _critic_ok("legal", 0.85)
        agentic = _critic_ok("agentic", 0.80)

        result = compute_acceptance_status(
            traj, validation, legal, agentic,
            legal_min_score=0.7, agentic_min_score=0.7,
        )
        assert result.accepted


# ══════════════════════════════════════════════════════════════════════════
# Test 8: Perfect scores + validations → accepted
# ══════════════════════════════════════════════════════════════════════════

class TestPerfectScoresAccepted:
    def test_perfect_scores_accepted(self):
        traj = _trajectory()
        validation = ValidationResult(valid=True, errors=[], warnings=[])
        legal = _critic_ok("legal", 1.0)
        agentic = _critic_ok("agentic", 1.0)

        result = compute_acceptance_status(
            traj, validation, legal, agentic)
        assert result.accepted
        assert not result.blocking_errors

    def test_absent_critics_not_blocking(self):
        traj = _trajectory()
        validation = ValidationResult(valid=True, errors=[], warnings=[])

        result = compute_acceptance_status(
            traj, validation, None, None)
        assert result.accepted


# ══════════════════════════════════════════════════════════════════════════
# Test 9: Rejection with explicit reason
# ══════════════════════════════════════════════════════════════════════════

class TestRejectionWithReason:
    def test_blocking_error_has_reason(self):
        traj = _trajectory()
        validation = ValidationResult(
            valid=False,
            errors=["réponse finale vide"],
        )
        result = compute_acceptance_status(
            traj, validation, _critic_ok(), _critic_ok())
        assert not result.accepted
        assert "réponse finale vide" in result.blocking_errors

    def test_hard_critic_label_blocks(self):
        traj = _trajectory()
        validation = ValidationResult(valid=True, errors=[])
        legal = _critic_fail(
            "legal", score=0.5,
            labels=["unsupported_claim"],
        )
        result = compute_acceptance_status(
            traj, validation, legal, _critic_ok("agentic"))
        assert not result.accepted
        assert any("legal_critic" in r for r in result.blocking_errors)


# ══════════════════════════════════════════════════════════════════════════
# Test 10: Repair status coherence
# ══════════════════════════════════════════════════════════════════════════

class TestRepairCoherence:
    def test_repair_report_model(self):
        repair = RepairReport(
            attempted=True,
            status="successful",
            reason="score amélioré",
            changes=["fidélité aux sources"],
        )
        assert repair.attempted
        assert repair.status == "successful"

    def test_quality_report_coherence(self):
        quality = QualityReport(
            deterministic_validation=True,
            legal_critic_score=1.0,
            agentic_critic_score=1.0,
            repair=RepairReport(attempted=True, status="successful"),
            repaired=True,
            repair_status="successful",
        )
        assert quality.repair.attempted
        assert quality.repair.status == "successful"
        assert quality.repaired is True
        assert quality.repair_status == "successful"

    def test_no_repair_coherence(self):
        quality = QualityReport(
            repair=RepairReport(),
            repaired=False,
            repair_status="not_needed",
        )
        assert not quality.repair.attempted
        assert quality.repair.status == "not_needed"
        assert not quality.repaired


# ══════════════════════════════════════════════════════════════════════════
# Test 11: Tool sequence logic
# ══════════════════════════════════════════════════════════════════════════

class TestToolSequenceLogic:
    def test_jurisprudence_before_article_warned(self):
        warnings = validate_tool_sequence_logic(
            "case_analysis",
            ["search_quebec_jurisprudence", "get_ccq_articles"],
        )
        assert any("jurisprudence recherchée avant" in w for w in warnings)

    def test_article_before_jurisprudence_ok(self):
        warnings = validate_tool_sequence_logic(
            "case_analysis",
            ["get_ccq_articles", "search_quebec_jurisprudence"],
        )
        assert not warnings

    def test_case_law_research_no_warning(self):
        warnings = validate_tool_sequence_logic(
            "case_law_research",
            ["search_quebec_jurisprudence"],
        )
        assert not warnings


# ══════════════════════════════════════════════════════════════════════════
# Test 12: Non-blocking issues don't cause rejection
# ══════════════════════════════════════════════════════════════════════════

class TestNonBlockingWarnings:
    def test_thinking_too_long_is_warning(self):
        traj = _trajectory()
        validation = ValidationResult(
            valid=False,
            errors=["thinking_too_long"],
        )
        result = compute_acceptance_status(
            traj, validation, _critic_ok(), _critic_ok())
        assert result.accepted
        assert "thinking_too_long" in result.warnings

    def test_soft_critic_issues_dont_block(self):
        traj = _trajectory()
        validation = ValidationResult(valid=True, errors=[])
        agentic = _critic_fail(
            "agentic", score=0.5,
            issues=["le style pourrait être amélioré"],
            labels=[],
        )
        agentic.hard_failures = []
        result = compute_acceptance_status(
            traj, validation, _critic_ok("legal"), agentic)
        # Soft critic issues without hard labels go to warnings
        assert result.accepted
