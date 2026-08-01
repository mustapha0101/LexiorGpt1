# -*- coding: utf-8 -*-
"""compute_acceptance — LA décision accepter/rejeter, avec raisons.

Dataset : règles complètes (validation déterministe + seuils des
critiques + cohérence juridiction/clarification/grounding) +
typed deterministic blockers from evidence classification.
Live : une réponse non vide est livrée seulement en l'absence de bloqueur
déterministe, notamment une affirmation juridique non fondée.

Deterministic blockers CANNOT be overridden by LLM critic scores.
"""

from __future__ import annotations

from typing import Any

from lexior.agentic.schemas import (
    AcceptanceResult,
    GroundingEntry,
    RejectionDetail,
    RepairReport,
)
from lexior.services.evidence import AcceptanceBlocker
from lexior.services.modes import is_live
from lexior.services.result_verification import ResultVerificationService
from lexior.services.evidence_first import merge_failures
from lexior.agentic.schemas import ClaimLedger

from ..context import GraphContext
from ..state import LexiorState, to_research_state, to_trajectory

NAME = "compute_acceptance"


def _compute_evidence_blockers(state: LexiorState) -> list[str]:
    """Compute typed deterministic blockers from evidence state.

    These cannot be overridden by high critic scores.
    """
    blockers: list[str] = []

    # Check for retrieval-only sources used as evidence.
    usable_entries = state.get("usable_evidence_entries", [])
    for entry in usable_entries:
        tool_name = entry.get("tool_name", "")
        if ResultVerificationService.is_retrieval_only(tool_name):
            blockers.append(
                AcceptanceBlocker.retrieval_only_source_used_as_evidence.value)
            break

    # Check for irrelevant official sources in usable evidence.
    for entry in usable_entries:
        if entry.get("official") and not entry.get("relevant"):
            blockers.append(
                AcceptanceBlocker.irrelevant_official_source.value)
            break

    # Check for wrong court scope.
    for entry in usable_entries:
        if entry.get("detailed_status") == "wrong_court_scope":
            blockers.append(AcceptanceBlocker.wrong_court_scope.value)
            break

    # Check for wrong source jurisdiction.
    for entry in usable_entries:
        if entry.get("detailed_status") == "wrong_jurisdiction":
            blockers.append(
                AcceptanceBlocker.wrong_source_jurisdiction.value)
            break

    # Coverage gaps.
    coverage_gaps = state.get("coverage_gaps", [])
    if coverage_gaps:
        blockers.append(AcceptanceBlocker.coverage_mismatch.value)

    # Alternative-only sources presented without labeling.
    alternative_sources = state.get("alternative_sources", [])
    if alternative_sources and not usable_entries:
        contract = state.get("answer_contract") or {}
        if not contract.get("sources_alternatives"):
            blockers.append(
                AcceptanceBlocker.silent_non_equivalent_fallback.value)

    return blockers


def _open_grounding_failures(state: LexiorState) -> list[str]:
    failures = [item for item in state.get("failure_history", [])
                if str(item.get("status", "open")) == "open"]
    failures.extend(item for item in state.get("grounding_failures", [])
                    if str(item.get("status", "open")) == "open")
    return list(dict.fromkeys(
        "unsupported_legal_claim" for item in failures
        if item.get("failure_type") in {
            "ungrounded_claim", "ungrounded_article", "unsupported_legal_claim"
        }))


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    live = is_live(state.get("mode", ""))

    if live:
        answer = (state.get("final_answer") or "").strip()
        blockers = list(dict.fromkeys(
            list(state.get("deterministic_blockers", []))
            + list(state.get("acceptance_blockers", []))
            + _compute_evidence_blockers(state)))
        acceptance = AcceptanceResult(accepted=bool(answer) and not blockers)
        if not answer:
            acceptance.blocking_errors = ["réponse finale vide"]
        elif blockers:
            acceptance.blocking_errors = blockers
        ledger = state.get("claim_ledger")
        if isinstance(ledger, dict):
            ledger = ClaimLedger.model_validate(ledger)
        claim_blockers = [
            "unsupported_legal_claim" for claim in getattr(ledger, "claims", [])
            if claim.verification_status == "failed"
        ]
        if claim_blockers:
            acceptance.accepted = False
            acceptance.blocking_errors = list(dict.fromkeys(
                [*acceptance.blocking_errors, *claim_blockers]))
        open_failures = _open_grounding_failures(state)
        if open_failures:
            acceptance.accepted = False
            acceptance.blocking_errors = list(dict.fromkeys(
                [*acceptance.blocking_errors, *open_failures]))
        return {"acceptance_result": acceptance,
                "acceptance_blockers": list(dict.fromkeys(
                    [*blockers, *claim_blockers, *open_failures])),
                "failure_history": merge_failures(
                    state.get("failure_history", []),
                    [{"failure_type": item, "reason": item}
                     for item in claim_blockers], node=NAME),
                "quality_accepted": acceptance.accepted,
                "trajectory_accepted": acceptance.accepted}

    critics = state.get("critic_results", {}) or {}
    trajectory = to_trajectory(state)
    validation = state.get("validation_result")

    acceptance = ctx.services.validation.compute_acceptance(
        trajectory, validation,
        critics.get("legal"), critics.get("agentic"),
        legal_min_score=ctx.config.legal_min_score,
        agentic_min_score=ctx.config.agentic_min_score,
        state=to_research_state(state),
    )

    # Typed deterministic blockers — override critic scores.
    evidence_blockers = _compute_evidence_blockers(state)
    existing_blockers = list(state.get("acceptance_blockers", []))
    all_blockers = list(dict.fromkeys(existing_blockers + evidence_blockers))
    all_blockers.extend(_open_grounding_failures(state))
    all_blockers = list(dict.fromkeys(all_blockers))

    if all_blockers and acceptance.accepted:
        acceptance.accepted = False
        acceptance.blocking_errors = list(
            acceptance.blocking_errors or []) + all_blockers

    grounding = [
        GroundingEntry(
            tool_name=o.tool_name,
            content_hash=o.content_hash,
            source_urls=o.source_urls,
            citations=o.citations,
        )
        for o in state.get("tool_history", [])
        if not ctx.catalog.is_local(o.tool_name)
    ]

    first_invalid = ctx.services.validation.find_first_invalid_step(
        state.get("tool_history", []),
        state["scenario"].request_type,
        state.get("exempt_tools", []),
    )

    updates: dict[str, Any] = {
        "acceptance_result": acceptance,
        "grounding": grounding,
        "first_invalid_step": first_invalid,
        "acceptance_blockers": all_blockers,
        "quality_accepted": acceptance.accepted,
        "trajectory_accepted": acceptance.accepted,
    }

    if not acceptance.accepted:
        repair = state.get("repair", RepairReport())
        updates["rejection_detail"] = RejectionDetail(
            scenario_id=state["scenario"].scenario_id,
            blocking_reason=(acceptance.blocking_errors[0]
                             if acceptance.blocking_errors else ""),
            repair_attempted=repair.attempted,
            repair_successful=repair.status == "successful",
            first_invalid_step=first_invalid,
        )

    return updates
