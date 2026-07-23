# -*- coding: utf-8 -*-
"""Service de validation déterministe + acceptation.

Point d'entrée unique du graphe vers ``StepVerifier`` (qui consolide
``validators.py``, ``acceptance.py`` et ``response_verifier.py``). Les
empreintes anti-doublon (``seen_fingerprints``) vivent ici, au niveau du
run, plus dans aucun moteur.
"""

from __future__ import annotations

from typing import Optional

from agentic_generation.schemas import (
    AcceptanceResult,
    CriticResult,
    PlannerDecision,
    ResearchState,
    ToolObservation,
    TrainingTrajectory,
)
from agentic_generation.tool_catalog import ToolCatalog
from agentic_generation.validators import ValidationResult
from lexior.agent_graph.step_verifier import (
    ProposalVerdict,
    StepVerifier,
    VerifiedProposal,
)


class ValidationService:
    def __init__(self, catalog: ToolCatalog):
        self.catalog = catalog
        self.verifier = StepVerifier(catalog)
        self.seen_fingerprints: set[str] = set()

    # ── Autorisation d'une proposition du planner ────────────────────────

    def verify_proposal(self, decision: PlannerDecision, request_type: str,
                        tool_history: list[ToolObservation],
                        max_tool_calls: int) -> VerifiedProposal:
        return self.verifier.verify_proposal(
            decision, request_type, tool_history, max_tool_calls)

    # ── Route et séquence ────────────────────────────────────────────────

    def validate_tool_route(self, request_type: str, sequence: list[str],
                            exempt_tools: Optional[list[str]] = None,
                            ) -> list[str]:
        return self.verifier.validate_tool_route(
            request_type, sequence, exempt_tools)

    def sequence_warnings(self, request_type: str,
                          sequence: list[str]) -> list[str]:
        return self.verifier.validate_tool_sequence(request_type, sequence)

    def compute_exempt_tools(
            self, tool_history: list[ToolObservation]) -> list[str]:
        return StepVerifier.compute_exempt_tools(tool_history)

    # ── Validation finale ────────────────────────────────────────────────

    def validate_trajectory(self, trajectory: TrainingTrajectory, *,
                            allow_mock: bool, max_tool_calls: int,
                            exempt_tools: Optional[list[str]] = None,
                            ) -> ValidationResult:
        return self.verifier.validate_trajectory(
            trajectory,
            allow_mock=allow_mock,
            max_tool_calls=max_tool_calls,
            seen_fingerprints=self.seen_fingerprints,
            exempt_tools=exempt_tools,
        )

    def compute_acceptance(self, trajectory: TrainingTrajectory,
                           validation: ValidationResult,
                           legal: Optional[CriticResult],
                           agentic: Optional[CriticResult], *,
                           legal_min_score: float, agentic_min_score: float,
                           state: Optional[ResearchState] = None,
                           ) -> AcceptanceResult:
        return self.verifier.compute_acceptance(
            trajectory, validation, legal, agentic,
            legal_min_score=legal_min_score,
            agentic_min_score=agentic_min_score,
            state=state,
        )

    def find_first_invalid_step(self, tool_history: list[ToolObservation],
                                request_type: str,
                                exempt_tools: Optional[list[str]] = None,
                                ) -> Optional[int]:
        return self.verifier.find_first_invalid_step(
            tool_history, request_type, exempt_tools)


__all__ = [
    "ValidationService",
    "ValidationResult",
    "ProposalVerdict",
    "VerifiedProposal",
]
