# -*- coding: utf-8 -*-
"""Machine à états contrôlée du pipeline agentique."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .agentic_critic import AgenticCritic
from .config import AgenticConfig
from .legal_critic import LegalCritic
from .mcp_executor import MCPExecutionError, MCPExecutor, MockMCPTransport
from .planner_agent import PlannerAgent
from .prompts import agent_system_prompt
from .response_verifier import verify_observation
from .schemas import (
    CriticResult, Decision, GenerationMetadata, GroundingEntry, Message, QualityReport,
    RejectionRecord, ResearchState, Role, ScenarioSpec, StateStatus,
    ToolCall, TrainingTrajectory,
)
from .tool_catalog import ToolCatalog
from .trajectory_agent import PRECISE_ARTICLE_TYPES, TrajectoryAgent
from .validators import ValidationResult, validate_tool_route, validate_trajectory


def _critic_failure_reasons(label: str, result, minimum: float) -> list[str]:
    """Produit toujours un motif exploitable pour un Critic sous le seuil."""
    reasons: list[str] = []
    if not result.accepted:
        reasons.append(f"{label}: décision rejected")
    if result.score < minimum:
        reasons.append(f"{label}: score {result.score:.2f} inférieur au seuil {minimum:.2f}")
    reasons.extend(f"{label}: {issue}" for issue in result.issues)
    reasons.extend(f"{label}: affirmation non étayée: {claim}"
                   for claim in result.unsupported_claims)
    reasons.extend(f"{label}: source manquante: {source}"
                   for source in result.missing_sources)
    return reasons


@dataclass
class OrchestrationResult:
    accepted: bool
    trajectory: Optional[TrainingTrajectory] = None
    rejection: Optional[RejectionRecord] = None
    validation: Optional[ValidationResult] = None


class AgenticOrchestrator:
    def __init__(self, config: AgenticConfig, catalog: ToolCatalog,
                 planner: PlannerAgent, executor: MCPExecutor,
                 trajectory_agent: TrajectoryAgent, legal_critic: LegalCritic,
                 agentic_critic: AgenticCritic,
                 progress: Optional[Callable[[str], None]] = None):
        self.config = config
        self.catalog = catalog
        self.planner = planner
        self.executor = executor
        self.trajectory_agent = trajectory_agent
        self.legal_critic = legal_critic
        self.agentic_critic = agentic_critic
        self.progress = progress or (lambda _message: None)
        self.seen_fingerprints: set[str] = set()

    def run(self, scenario: ScenarioSpec) -> OrchestrationResult:
        state = ResearchState(
            scenario=scenario,
            messages=[
                Message(role=Role.system, content=agent_system_prompt(self.catalog)),
                Message(role=Role.user, content=scenario.user_query),
            ],
            max_tool_calls=self.config.max_tool_calls,
            missing_critical_facts=list(scenario.facts_missing),
        )
        resolved_jurisdiction = ""
        clarification_count = 0
        max_planner_steps = state.max_tool_calls + 2
        try:
            while state.status == StateStatus.planning:
                if state.step >= max_planner_steps:
                    return self._reject(
                        scenario, "planner",
                        [f"limite de {max_planner_steps} décisions Planner atteinte"],
                    )
                self.progress(
                    f"Planner: décision {state.step + 1}/{max_planner_steps}..."
                )
                decision = self.planner.decide(state)
                self.progress(f"Planner: {decision.decision.value}")
                resolved_jurisdiction = decision.jurisdiction or resolved_jurisdiction
                state.step += 1
                state.missing_critical_facts = decision.missing_critical_facts

                if decision.decision == Decision.ask_clarification:
                    if clarification_count >= 1:
                        return self._reject(
                            scenario, "planner",
                            ["clarification répétée après la réponse de l'utilisateur"],
                        )
                    clarification_count += 1
                    state.messages.append(Message(role=Role.assistant,
                                                  content=decision.clarification_question or "Pouvez-vous préciser?"))
                    if scenario.clarification_answer:
                        state.messages.append(Message(role=Role.user, content=scenario.clarification_answer))
                        state.status = StateStatus.planning
                        continue
                    state.stop_reason = "clarification_required"
                    state.status = StateStatus.answering
                    break

                if decision.decision in {Decision.final_answer, Decision.cannot_conclude}:
                    state.stop_reason = decision.decision.value
                    state.status = StateStatus.answering
                    break

                if decision.decision != Decision.call_tool or not decision.next_tool:
                    return self._reject(scenario, "planner", ["décision inconnue ou incomplète"])
                if state.tool_calls_made() >= state.max_tool_calls:
                    state.stop_reason = "max_tool_calls"
                    state.status = StateStatus.answering
                    break

                call = ToolCall(name=decision.next_tool, arguments=decision.arguments)
                self.progress(f"MCP: appel {call.name}...")
                state.messages.append(Message(
                    role=Role.assistant,
                    thinking=decision.thinking_text or None,
                    content=f"<tool_call>\n{call.render()}\n</tool_call>"))
                state.status = StateStatus.waiting_tool
                if isinstance(self.executor.transport, MockMCPTransport):
                    if scenario.failure_mode == "panne_mcp" and not state.tool_history:
                        self.executor.transport.fail_next = MCPExecutionError("panne MCP simulée explicitement")
                    elif (scenario.failure_mode == "resultat_vide" and
                          call.name in {"semantic_search_ccq", "semantic_search_cpc"} and
                          sum(o.tool_name == call.name for o in state.tool_history) < 2):
                        self.executor.transport.empty_next = True
                observation = self.executor.execute(call)
                observation, verifier_issues = verify_observation(observation)
                if verifier_issues:
                    self.progress(f"Verifier: {'; '.join(verifier_issues)}")
                self.progress(
                    f"MCP: {call.name} terminé ({'ok' if observation.ok else 'erreur'})"
                )
                state.tool_history.append(observation)
                state.sources.extend(observation.source_urls)
                state.messages.append(Message(role=Role.tool, name=call.name,
                                              content=observation.normalized_response))
                state.status = StateStatus.planning

            route_errors = validate_tool_route(
                scenario.request_type,
                [observation.tool_name for observation in state.tool_history],
            )
            if route_errors:
                return self._reject(scenario, "planner", route_errors)

            self.progress("Trajectory Agent: rédaction finale...")
            final_thinking, answer = self.trajectory_agent.final_answer(state)
            if not (state.stop_reason == "clarification_required" and
                    state.messages[-1].role == Role.assistant):
                state.messages.append(Message(
                    role=Role.assistant,
                    thinking=final_thinking or None,
                    content=answer))
            else:
                answer = state.messages[-1].content

            self.progress("Critics: évaluation juridique et agentique...")
            if self.config.no_critics:
                legal = None
                agentic = None
            elif scenario.request_type in PRECISE_ARTICLE_TYPES:
                # La fidélité mot pour mot et la route mono-outil sont des
                # propriétés déterministes. Un juge LLM serait ici moins
                # fiable, plus coûteux et susceptible de réclamer à tort une
                # explication ou une mise en garde non demandée.
                legal = CriticResult(critic="legal", accepted=True, score=1.0)
                agentic = CriticResult(critic="agentic", accepted=True, score=1.0)
                self.progress("Critics: contrôles déterministes pour texte officiel précis.")
            else:
                legal = self.legal_critic.evaluate(state, answer)
                agentic = self.agentic_critic.evaluate(state, answer)
            repair_instructions = []
            if legal and (not legal.accepted or legal.score < self.config.legal_min_score):
                repair_instructions.extend(
                    legal.repair_instructions or legal.issues or
                    ["Rendre la réponse fidèle et suffisante dans la portée exacte de la question."]
                )
            if agentic and (not agentic.accepted or
                            agentic.score < self.config.agentic_min_score):
                repair_instructions.extend(
                    agentic.repair_instructions or agentic.issues or
                    ["Corriger la réponse sans ajouter de recherche ou de source absente."]
                )
            repaired = False
            if repair_instructions and self.config.max_repairs > 0:
                self.progress("Trajectory Agent: réparation de la réponse...")
                repaired_thinking, repaired_answer = self.trajectory_agent.repair(
                    state, answer, final_thinking, repair_instructions)
                if repaired_answer != answer:
                    state.messages[-1] = Message(
                        role=Role.assistant,
                        thinking=repaired_thinking or None,
                        content=repaired_answer)
                    answer = repaired_answer
                    final_thinking = repaired_thinking
                    repaired = True
                    legal = self.legal_critic.evaluate(state, answer)
                    agentic = self.agentic_critic.evaluate(state, answer)

            trajectory = TrainingTrajectory(
                scenario_id=scenario.scenario_id,
                scenario_family_id=scenario.scenario_family_id,
                language=scenario.language,
                request_type=scenario.request_type,
                legal_domain=scenario.legal_domain,
                expected_jurisdiction=scenario.expected_jurisdiction,
                resolved_jurisdiction=resolved_jurisdiction,
                messages=state.messages,
                tool_trace=state.tool_history,
                grounding=[GroundingEntry(tool_name=o.tool_name, content_hash=o.content_hash,
                                          source_urls=o.source_urls, citations=o.citations)
                           for o in state.tool_history
                           if not self.catalog.is_local(o.tool_name)],
                generation_metadata=GenerationMetadata(
                    teacher_model=self.config.teacher.model,
                    teacher_base_url_hash=self.config.teacher.base_url_hash,
                    critic_model=self.config.critic.model,
                    seed=self.config.seed,
                    prompt_version=self.config.prompt_version,
                    tool_catalog_hash=self.catalog.catalog_hash,
                ),
                quality=QualityReport(
                    legal_critic_score=legal.score if legal else None,
                    agentic_critic_score=agentic.score if agentic else None,
                    repaired=repaired,
                ),
            )
            validation = validate_trajectory(
                trajectory, self.catalog,
                allow_mock=self.config.offline or self.config.dry_run,
                max_tool_calls=state.max_tool_calls,
                seen_fingerprints=self.seen_fingerprints,
            )
            critics_ok = ((not legal or (legal.accepted and legal.score >= self.config.legal_min_score)) and
                          (not agentic or (agentic.accepted and agentic.score >= self.config.agentic_min_score)))
            trajectory.quality.deterministic_validation = validation.valid
            if not validation.valid or not critics_ok:
                reasons = list(validation.errors)
                if legal and (not legal.accepted or legal.score < self.config.legal_min_score):
                    reasons.extend(_critic_failure_reasons(
                        "legal_critic", legal, self.config.legal_min_score))
                if agentic and (not agentic.accepted or
                                agentic.score < self.config.agentic_min_score):
                    reasons.extend(_critic_failure_reasons(
                        "agentic_critic", agentic, self.config.agentic_min_score))
                return self._reject(scenario, "validator", reasons, trajectory, validation)
            state.status = StateStatus.accepted
            return OrchestrationResult(accepted=True, trajectory=trajectory, validation=validation)
        except Exception as exc:
            return self._reject(scenario, "orchestrator", [f"{type(exc).__name__}: {exc}"])

    @staticmethod
    def _reject(scenario: ScenarioSpec, stage: str, reasons: list[str],
                trajectory: Optional[TrainingTrajectory] = None,
                validation: Optional[ValidationResult] = None) -> OrchestrationResult:
        rejection = RejectionRecord(
            scenario_id=scenario.scenario_id,
            request_type=scenario.request_type,
            stage=stage,
            reasons=list(dict.fromkeys(reasons or ["rejet sans raison"])),
            trajectory=trajectory.model_dump(mode="json") if trajectory else None,
        )
        return OrchestrationResult(accepted=False, trajectory=trajectory,
                                   rejection=rejection, validation=validation)
