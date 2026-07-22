# -*- coding: utf-8 -*-
"""Graph node functions for the Lexior agent graph.

Each method reads from ``LexiorState``, performs one logical step,
and returns a partial state update dict.
"""

from __future__ import annotations

from typing import Any, Optional

from agentic_generation.agentic_critic import AgenticCritic
from agentic_generation.case_law_gate import gate_search_results
from agentic_generation.config import AgenticConfig
from agentic_generation.legal_critic import LegalCritic
from agentic_generation.mcp_executor import MCPExecutionError, MCPExecutor, MockMCPTransport
from agentic_generation.planner_agent import PlannerAgent
from agentic_generation.schemas import (
    CriticResult,
    Decision,
    GenerationMetadata,
    GroundingEntry,
    Message,
    PlannerDecision,
    RejectionDetail,
    RejectionRecord,
    RepairReport,
    Role,
    ToolCall,
)
from agentic_generation.storage import RunStorage
from agentic_generation.tool_catalog import ToolCatalog
from agentic_generation.trajectory_agent import PRECISE_ARTICLE_TYPES, TrajectoryAgent

from .state import LexiorState, to_quality_report, to_research_state, to_trajectory
from .step_verifier import ProposalVerdict, StepVerifier


class GraphNodes:
    """Node functions bound to pipeline dependencies."""

    def __init__(
        self,
        config: AgenticConfig,
        catalog: ToolCatalog,
        planner: PlannerAgent,
        executor: MCPExecutor,
        trajectory_agent: TrajectoryAgent,
        legal_critic: LegalCritic,
        agentic_critic: AgenticCritic,
        storage: Optional[RunStorage] = None,
    ):
        self.config = config
        self.catalog = catalog
        self.planner = planner
        self.executor = executor
        self.trajectory_agent = trajectory_agent
        self.legal_critic = legal_critic
        self.agentic_critic = agentic_critic
        self.storage = storage
        self.verifier = StepVerifier(catalog)
        self.seen_fingerprints: set[str] = set()

    # ── plan ─────────────────────────────────────────────────────────────

    def plan(self, state: LexiorState) -> dict[str, Any]:
        try:
            rs = to_research_state(state)
            decision = self.planner.decide(rs)
            import json as _json
            log_line = (
                f"[planner] etape {state.get('step', 0) + 1}: "
                f"{decision.decision.value}"
                + (f" -> {decision.next_tool} "
                   f"{_json.dumps(decision.arguments)[:120]}"
                   if decision.next_tool else "")
                + (f" | juridiction: {decision.jurisdiction}"
                   if decision.jurisdiction else "")
            )
            # Console Windows cp1252 : rester en ASCII pour ne jamais
            # faire échouer le noeud sur un simple log.
            print(log_line.encode("ascii", "backslashreplace")
                  .decode("ascii"), flush=True)
            return {
                "current_decision": decision.model_dump(mode="json"),
                "step": state.get("step", 0) + 1,
                "missing_critical_facts": decision.missing_critical_facts,
                "resolved_jurisdiction": (
                    decision.jurisdiction
                    or state.get("resolved_jurisdiction", "")
                ),
            }
        except Exception as exc:
            return {
                "status": "rejected",
                "stop_reason": f"planner: {type(exc).__name__}: {exc}",
            }

    # ── execute_tool ─────────────────────────────────────────────────────

    def execute_tool(self, state: LexiorState) -> dict[str, Any]:
        try:
            decision = PlannerDecision.model_validate(
                state["current_decision"])

            proposal = self.verifier.verify_proposal(
                decision,
                state["scenario"].request_type,
                state.get("tool_history", []),
                state.get("max_tool_calls", 4),
            )
            if proposal.verdict == ProposalVerdict.reject:
                return {
                    "status": "rejected",
                    "stop_reason": "; ".join(proposal.errors),
                }

            call = ToolCall(
                name=decision.next_tool, arguments=decision.arguments)

            self._inject_failure_mode(state, call)

            observation = self.executor.execute(call)
            observation, _ = self.verifier.validate_observation(observation)

            new_messages = list(state.get("messages", []))
            new_messages.append(Message(
                role=Role.assistant,
                thinking=decision.thinking_text or None,
                content=f"<tool_call>\n{call.render()}\n</tool_call>",
            ))
            new_messages.append(Message(
                role=Role.tool, name=call.name,
                content=observation.normalized_response,
            ))

            tool_history = list(state.get("tool_history", []))
            tool_history.append(observation)

            updates: dict[str, Any] = {
                "messages": new_messages,
                "tool_history": tool_history,
                "sources": list(state.get("sources", []))
                + list(observation.source_urls),
                "status": "planning",
            }

            if call.name in (
                "get_ccq_articles", "get_cpc_articles"
            ) and observation.ok:
                updates["official_rule_retrieved"] = True
                updates["official_rule_sources"] = list(
                    state.get("official_rule_sources", [])
                ) + [call.name]

            if call.name == "search_quebec_jurisprudence" and observation.ok:
                article_nums = [
                    str(o.arguments.get("start_article", ""))
                    for o in tool_history
                    if o.tool_name in (
                        "get_ccq_articles", "get_cpc_articles")
                    and o.ok and o.arguments.get("start_article")
                ]
                usable, status = gate_search_results(
                    observation.normalized_response,
                    article_nums,
                    state["scenario"].user_query,
                )
                updates["usable_case_sources"] = list(
                    state.get("usable_case_sources", [])
                ) + list(usable)
                updates["case_law_search_status"] = (
                    status.value if hasattr(status, "value")
                    else str(status))

            return updates
        except Exception as exc:
            return {
                "status": "rejected",
                "stop_reason": f"execute: {type(exc).__name__}: {exc}",
            }

    # ── handle_clarification ─────────────────────────────────────────────

    def handle_clarification(self, state: LexiorState) -> dict[str, Any]:
        decision = PlannerDecision.model_validate(state["current_decision"])
        scenario = state["scenario"]

        question = decision.clarification_question or "Pouvez-vous préciser?"

        new_messages = list(state.get("messages", []))
        new_messages.append(Message(role=Role.assistant, content=question))

        if state.get("mode") == "chat":
            # Chat réel : la question termine le tour; la réponse de
            # l'utilisateur arrivera au prochain message.
            return {
                "messages": new_messages,
                "clarification_count": state.get(
                    "clarification_count", 0) + 1,
                "final_answer": question,
                "status": "clarification",
                "stop_reason": "clarification_required",
            }

        answer = scenario.effective_clarification_answer
        if answer:
            new_messages.append(Message(role=Role.user, content=answer))

        return {
            "messages": new_messages,
            "clarification_count": state.get("clarification_count", 0) + 1,
            "status": "planning",
        }

    # ── generate_answer ──────────────────────────────────────────────────

    def generate_answer(self, state: LexiorState) -> dict[str, Any]:
        try:
            scenario = state["scenario"]
            tool_history = state.get("tool_history", [])

            exempt = StepVerifier.compute_exempt_tools(tool_history)
            is_chat = state.get("mode") == "chat"
            if not is_chat:
                route_errors = self.verifier.validate_tool_route(
                    scenario.request_type,
                    [o.tool_name for o in tool_history],
                    exempt_tools=exempt,
                )
                if route_errors:
                    return {
                        "status": "rejected",
                        "stop_reason": "; ".join(route_errors),
                        "exempt_tools": exempt,
                    }

            rs = to_research_state(state)
            stop = state.get("stop_reason", "")
            msgs = state.get("messages", [])

            if (stop == "clarification_required"
                    and msgs and msgs[-1].role == Role.assistant):
                answer = msgs[-1].content
                thinking = ""
                new_messages = list(msgs)
            else:
                thinking, answer = self.trajectory_agent.final_answer(rs)
                new_messages = list(msgs)
                new_messages.append(Message(
                    role=Role.assistant,
                    thinking=thinking or None,
                    content=answer,
                ))

            return {
                "messages": new_messages,
                "final_answer": answer,
                "final_thinking": thinking or "",
                "exempt_tools": exempt,
                "status": "answering",
            }
        except Exception as exc:
            return {
                "status": "rejected",
                "stop_reason": f"answer: {type(exc).__name__}: {exc}",
            }

    # ── run_critics ──────────────────────────────────────────────────────

    def run_critics(self, state: LexiorState) -> dict[str, Any]:
        scenario = state["scenario"]
        answer = state.get("final_answer", "")

        if self.config.no_critics:
            return {
                "legal_critic_result": None,
                "agentic_critic_result": None,
            }

        if scenario.request_type in PRECISE_ARTICLE_TYPES:
            perfect = CriticResult(
                critic="deterministic", accepted=True, score=1.0)
            return {
                "legal_critic_result": perfect,
                "agentic_critic_result": perfect,
            }

        rs = to_research_state(state)
        legal = self.legal_critic.evaluate(rs, answer)
        agentic = self.agentic_critic.evaluate(rs, answer)
        return {
            "legal_critic_result": legal,
            "agentic_critic_result": agentic,
        }

    # ── repair ───────────────────────────────────────────────────────────

    def repair(self, state: LexiorState) -> dict[str, Any]:
        legal = state.get("legal_critic_result")
        agentic = state.get("agentic_critic_result")
        answer = state.get("final_answer", "")
        thinking = state.get("final_thinking", "")

        instructions: list[str] = []
        if legal and (not legal.accepted
                      or legal.score < self.config.legal_min_score):
            instructions.extend(
                legal.repair_instructions or legal.issues
                or ["Rendre la réponse fidèle et suffisante."])
        if agentic and (not agentic.accepted
                        or agentic.score < self.config.agentic_min_score):
            instructions.extend(
                agentic.repair_instructions or agentic.issues
                or ["Corriger sans ajouter de source absente."])

        if not instructions:
            return {"repair_count": state.get("repair_count", 0) + 1}

        rs = to_research_state(state)
        rep_thinking, rep_answer = self.trajectory_agent.repair(
            rs, answer, thinking, instructions)

        messages = list(state.get("messages", []))
        if rep_answer != answer and messages:
            if messages[-1].role == Role.assistant:
                messages[-1] = Message(
                    role=Role.assistant,
                    thinking=rep_thinking or None,
                    content=rep_answer,
                )
            return {
                "messages": messages,
                "final_answer": rep_answer,
                "final_thinking": rep_thinking or "",
                "repair": RepairReport(
                    attempted=True, status="successful",
                    changes=list(instructions)),
                "repair_count": state.get("repair_count", 0) + 1,
            }

        return {
            "repair": RepairReport(
                attempted=True, status="failed",
                reason="la réparation n'a pas modifié la réponse"),
            "repair_count": state.get("repair_count", 0) + 1,
        }

    # ── validate_final ───────────────────────────────────────────────────

    def validate_final(self, state: LexiorState) -> dict[str, Any]:
        from agentic_generation.schemas import AcceptanceResult
        is_chat = state.get("mode") == "chat"

        traj = to_trajectory(state)
        exempt = state.get("exempt_tools", [])

        if is_chat:
            acceptance = AcceptanceResult(accepted=True)
        else:
            validation = self.verifier.validate_trajectory(
                traj,
                allow_mock=self.config.offline or self.config.dry_run,
                max_tool_calls=state.get("max_tool_calls", 4),
                seen_fingerprints=self.seen_fingerprints,
                exempt_tools=exempt,
            )
            traj.quality.deterministic_validation = validation.valid

            seq_warnings = self.verifier.validate_tool_sequence(
                state["scenario"].request_type,
                [o.tool_name for o in state.get("tool_history", [])],
            )
            validation.warnings.extend(seq_warnings)

            rs = to_research_state(state)
            acceptance = self.verifier.compute_acceptance(
                traj, validation,
                state.get("legal_critic_result"),
                state.get("agentic_critic_result"),
                legal_min_score=self.config.legal_min_score,
                agentic_min_score=self.config.agentic_min_score,
                state=rs,
            )

        if is_chat:
            return {"acceptance": acceptance}

        grounding = [
            GroundingEntry(
                tool_name=o.tool_name,
                content_hash=o.content_hash,
                source_urls=o.source_urls,
                citations=o.citations,
            )
            for o in state.get("tool_history", [])
            if not self.catalog.is_local(o.tool_name)
        ]

        first_invalid = self.verifier.find_first_invalid_step(
            state.get("tool_history", []),
            state["scenario"].request_type,
            exempt,
        )

        updates: dict[str, Any] = {
            "deterministic_validation": validation.valid,
            "acceptance": acceptance,
            "first_invalid_step": first_invalid,
            "grounding": grounding,
        }

        if not acceptance.accepted:
            repair = state.get("repair", RepairReport())
            updates["rejection_detail"] = RejectionDetail(
                scenario_id=state["scenario"].scenario_id,
                blocking_reason=(
                    acceptance.blocking_errors[0]
                    if acceptance.blocking_errors else ""),
                repair_attempted=repair.attempted,
                repair_successful=repair.status == "successful",
                first_invalid_step=first_invalid,
            )

        return updates

    # ── export ───────────────────────────────────────────────────────────

    def export(self, state: LexiorState) -> dict[str, Any]:
        traj = to_trajectory(state)
        traj.quality.acceptance = state.get("acceptance")
        traj.quality.accepted_for_intermediate = True
        if self.storage:
            self.storage.append_intermediate(traj)
        return {"status": "accepted"}

    # ── reject ───────────────────────────────────────────────────────────

    def reject(self, state: LexiorState) -> dict[str, Any]:
        scenario = state["scenario"]
        reason = state.get("stop_reason", "rejet sans raison")
        rejection = RejectionRecord(
            scenario_id=scenario.scenario_id,
            request_type=scenario.request_type,
            stage="graph",
            reasons=[reason],
        )
        if self.storage:
            self.storage.append_rejection(rejection)
        return {
            "status": "rejected",
            "rejection_detail": RejectionDetail(
                scenario_id=scenario.scenario_id,
                blocking_reason=reason,
            ),
        }

    # ── internal helpers ─────────────────────────────────────────────────

    def _inject_failure_mode(
        self, state: LexiorState, call: ToolCall,
    ) -> None:
        if not isinstance(self.executor.transport, MockMCPTransport):
            return
        scenario = state["scenario"]
        fm = scenario.effective_failure_mode
        history = state.get("tool_history", [])
        if fm == "tool_error" and not history:
            self.executor.transport.fail_next = MCPExecutionError(
                "panne MCP simulée")
        elif (fm == "empty_result"
              and call.name in {
                  "semantic_search_ccq", "semantic_search_cpc"}
              and sum(1 for o in history
                      if o.tool_name == call.name) < 2):
            self.executor.transport.empty_next = True
