# -*- coding: utf-8 -*-
"""GraphRunner — l'unique point d'entrée d'exécution des deux modes.

    dataset : run_dataset(scenario)  → DatasetRunResult
    live    : stream_live(...)       → événements SSE (générateur)
              resume_live(...)       → reprise après interrupt()

Un seul graphe compilé, un seul contexte de services. L'ancien
``AgenticOrchestrator`` n'est plus qu'une façade au-dessus de
``run_dataset``.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from langgraph.types import Command

from lexior.agentic.prompts import agent_system_prompt
from lexior.agentic.schemas import (
    RejectionRecord,
    ScenarioSpec,
    TrainingTrajectory,
)
from lexior.agentic.validators import ValidationResult

from .checkpointing import create_memory_checkpointer
from .context import GraphContext
from .events import NODE_LABELS, StreamTranslator, extract_interrupt_question
from .graph import build_graph
from .state import initial_state


@dataclass
class DatasetRunResult:
    """Résultat d'un run dataset — même contrat que l'orchestrateur."""

    accepted: bool
    trajectory: Optional[TrainingTrajectory] = None
    rejection: Optional[RejectionRecord] = None
    validation: Optional[ValidationResult] = None
    final_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveTurnResult:
    """Résultat d'un tour live (réponse OU clarification en attente)."""

    thread_id: str
    final_answer: str = ""
    reasoning: str = ""
    sources: list[str] = field(default_factory=list)
    pending_question: Optional[str] = None
    status: str = ""
    final_state: dict[str, Any] = field(default_factory=dict)


class GraphRunner:
    """Compile LE graphe une fois; sert les deux modes."""

    def __init__(self, context: GraphContext, checkpointer=None):
        self.context = context
        self.checkpointer = checkpointer or create_memory_checkpointer()
        self.graph = build_graph(self.context, checkpointer=self.checkpointer)

    # ── Aides communes ───────────────────────────────────────────────────

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": 80}

    def system_prompt(self) -> str:
        return agent_system_prompt(self.context.catalog)

    def has_pending_interrupt(self, thread_id: str) -> bool:
        try:
            snapshot = self.graph.get_state(self._config(thread_id))
        except Exception:
            return False
        return bool(snapshot and snapshot.next)

    # ── Mode dataset ─────────────────────────────────────────────────────

    def run_dataset(
        self,
        scenario: ScenarioSpec,
        *,
        progress: Optional[Callable[[str], None]] = None,
        thread_id: Optional[str] = None,
    ) -> DatasetRunResult:
        """ScenarioSpec → graphe central → trajectoire validée."""
        thread_id = thread_id or f"dataset-{scenario.scenario_id}-{uuid.uuid4().hex[:6]}"
        state = initial_state(
            scenario,
            mode="dataset",
            max_tool_calls=self.context.config.max_tool_calls,
            system_prompt=self.system_prompt(),
            thread_id=thread_id,
            max_reformulations=self.context.max_reformulations,
            max_repairs=self.context.max_repairs,
        )

        notify = progress or (lambda _message: None)
        final: dict[str, Any] = {}
        try:
            for chunk in self.graph.stream(
                    state, config=self._config(thread_id),
                    stream_mode=["updates", "values"]):
                stream_type, payload = chunk
                if stream_type == "updates" and isinstance(payload, dict):
                    for node_name in payload:
                        if node_name != "__interrupt__":
                            notify(NODE_LABELS.get(node_name, node_name))
                elif stream_type == "values" and isinstance(payload, dict):
                    final = payload
        finally:
            self._forget_thread(thread_id)

        return self._to_dataset_result(scenario, final)

    def _to_dataset_result(self, scenario: ScenarioSpec,
                           final: dict[str, Any]) -> DatasetRunResult:
        accepted = final.get("status") == "accepted"
        trajectory = None
        if final.get("trajectory"):
            trajectory = TrainingTrajectory.model_validate(
                final["trajectory"])

        rejection = None
        if not accepted:
            export = final.get("export_result") or {}
            raw = export.get("rejection") if isinstance(export, dict) else None
            if raw:
                rejection = RejectionRecord.model_validate(raw)
            else:
                rejection = RejectionRecord(
                    scenario_id=scenario.scenario_id,
                    request_type=scenario.request_type,
                    stage="graph",
                    reasons=[final.get("stop_reason")
                             or "rejet sans raison"],
                )

        return DatasetRunResult(
            accepted=accepted,
            trajectory=trajectory,
            rejection=rejection,
            validation=final.get("validation_result"),
            final_state=final,
        )

    def _forget_thread(self, thread_id: str) -> None:
        """Purge le thread éphémère d'un run dataset (best effort)."""
        delete = getattr(self.checkpointer, "delete_thread", None)
        if callable(delete):
            try:
                delete(thread_id)
            except Exception:
                pass

    # ── Mode live ────────────────────────────────────────────────────────

    def build_live_state(self, query: str, *, thread_id: str,
                         history: Optional[list[dict]] = None,
                         system_prompt: Optional[str] = None,
                         request_type: str = "case_analysis",
                         prior_case_context: Optional[dict] = None) -> dict:
        """État initial d'un tour live (historique client inclus)."""
        from lexior.agentic.schemas import Message, Role

        scenario = ScenarioSpec(
            scenario_id=f"live-{uuid.uuid4().hex[:8]}",
            scenario_family_id="live",
            request_type=request_type,
            language="fr",
            user_query=query,
            jurisdiction="",
        )
        state = initial_state(
            scenario,
            mode="live",
            max_tool_calls=getattr(self.context.config, "max_tool_calls_live",
                                   self.context.config.max_tool_calls),
            system_prompt=system_prompt or self.system_prompt(),
            thread_id=thread_id,
            max_reformulations=self.context.max_reformulations,
            max_repairs=self.context.max_repairs,
            max_clarifications=getattr(
                self.context.config, "max_clarifications_live", 2),
            max_planner_decisions=getattr(
                self.context.config, "max_planner_decisions_live", 12),
            max_search_reformulations=getattr(
                self.context.config, "max_search_reformulations_live",
                self.context.max_reformulations),
        )
        if prior_case_context:
            context = deepcopy(prior_case_context)
            state.update({
                "case_context": context,
                "prior_evidence": list(context.get("prior_evidence", [])),
                "article_reviews": dict(context.get("article_reviews", {})),
                "clarification_history": list(
                    context.get("clarification_history", [])),
                "active_issue": context.get("active_issue", ""),
                "facts": dict(context.get("facts", {})),
                "missing_facts_before_application": list(
                    context.get("missing_facts_before_application", [])),
                "resolved_jurisdiction": context.get(
                    "resolved_jurisdiction", ""),
                "jurisdiction_status": context.get(
                    "jurisdiction_status", "unknown"),
                "jurisdiction_locked": bool(context.get(
                    "jurisdiction_locked", False)),
                "jurisdiction_verified": bool(context.get(
                    "jurisdiction_verified", False)),
                "jurisdiction_basis": context.get("jurisdiction_basis", ""),
                "official_rule_retrieved": bool(context.get(
                    "official_rule_retrieved", False)),
                "official_rule_sources": list(context.get(
                    "official_rule_sources", [])),
                "usable_case_sources": list(context.get(
                    "usable_case_sources", [])),
                "case_law_search_status": context.get(
                    "case_law_search_status", "not_required"),
                "case_law_verified": list(context.get(
                    "case_law_verified", [])),
                "pending_clarification": dict(context.get(
                    "pending_clarification", {})),
                "legislative_sufficiency": dict(context.get(
                    "legislative_sufficiency", {})),
                "max_clarifications": getattr(
                    self.context.config, "max_clarifications_live", 2),
                "max_planner_decisions": getattr(
                    self.context.config, "max_planner_decisions_live", 12),
                "max_search_reformulations": getattr(
                    self.context.config, "max_search_reformulations_live",
                    self.context.max_reformulations),
                "clarification_count": len(context.get(
                    "clarification_history", [])),
            })
        if history:
            turns = []
            for turn in history:
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                role = (Role.user if turn.get("role") == "user"
                        else Role.assistant)
                turns.append(Message(role=role, content=content))
            if turns:
                messages = state["messages"]
                state["messages"] = messages[:-1] + turns + messages[-1:]
        return state

    def _prior_case_context(self, config: dict) -> dict:
        """Lit uniquement le contexte persistant du dernier checkpoint."""
        try:
            snapshot = self.graph.get_state(config)
            values = getattr(snapshot, "values", {}) or {}
            context = values.get("case_context", {})
            return deepcopy(context) if isinstance(context, dict) else {}
        except (AttributeError, KeyError, TypeError, ValueError):
            # Checkpointer absent ou thread encore inexistant.
            return {}

    def _checkpoint_tool_count(self, config: dict) -> int:
        """Nombre d'observations déjà diffusées avant une reprise suspendue."""
        try:
            snapshot = self.graph.get_state(config)
            values = getattr(snapshot, "values", {}) or {}
            history = values.get("tool_history", [])
            return len(history) if isinstance(history, list) else 0
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0

    def stream_live(self, query: str, *, thread_id: Optional[str] = None,
                    history: Optional[list[dict]] = None,
                    system_prompt: Optional[str] = None,
                    ) -> Iterator[dict[str, Any]]:
        """Tour live streamé : événements SSE produits DEPUIS le graphe.

        Si le thread porte une clarification en attente, le message
        reprend le graphe via ``Command(resume=...)``; sinon un nouveau
        tour démarre sur le même thread.
        """
        thread_id = thread_id or f"live-{uuid.uuid4().hex[:8]}"
        config = self._config(thread_id)

        pending_interrupt = self.has_pending_interrupt(thread_id)
        prior_tool_count = self._checkpoint_tool_count(config) \
            if pending_interrupt else 0
        if pending_interrupt:
            payload: Any = Command(resume=query)
        else:
            payload = self.build_live_state(
                query, thread_id=thread_id, history=history,
                system_prompt=system_prompt,
                prior_case_context=self._prior_case_context(config))

        translator = StreamTranslator(initial_tool_count=prior_tool_count)
        final: dict[str, Any] = {}
        interrupted_question: Optional[str] = None

        for chunk in self.graph.stream(
                payload, config=config, stream_mode=["updates", "values"]):
            stream_type, data = chunk
            if stream_type == "updates" and isinstance(data, dict):
                if "__interrupt__" in data:
                    interrupted_question = extract_interrupt_question(data)
                for event in translator.translate_chunk(data):
                    yield event
            elif stream_type == "values" and isinstance(data, dict):
                final = data
        # Rien ne doit rester en attente si le graphe s'est arrêté avant un
        # nœud terminal connu.
        for event in translator.flush():
            yield event

        if interrupted_question is not None:
            # clarification déjà émise par le traducteur
            yield {"type": "done", "accepted": True,
                   "pending_clarification": True, "thread_id": thread_id}
            return

        answer = final.get("final_answer", "")
        if final.get("status") == "rejected":
            yield {"type": "token",
                   "content": ("Request could not be completed: "
                               f"{final.get('stop_reason', '')}")}
            yield {"type": "done", "accepted": False,
                   "thread_id": thread_id}
            return

        for start in range(0, len(answer), 20):
            yield {"type": "token", "content": answer[start:start + 20]}
        yield {"type": "done", "accepted": True, "thread_id": thread_id}

    def run_live(self, query: str, *, thread_id: Optional[str] = None,
                 history: Optional[list[dict]] = None,
                 system_prompt: Optional[str] = None) -> LiveTurnResult:
        """Tour live synchrone (invoke) — utile aux tests et scripts."""
        thread_id = thread_id or f"live-{uuid.uuid4().hex[:8]}"
        config = self._config(thread_id)

        if self.has_pending_interrupt(thread_id):
            payload: Any = Command(resume=query)
        else:
            payload = self.build_live_state(
                query, thread_id=thread_id, history=history,
                system_prompt=system_prompt,
                prior_case_context=self._prior_case_context(config))

        result = self.graph.invoke(payload, config=config)
        question = extract_interrupt_question(result)
        return LiveTurnResult(
            thread_id=thread_id,
            final_answer=result.get("final_answer", ""),
            reasoning=result.get("final_reasoning_summary", ""),
            sources=list(result.get("sources", [])),
            pending_question=question,
            status=result.get("status", ""),
            final_state=result,
        )

    def resume_live(self, thread_id: str, answer: str) -> LiveTurnResult:
        """Reprend explicitement une clarification en attente."""
        result = self.graph.invoke(
            Command(resume=answer), config=self._config(thread_id))
        question = extract_interrupt_question(result)
        return LiveTurnResult(
            thread_id=thread_id,
            final_answer=result.get("final_answer", ""),
            reasoning=result.get("final_reasoning_summary", ""),
            sources=list(result.get("sources", [])),
            pending_question=question,
            status=result.get("status", ""),
            final_state=result,
        )
