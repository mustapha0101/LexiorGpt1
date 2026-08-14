# -*- coding: utf-8 -*-
"""Exécution du graphe de la démonstration live Lexior."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from lexior.agentic.prompts import agent_system_prompt
from lexior.agentic.schemas import ScenarioSpec

from .checkpointing import create_memory_checkpointer
from .context import GraphContext
from .events import StreamTranslator, extract_interrupt_question
from .graph import build_graph
from .state import initial_state

# Coût mesuré d'un tour : préfixe linéaire + contrat de réponse + critiques
# + acceptation, puis un cycle par appel d'outil / réparation / clarification.
_TURN_SUPERSTEPS = 15
_SUPERSTEPS_PER_CYCLE = 9
_RECURSION_SLACK = 12
_MIN_RECURSION_LIMIT = 80


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
    """Compile une fois le graphe utilisé par le chat live."""

    def __init__(self, context: GraphContext, checkpointer=None):
        self.context = context
        self.checkpointer = checkpointer or create_memory_checkpointer()
        self.graph = build_graph(self.context, checkpointer=self.checkpointer)
        self.recursion_limit = self._derive_recursion_limit()

    # ── Aides communes ───────────────────────────────────────────────────

    def _derive_recursion_limit(self) -> int:
        """Plafond LangGraph déduit des budgets réellement configurés.

        Un tour coûte ``_TURN_SUPERSTEPS + _SUPERSTEPS_PER_CYCLE × N`` où N
        est le nombre de cycles (appel d'outil, réparation, clarification).
        Un plafond fixe plus bas que les budgets rendrait ceux-ci inatteignables
        et transformerait un tour normal en ``GraphRecursionError``.
        """
        config = self.context.config
        tool_calls = getattr(config, "max_tool_calls_live",
                             getattr(config, "max_tool_calls", 6))
        repairs = getattr(config, "max_repairs", 1)
        clarifications = getattr(config, "max_clarifications_live", 2)
        reformulations = getattr(config, "max_search_reformulations_live", 1)
        cycles = int(tool_calls) + int(repairs) + int(clarifications) \
            + int(reformulations)
        return max(
            _MIN_RECURSION_LIMIT,
            _TURN_SUPERSTEPS + _SUPERSTEPS_PER_CYCLE * cycles
            + _RECURSION_SLACK,
        )

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": self.recursion_limit}

    def system_prompt(self) -> str:
        return agent_system_prompt(self.context.catalog)

    def has_pending_interrupt(self, thread_id: str) -> bool:
        """Vrai UNIQUEMENT si le thread est suspendu sur un ``interrupt()``.

        ``snapshot.next`` est non vide pour tout run inachevé — y compris un
        run mort sur ``GraphRecursionError``. S'y fier ferait passer la
        question suivante en ``Command(resume=...)`` vers un interrupt
        inexistant, où elle serait silencieusement perdue.
        """
        return self._interrupt_state(thread_id)[0]

    def _interrupt_state(self, thread_id: str) -> tuple[bool, bool]:
        """``(suspendu sur interrupt, thread inachevé sans interrupt)``."""
        try:
            snapshot = self.graph.get_state(self._config(thread_id))
        except Exception:
            return False, False
        if snapshot is None:
            return False, False
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if not interrupts:
            interrupts = tuple(
                interrupt
                for task in (getattr(snapshot, "tasks", ()) or ())
                for interrupt in (getattr(task, "interrupts", ()) or ())
            )
        if interrupts:
            return True, False
        # Non vide sans interrupt : le thread est resté en plan (crash,
        # limite de récursion, consommateur SSE parti). On repart à neuf
        # plutôt que d'avaler le message suivant.
        return False, bool(getattr(snapshot, "next", ()))

    # ── Mode live ────────────────────────────────────────────────────────

    def build_live_state(self, query: str, *, thread_id: str,
                         history: Optional[list[dict]] = None,
                         system_prompt: Optional[str] = None,
                         request_type: str = "unknown",
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
        state["evidence_first_maximum_article_batches"] = self.context.config.evidence_first_maximum_article_batches
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
                "primary_authority_selection": dict(context.get(
                    "primary_authority_selection", {})),
                "rule_contract": dict(context.get("rule_contract", {})),
                "source_sufficiency_decision": dict(context.get(
                    "source_sufficiency_decision", {})),
                "normative_references": list(context.get(
                    "normative_references", [])),
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
        except Exception:  # noqa: BLE001 — checkpoint illisible ou corrompu
            # Checkpointer absent ou thread encore inexistant.
            return {}

    def _checkpoint_tool_count(self, config: dict) -> int:
        """Nombre d'observations déjà diffusées avant une reprise suspendue."""
        try:
            snapshot = self.graph.get_state(config)
            values = getattr(snapshot, "values", {}) or {}
            history = values.get("tool_history", [])
            return len(history) if isinstance(history, list) else 0
        except Exception:  # noqa: BLE001 — checkpoint illisible ou corrompu
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

        pending_interrupt, stalled = self._interrupt_state(thread_id)
        if stalled:
            # Thread resté en plan : on redémarre un tour propre sur un
            # nouveau fil plutôt que d'écrire par-dessus un checkpoint mort.
            thread_id = f"live-{uuid.uuid4().hex[:8]}"
            config = self._config(thread_id)
            yield {"type": "status", "node": "runner",
                   "label": "Reprise sur un nouveau fil (tour précédent interrompu)"}
        prior_tool_count = self._checkpoint_tool_count(config) \
            if pending_interrupt else 0
        if pending_interrupt:
            payload: Any = Command(resume=query)
        else:
            payload = self.build_live_state(
                query, thread_id=thread_id, history=history,
                system_prompt=system_prompt,
                prior_case_context=self._prior_case_context(config))

        translator = StreamTranslator(
            initial_tool_count=prior_tool_count, thread_id=thread_id)
        final: dict[str, Any] = {}
        interrupted_question: Optional[str] = None

        try:
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
        except GraphRecursionError:
            # Budget de supersteps épuisé : dégradation annoncée, jamais une
            # trace interne brute ni un tour sans « done ».
            for event in translator.flush():
                yield event
            yield {"type": "token",
                   "content": ("La recherche n'a pas convergé dans le budget "
                               "d'étapes alloué. Reformulez la question ou "
                               "précisez la province concernée.")}
            yield {"type": "done", "accepted": False,
                   "stop_reason": "recursion_limit_reached",
                   "thread_id": thread_id}
            return
        # Rien ne doit rester en attente si le graphe s'est arrêté avant un
        # nœud terminal connu.
        for event in translator.flush():
            yield event

        if interrupted_question is not None:
            # clarification déjà émise par le traducteur; le tour n'est PAS
            # une réponse acceptée, seulement une question en attente.
            yield {"type": "done", "accepted": False,
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
        yield {"type": "done", "accepted": self._accepted(final),
               "thread_id": thread_id}

    @staticmethod
    def _accepted(final: dict[str, Any]) -> bool:
        """Verdict réel du tour — jamais « accepté » par défaut.

        ``status != "rejected"`` ne dit rien de l'acceptation : une réponse
        de repli (« je ne peux pas déterminer ») en sort aussi. Seul
        ``acceptance_result`` fait foi, et une réponse vide n'est jamais
        acceptée.
        """
        if not str(final.get("final_answer") or "").strip():
            return False
        acceptance = final.get("acceptance_result")
        if acceptance is None:
            return False
        return bool(getattr(acceptance, "accepted", False))

    def run_live(self, query: str, *, thread_id: Optional[str] = None,
                 history: Optional[list[dict]] = None,
                 system_prompt: Optional[str] = None) -> LiveTurnResult:
        """Tour live synchrone (invoke) — utile aux tests et scripts."""
        thread_id = thread_id or f"live-{uuid.uuid4().hex[:8]}"
        config = self._config(thread_id)

        pending_interrupt, stalled = self._interrupt_state(thread_id)
        if stalled:
            thread_id = f"live-{uuid.uuid4().hex[:8]}"
            config = self._config(thread_id)
        if pending_interrupt:
            payload: Any = Command(resume=query)
        else:
            payload = self.build_live_state(
                query, thread_id=thread_id, history=history,
                system_prompt=system_prompt,
                prior_case_context=self._prior_case_context(config))

        try:
            result = self.graph.invoke(payload, config=config)
        except GraphRecursionError:
            return LiveTurnResult(
                thread_id=thread_id,
                status="rejected",
                final_state={"stop_reason": "recursion_limit_reached"},
            )
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
