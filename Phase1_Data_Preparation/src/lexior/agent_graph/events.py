# -*- coding: utf-8 -*-
"""Événements de streaming — produits DEPUIS le graphe central.

``graph.stream(..., stream_mode="updates")`` émet un dict
``{nom_du_nœud: mise_à_jour_partielle}`` par nœud exécuté (et une clé
``__interrupt__`` quand une clarification interrompt le run).
:func:`translate_chunk` transforme ces chunks en événements SSE sûrs
(aucun secret, réponses d'outils tronquées) — le même format de fil que
l'interface web consomme depuis la première version :

    thinking / status / decision / tool_call / tool_result /
    clarification / token / done / error
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

NODE_LABELS = {
    "initialize": "Initializing turn",
    "classify_request": "Classifying request",
    "classify_follow_up": "Reading conversation context",
    "update_active_task": "Updating active task",
    "resolve_jurisdiction": "Resolving jurisdiction",
    "analyze_facts": "Analyzing facts",
    "plan": "Planning next step",
    "validate_plan": "Validating plan",
    "handle_clarification": "Asking for clarification",
    "execute_tool": "Executing tool",
    "verify_tool_result": "Verifying tool result",
    "classify_tool_result": "Classifying tool result",
    "update_research_state": "Recording evidence",
    "reformulate_search": "Reformulating search",
    "build_answer_contract": "Preparing answer contract",
    "generate_answer": "Generating answer",
    "run_critics": "Evaluating quality",
    "classify_failures": "Classifying failures",
    "repair_answer": "Repairing answer",
    "repair_trajectory": "Repairing trajectory",
    "validate_final": "Validating trajectory",
    "compute_acceptance": "Computing acceptance",
    "export_dataset": "Exporting result",
    "return_live_answer": "Delivering answer",
    "reject": "Processing rejection",
}

_TOOL_RESULT_PREVIEW_CHARS = 500
_THINKING_PREVIEW_CHARS = 280


class StreamTranslator:
    """Traducteur avec état minimal (déduplication des observations)."""

    def __init__(self) -> None:
        self._tool_count = 0

    def translate_chunk(
        self, chunk: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Chunk ``updates`` de LangGraph → événements SSE (dicts)."""
        if not isinstance(chunk, dict):
            return
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                yield from self._interrupt_events(update)
                continue
            if not isinstance(update, dict):
                continue

            yield {
                "type": "status",
                "node": node_name,
                "label": NODE_LABELS.get(node_name, node_name),
            }

            if node_name == "validate_plan":
                decision = update.get("latest_decision")
                if isinstance(decision, dict):
                    yield {
                        "type": "decision",
                        "step": update.get("step", 0),
                        "decision": decision.get("decision", ""),
                        "tool": decision.get("next_tool"),
                        "args": decision.get("arguments", {}),
                        "jurisdiction": update.get(
                            "resolved_jurisdiction",
                            decision.get("jurisdiction", "")),
                        "thinking": (decision.get("thinking_text")
                                     or "")[:_THINKING_PREVIEW_CHARS],
                    }

            tool_history = update.get("tool_history")
            if isinstance(tool_history, list):
                for obs in tool_history[self._tool_count:]:
                    yield {
                        "type": "tool_call",
                        "tool": obs.tool_name,
                        "args": obs.arguments,
                    }
                    yield {
                        "type": "tool_result",
                        "tool": obs.tool_name,
                        "result": (obs.normalized_response
                                   or "")[:_TOOL_RESULT_PREVIEW_CHARS],
                        "ok": obs.ok,
                    }
                self._tool_count = max(self._tool_count, len(tool_history))

    @staticmethod
    def _interrupt_events(payload: Any) -> Iterator[dict[str, Any]]:
        interrupts = payload if isinstance(payload, (list, tuple)) else [payload]
        for intr in interrupts:
            value = getattr(intr, "value", intr)
            question = (value.get("question", "")
                        if isinstance(value, dict) else str(value))
            yield {"type": "clarification", "question": question}


def extract_interrupt_question(result: dict[str, Any]) -> Optional[str]:
    """Question de clarification d'un résultat ``invoke`` interrompu."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    first = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        return str(value.get("question", ""))
    return str(value)
