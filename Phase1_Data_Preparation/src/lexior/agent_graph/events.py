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

import json
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

NODE_LABELS = {
    "initialize": "Initializing turn",
    "classify_request": "Classifying request",
    "classify_follow_up": "Reading conversation context",
    "update_active_task": "Updating active task",
    "resolve_jurisdiction": "Resolving jurisdiction",
    "analyze_facts": "Analyzing facts",
    "select_primary_authorities": "Selecting primary authorities",
    "extract_rule_contract": "Building rule contract",
    "derive_rule_specific_facts": "Checking decisive facts",
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
_TOOL_RESULT_INLINE_MAX = max(1000, int(os.environ.get(
    "HUMAN_EVAL_MAX_INLINE_TOOL_RESULT_CHARS", "20000")))

# Nœuds après lesquels un résultat d'outil encore non classé doit être émis
# quand même : mieux vaut l'afficher sans classification que le perdre.
_NOEUDS_TERMINAUX = frozenset({
    "generate_answer", "return_live_answer", "reject",
    "handle_clarification", "export_dataset",
})

_ARTICLE_RE = re.compile(r"\bArticle\s+(\d{1,4}(?:\.\d+)?)", re.IGNORECASE)


def result_metadata(tool: str, arguments: dict[str, Any], text: str) -> dict[str, Any]:
    """Décrit l'aperçu SSE sans exposer tout le résultat."""
    article_numbers = list(dict.fromkeys(_ARTICLE_RE.findall(text or "")))
    metadata: dict[str, Any] = {
        "preview_truncated": len(text or "") > _TOOL_RESULT_PREVIEW_CHARS,
        "preview_character_count": min(len(text or ""), _TOOL_RESULT_PREVIEW_CHARS),
    }
    if tool in {"semantic_search_ccq", "semantic_search_cpc"}:
        metadata.update({
            "candidate_count": len(article_numbers),
            "candidate_articles": article_numbers,
        })
    if tool in {"get_ccq_articles", "get_cpc_articles"}:
        requested = arguments.get("articles") or []
        if not isinstance(requested, list):
            requested = [requested]
        metadata.update({
            "article_count": len(requested),
            "article_numbers": [str(value) for value in requested],
        })
    return metadata


class StreamTranslator:
    """Traducteur avec état minimal (déduplication des observations).

    ``tool_result`` est retenu le temps d'un nœud — ``execute_tool`` puis
    ``classify_tool_result`` s'enchaînent — pour porter la classification
    dans le MÊME événement plutôt que dans un second.
    """

    def __init__(self, initial_tool_count: int = 0, thread_id: str = "") -> None:
        self._tool_count = max(0, int(initial_tool_count))
        self._thread_id = thread_id
        self._en_attente: list[dict[str, Any]] = []
        self._normalizations: dict[tuple[str, str], list[str]] = {}

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

            status_event = {
                "type": "status",
                "node": node_name,
                "label": NODE_LABELS.get(node_name, node_name),
            }
            if node_name in {"validate_final", "compute_acceptance"}:
                for field in ("validation_issues", "grounding_failures", "accepted",
                              "grounding_failures_delta", "open_grounding_failures_total",
                              "resolved_grounding_failures_total"):
                    if field in update:
                        status_event[field] = update[field]
            if update.get("node_failed"):
                yield {
                    "type": "observability",
                    "event": {
                        "event_name": "node_failed",
                        "event_names": ["node_failed"],
                        "node": str(update.get("node_failed") or node_name),
                        "task_id": update.get("task_id", ""),
                        "thread_id": update.get("thread_id", "") or self._thread_id,
                        "reason": str(update.get("stop_reason", "")),
                        "error_type": str(update.get("error_type", "")),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            yield status_event

            if node_name in {
                "update_active_task", "select_primary_authorities",
                "extract_rule_contract", "derive_rule_specific_facts",
                "handle_clarification", "validate_plan",
                "build_answer_contract", "validate_final", "compute_acceptance",
                "repair_answer",
            }:
                selection = update.get("primary_authority_selection") or {}
                if hasattr(selection, "model_dump"):
                    selection = selection.model_dump(mode="json")
                event_names: list[str] = []
                if node_name == "update_active_task" and update.get("active_task_reset"):
                    event_names.append("active_task_reset")
                if node_name == "select_primary_authorities":
                    event_names.append("primary_authorities_selected")
                if node_name == "extract_rule_contract":
                    event_names.append("rule_contract_built")
                    event_names.append("source_sufficiency_decided")
                    contract = update.get("rule_contract")
                    extraction_status = getattr(contract, "extraction_status", "")
                    if isinstance(contract, dict):
                        extraction_status = contract.get("extraction_status", "")
                    if extraction_status == "fallback":
                        event_names.append("rule_extraction_fallback_used")
                if node_name == "derive_rule_specific_facts":
                    event_names.append("clarification_evaluated")
                    analysis = update.get("fact_analysis") or {}
                    if not update.get("missing_critical_facts"):
                        event_names.append("clarification_skipped_as_non_blocking")
                    if analysis.get("conditional_facts"):
                        event_names.append("conditional_answer_selected")
                if node_name == "handle_clarification":
                    event_names.append("clarification_evaluated")
                    if update.get("stop_reason") == "clarification_already_asked":
                        event_names.append("clarification_skipped_as_already_asked")
                if node_name == "validate_plan":
                    if update.get("last_tool_normalization"):
                        event_names.append("repair_routed")
                    if update.get("pending_clarification"):
                        event_names.append("clarification_evaluated")
                if node_name == "build_answer_contract":
                    event_names.append("answer_contract_built")
                if node_name == "validate_final":
                    event_names.append("claim_verification_started")
                    event_names.append("claim_ledger_built")
                    event_names.extend(
                        item.get("event_name", "")
                        for item in update.get("claim_events", [])
                        if item.get("event_name") in {"claim_verified", "claim_failed"}
                    )
                    if update.get("claim_ledger_rebuilt"):
                        event_names.append("claim_ledger_rebuilt")
                    if update.get("failure_resolved"):
                        event_names.append("failure_resolved")
                    if update.get("answer_repair_started"):
                        event_names.append("answer_repair_started")
                    if update.get("answer_repair_succeeded"):
                        event_names.append("answer_repair_succeeded")
                    if update.get("answer_repair_failed"):
                        event_names.append("answer_repair_failed")
                    if update.get("safe_fallback_built"):
                        event_names.append("safe_fallback_built")
                if node_name == "repair_answer":
                    event_names.append(
                        "answer_repair_succeeded"
                        if (update.get("repair") and getattr(
                            update["repair"], "status", "") == "successful")
                        else "answer_repair_failed")
                if node_name == "compute_acceptance":
                    event_names.append("acceptance_computed")
                references = update.get("normative_references") or []
                if update.get("regulation_verified") or any(
                    isinstance(item, dict) and item.get("status") == "resolved"
                    for item in references
                ):
                    event_names.append("normative_reference_resolved")
                if references:
                    event_names.append("normative_reference_detected")
                event_names = list(dict.fromkeys(name for name in event_names if name))
                yield {
                    "type": "observability",
                    "event": {
                        "event_name": event_names[0] if event_names else node_name,
                        "event_names": event_names,
                        "active_task_reset": "active_task_reset" if update.get("active_task_reset") else "",
                        "node": node_name,
                        "task_id": update.get("task_id", ""),
                        "thread_id": update.get("thread_id", "") or self._thread_id,
                        "source_ids": [
                            *selection.get("primary_sources", []),
                            *selection.get("secondary_sources", []),
                        ] if isinstance(selection, dict) else [],
                        "reason": str(update.get("stop_reason", "") or update.get("reason", "")),
                        "status": str(update.get("status", "")),
                        "claim_events": update.get("claim_events", []),
                        "normative_reference_events": (
                            ["normative_reference_resolved"]
                            if update.get("regulation_verified") else
                            ["normative_reference_detected"]
                            if update.get("normative_references") else []),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
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
                    }

            normalization = update.get("last_tool_normalization")
            if isinstance(normalization, dict):
                tool = str(normalization.get("tool") or "")
                args = normalization.get("remaining_arguments")
                fields = normalization.get("removed_fields")
                if tool and isinstance(args, dict) and isinstance(fields, list):
                    self._normalizations[(tool, json.dumps(
                        args, sort_keys=True, ensure_ascii=False))] = [
                            str(field) for field in fields]

            tool_history = update.get("tool_history")
            if isinstance(tool_history, list):
                for rang, obs in enumerate(tool_history[self._tool_count:],
                                           start=self._tool_count):
                    text = obs.normalized_response or ""
                    metadata = dict(getattr(obs, "result_metadata", {}) or {})
                    metadata = {
                        **result_metadata(obs.tool_name, obs.arguments, text),
                        **metadata,
                    }
                    yield {
                        "type": "tool_call",
                        "tool": obs.tool_name,
                        "args": obs.arguments,
                        "schema_correction": self._normalizations.pop(
                            (obs.tool_name, json.dumps(
                                obs.arguments, sort_keys=True,
                                ensure_ascii=False)), []),
                    }
                    self._en_attente.append({
                        "type": "tool_result",
                        "index": rang,
                        "tool": obs.tool_name,
                        "result": text[:_TOOL_RESULT_PREVIEW_CHARS],
                        "ok": obs.ok,
                        "classification": "",
                        "reason": "",
                        "metadata": metadata,
                        "preview_truncated": metadata["preview_truncated"],
                        "preview_character_count": metadata[
                            "preview_character_count"],
                        "original_character_count": len(text),
                        "result_sha256": hashlib.sha256(
                            text.encode("utf-8")).hexdigest(),
                        "result_truncated": len(text) > _TOOL_RESULT_INLINE_MAX,
                        **({"result_full": text}
                           if len(text) <= _TOOL_RESULT_INLINE_MAX else {}),
                    })
                self._tool_count = max(self._tool_count, len(tool_history))

            if node_name == "classify_tool_result":
                yield from self._vider(update.get("search_evaluations"))
            elif node_name in _NOEUDS_TERMINAUX:
                yield from self._vider(None)

    def flush(self) -> Iterator[dict[str, Any]]:
        """À appeler en fin de flux : rien ne doit rester en attente."""
        yield from self._vider(None)

    def _vider(self, evaluations: Any) -> Iterator[dict[str, Any]]:
        """Émet les résultats en attente, classés si l'évaluation est là."""
        par_index: dict[int, tuple[str, str]] = {}
        for ev in (evaluations or []):
            lire = (ev.get if isinstance(ev, dict)
                    else lambda k, d="": getattr(ev, k, d))
            index = lire("tool_call_index", None)
            if index is None:
                continue
            par_index[int(index)] = (str(lire("result_status", "") or ""),
                                     str(lire("result_reason", "") or ""))
        for evenement in self._en_attente:
            statut, motif = par_index.get(evenement["index"], ("", ""))
            evenement["classification"] = statut
            evenement["reason"] = motif
            yield evenement
        self._en_attente = []

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
