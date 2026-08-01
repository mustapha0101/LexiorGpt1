# -*- coding: utf-8 -*-
"""Atomic, single-file recorder for the human 40-situation evaluation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import uuid
from typing import Any, Iterable

from .human_40_scenarios import load_reference_scenarios


SCHEMA_VERSION = "human-evaluation-40-v1"
ALLOWED_MESSAGE_TYPES = {
    "initial_query", "clarification", "clarification_answer", "final_answer",
    "intermediate_answer", "technical_error", "system_notice",
}
ALLOWED_EVENT_TYPES = {
    "initialize", "classify_request", "classify_follow_up", "update_active_task",
    "resolve_jurisdiction", "analyze_facts", "plan", "validate_plan",
    "execute_tool", "verify_tool_result", "classify_tool_result",
    "update_research_state", "reformulate_search", "build_answer_contract",
    "generate_answer", "run_critics", "repair_answer", "validate_final",
    "compute_acceptance", "return_live_answer", "reject", "thinking",
    "request_started",
    "observability",
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _locks_guard:
        return _locks.setdefault(key, threading.RLock())


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="milliseconds")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _duration(start: str | None, end: str | None) -> int | None:
    first, last = _parse_time(start), _parse_time(end)
    if not first or not last:
        return None
    return max(0, round((last - first).total_seconds() * 1000))


def _redact(value: Any, key: str = "") -> Any:
    sensitive = ("api_key", "apikey", "authorization", "cookie", "password",
                 "secret", "token", "access_token", "refresh_token")
    if any(marker in key.casefold() for marker in sensitive):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, key) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", value)
    return value


def _result_payload(result: Any, max_chars: int) -> dict[str, Any]:
    text = result if isinstance(result, str) else json.dumps(
        result, ensure_ascii=False, default=str)
    encoded = text.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    payload: dict[str, Any] = {
        "original_character_count": len(text),
        "result_sha256": digest,
        "result_truncated": len(text) > max_chars,
    }
    if len(text) <= max_chars:
        payload["result"] = _redact(text)
        payload["result_preview"] = text[: min(4000, len(text))]
    else:
        payload["result_preview"] = text[: min(4000, max_chars)]
    return payload


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=root, capture_output=True,
                text=True, timeout=2, check=False)
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "working_tree_dirty": bool(run("status", "--porcelain")),
    }


class Human40Recorder:
    """Owns the lifecycle of one JSON file containing one evaluation run."""

    def __init__(self, root: Path | str | None = None,
                 phase1_root: Path | str | None = None) -> None:
        self.root = Path(root or Path("data/evaluations/human_40"))
        self.phase1_root = Path(phase1_root or Path(__file__).resolve().parents[3])
        self.max_inline_chars = max(1000, int(os.environ.get(
            "HUMAN_EVAL_MAX_INLINE_TOOL_RESULT_CHARS", "20000")))

    def _path(self, run_id: str) -> Path:
        self.validate_run_id(run_id)
        return self.root / f"{run_id}.json"

    @staticmethod
    def validate_run_id(run_id: str) -> None:
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("invalid run_id")

    @staticmethod
    def validate_scenario_id(scenario_id: int) -> int:
        if isinstance(scenario_id, bool) or not isinstance(scenario_id, int) or not 1 <= scenario_id <= 40:
            raise ValueError("scenario_id must be between 1 and 40")
        return scenario_id

    def create_run(self, run_id: str | None = None) -> dict[str, Any]:
        started = _now()
        generated_id = (run_id if run_id is not None
                        else started.strftime("%Y-%m-%d_%H%M%S"))
        if run_id is None:
            candidate = generated_id
            suffix = 0
            while self._path(candidate).exists():
                suffix += 1
                candidate = f"{generated_id}_{suffix}"
            generated_id = candidate
        self.validate_run_id(generated_id)
        path = self._path(generated_id)
        lock = _lock_for(path)
        with lock:
            if path.exists():
                raise FileExistsError("evaluation run already exists")
            git = _git_metadata(self.phase1_root)
            data = {
                "schema_version": SCHEMA_VERSION,
                "run": {
                    "run_id": generated_id,
                    "started_at": _iso(started),
                    "completed_at": None,
                    "status": "in_progress",
                    "current_scenario_id": None,
                    **git,
                    "backend_model": None,
                    "planner_model": None,
                    "writer_model": None,
                    "environment": "local",
                    "total_scenarios": 40,
                },
                "scenarios": [self._new_scenario(item) for item in load_reference_scenarios(self.phase1_root)],
                "run_summary": None,
            }
            self._save_locked(path, data)
            return deepcopy(data)

    @staticmethod
    def _new_scenario(reference: dict[str, Any]) -> dict[str, Any]:
        return {
            **reference,
            "planned_order": reference["scenario_id"],
            "actual_order": None,
            "status": "not_started",
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
            "thread_id": None,
            "start_count": 0,
            "resume_count": 0,
            "human_query": None,
            "expected_answer_note": None,
            "human_evaluation": None,
            "conversation": [],
            "tool_calls": [],
            "graph_events": [],
            "validation_events": [],
            "final_result": None,
            "technical_summary": None,
            "timing": {
                "started_at": None,
                "first_message_at": None,
                "first_assistant_response_at": None,
                "final_answer_at": None,
                "human_review_completed_at": None,
                "time_to_first_response_ms": None,
                "conversation_duration_ms": None,
                "tool_duration_ms": 0,
                "human_review_duration_ms": None,
            },
        }

    def load_run(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id)
        lock = _lock_for(path)
        with lock:
            if not path.exists():
                raise FileNotFoundError("evaluation run not found")
            return json.loads(path.read_text(encoding="utf-8"))

    def _save_locked(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(_redact(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _mutate(self, run_id: str, callback) -> dict[str, Any]:
        path = self._path(run_id)
        lock = _lock_for(path)
        with lock:
            if not path.exists():
                raise FileNotFoundError("evaluation run not found")
            data = json.loads(path.read_text(encoding="utf-8"))
            callback(data)
            self._save_locked(path, data)
            return deepcopy(data)

    @staticmethod
    def _scenario(data: dict[str, Any], scenario_id: int) -> dict[str, Any]:
        for scenario in data["scenarios"]:
            if scenario["scenario_id"] == scenario_id:
                return scenario
        raise ValueError("scenario_id not found")

    def start_scenario(self, run_id: str, scenario_id: int,
                       thread_id: str | None = None) -> dict[str, Any]:
        scenario_id = self.validate_scenario_id(scenario_id)
        now = _iso()

        def mutate(data: dict[str, Any]) -> None:
            scenario = self._scenario(data, scenario_id)
            if any(item["status"] == "in_progress" and item["scenario_id"] != scenario_id
                   for item in data["scenarios"]):
                raise ValueError("another scenario is still in progress")
            is_resume = scenario["status"] in {"interrupted", "completed"}
            if scenario["actual_order"] is None:
                order = [item["actual_order"] for item in data["scenarios"] if item["actual_order"]]
                scenario["actual_order"] = len(order) + 1
            scenario["status"] = "in_progress"
            scenario["started_at"] = scenario["started_at"] or now
            scenario["start_count"] += 1
            if is_resume:
                scenario["resume_count"] += 1
            scenario["thread_id"] = thread_id or f"eval-{run_id}-{scenario_id}-{uuid.uuid4().hex[:8]}"
            scenario["timing"]["started_at"] = scenario["started_at"]
            data["run"]["current_scenario_id"] = scenario_id

        return self._mutate(run_id, mutate)

    def _current_scenario(self, data: dict[str, Any], scenario_id: int | None) -> dict[str, Any]:
        if scenario_id is None:
            scenario_id = data["run"].get("current_scenario_id")
        if scenario_id is None:
            raise ValueError("no current scenario")
        scenario = self._scenario(data, self.validate_scenario_id(scenario_id))
        if scenario["status"] not in {"in_progress", "interrupted"}:
            raise ValueError("scenario is not active")
        return scenario

    def append_message(self, run_id: str, *, scenario_id: int | None,
                       role: str, content: str, message_type: str | None = None,
                       accepted: bool | None = None, turn_index: int | None = None) -> dict[str, Any]:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("invalid message role")
        if not isinstance(content, str) or len(content) > 100_000:
            raise ValueError("invalid message content")
        if message_type and message_type not in ALLOWED_MESSAGE_TYPES:
            raise ValueError("invalid message_type")

        def mutate(data: dict[str, Any]) -> None:
            scenario = self._current_scenario(data, scenario_id)
            if message_type is None:
                inferred = "initial_query" if role == "user" and not scenario["human_query"] else "intermediate_answer"
                message_kind = inferred
            else:
                message_kind = message_type
            timestamp = _iso()
            if role == "user" and not scenario["human_query"]:
                scenario["human_query"] = content
                message_kind = "initial_query"
            message = {
                "message_id": uuid.uuid4().hex,
                "timestamp": timestamp,
                "role": role,
                "content": content,
                "message_type": message_kind,
                "turn_index": turn_index if turn_index is not None else len(scenario["conversation"]) + 1,
            }
            if accepted is not None:
                message["accepted"] = accepted
            scenario["conversation"].append(_redact(message))
            timing = scenario["timing"]
            timing["first_message_at"] = timing["first_message_at"] or timestamp
            if role == "assistant":
                timing["first_assistant_response_at"] = timing["first_assistant_response_at"] or timestamp
            timing["conversation_duration_ms"] = _duration(
                timing["started_at"], timestamp)

        return self._mutate(run_id, mutate)

    def append_backend_event(self, run_id: str, *, scenario_id: int | None,
                             event: dict[str, Any], turn_index: int | None = None) -> dict[str, Any]:
        event_type = str(event.get("type", ""))
        timestamp = _iso()

        def mutate(data: dict[str, Any]) -> None:
            scenario = self._current_scenario(data, scenario_id)
            if event_type == "tool_call":
                scenario["tool_calls"].append({
                    "event_id": uuid.uuid4().hex,
                    "timestamp": timestamp,
                    "turn_index": turn_index or len(scenario["conversation"]) + 1,
                    "step": event.get("step"),
                    "tool": str(event.get("tool", "")),
                    "arguments": _redact(event.get("args", {})),
                    "schema_correction": _redact(event.get("schema_correction", [])),
                    "started_at": timestamp,
                    "completed_at": None,
                    "duration_ms": None,
                })
            elif event_type == "tool_result":
                tool = str(event.get("tool", ""))
                call = next((item for item in reversed(scenario["tool_calls"])
                             if item.get("tool") == tool and item.get("completed_at") is None), None)
                if call is None:
                    call = {"event_id": uuid.uuid4().hex, "tool": tool, "started_at": timestamp}
                    scenario["tool_calls"].append(call)
                result_value = event.get("result_full", event.get("result", ""))
                result_payload = _result_payload(result_value, self.max_inline_chars)
                for field in ("original_character_count", "result_sha256", "result_truncated"):
                    if field in event:
                        result_payload[field] = event[field]
                if event.get("result_truncated") and "result_full" not in event:
                    result_payload.pop("result", None)
                call.update({
                    "completed_at": timestamp,
                    "duration_ms": _duration(call.get("started_at"), timestamp),
                    "ok": bool(event.get("ok", False)),
                    "classification": event.get("classification"),
                    "classification_reason": event.get("reason"),
                    "metadata": _redact(event.get("metadata", {})),
                    **result_payload,
                })
                scenario["timing"]["tool_duration_ms"] = sum(
                    item.get("duration_ms") or 0 for item in scenario["tool_calls"])
            elif event_type in {"token", "clarification"}:
                return
            else:
                node_name = str(event.get("node") or event_type)
                safe = {
                    "event_id": uuid.uuid4().hex,
                    "timestamp": timestamp,
                    "node": node_name,
                    "label": str(event.get("label") or event_type),
                    "step": event.get("step"),
                    "turn_index": turn_index or len(scenario["graph_events"]) + 1,
                }
                if event_type == "decision":
                    safe.update({"decision": event.get("decision"), "tool": event.get("tool"),
                                 "arguments": _redact(event.get("args", {}))})
                if event_type == "request_started":
                    safe.update({
                        "model": str(event.get("model", "")),
                        "provider_model": str(event.get("provider_model", "")),
                        "thread_id": str(event.get("thread_id", "")),
                    })
                if event_type == "observability":
                    details = event.get("event", {}) or {}
                    safe.update({
                        "task_id": str(details.get("task_id", "")),
                        "thread_id": str(details.get("thread_id", "")),
                        "source_ids": _redact(details.get("source_ids", [])),
                        "reason": str(details.get("reason", "")),
                        "status": str(details.get("status", "")),
                        "event_timestamp": str(details.get("timestamp", "")),
                    })
                if event_type == "error":
                    safe["error"] = str(event.get("message", ""))
                if event_type in ALLOWED_EVENT_TYPES or event_type == "error":
                    scenario["graph_events"].append(safe)
            node_name = str(event.get("node") or event_type)
            if node_name in {"validate_final", "compute_acceptance", "reject"} or event_type == "error":
                scenario["validation_events"].append({
                    "timestamp": timestamp,
                    "event": node_name,
                    "accepted": event.get("accepted"),
                    "validation_problems": _redact(event.get(
                        "validation_problems", event.get("validation_issues", []))),
                    "grounding_failures": _redact(event.get("grounding_failures", [])),
                    "message": str(event.get("message", "")),
                })

        return self._mutate(run_id, mutate)

    def append_final_answer(self, run_id: str, *, scenario_id: int | None,
                            content: str, accepted: bool | None = None) -> dict[str, Any]:
        data = self.append_message(
            run_id, scenario_id=scenario_id, role="assistant", content=content,
            message_type="final_answer", accepted=accepted)

        def mutate(current: dict[str, Any]) -> None:
            scenario = self._current_scenario(current, scenario_id)
            timestamp = _iso()
            scenario["final_result"] = {
                "content": content,
                "accepted": accepted,
                "recorded_at": timestamp,
            }
            scenario["timing"]["final_answer_at"] = timestamp
            scenario["timing"]["conversation_duration_ms"] = _duration(
                scenario["timing"].get("started_at"), timestamp)
        return self._mutate(run_id, mutate)

    def update_human_review(self, run_id: str, scenario_id: int,
                            *, expected_answer_note: str | None = None,
                            overall_rating: str | None = None,
                            route_quality: str | None = None,
                            response_quality: list[str] | None = None,
                            notes: str | None = None,
                            observed_category: str | None = None) -> dict[str, Any]:
        self.validate_scenario_id(scenario_id)
        allowed_ratings = {"successful", "partially_successful", "failed", "unable_to_judge"}

        def mutate(data: dict[str, Any]) -> None:
            scenario = self._current_scenario(data, scenario_id)
            if expected_answer_note is not None:
                if len(expected_answer_note) > 20_000:
                    raise ValueError("expected answer note is too long")
                scenario["expected_answer_note"] = expected_answer_note
            evaluation = scenario.get("human_evaluation")
            if not isinstance(evaluation, dict):
                evaluation = {}
                scenario["human_evaluation"] = evaluation
            if overall_rating is not None:
                if overall_rating not in allowed_ratings:
                    raise ValueError("invalid overall_rating")
                evaluation["overall_rating"] = overall_rating
            if route_quality is not None:
                evaluation["route_quality"] = route_quality
            if response_quality is not None:
                evaluation["response_quality"] = list(response_quality)
            if notes is not None:
                evaluation["notes"] = notes[:20_000]
            if observed_category is not None:
                evaluation["observed_category"] = observed_category
            evaluation["updated_at"] = _iso()

        return self._mutate(run_id, mutate)

    def interrupt_scenario(self, run_id: str, scenario_id: int) -> dict[str, Any]:
        self.validate_scenario_id(scenario_id)

        def mutate(data: dict[str, Any]) -> None:
            scenario = self._current_scenario(data, scenario_id)
            scenario["status"] = "interrupted"
            scenario["timing"]["conversation_duration_ms"] = _duration(
                scenario["timing"].get("started_at"), _iso())

        return self._mutate(run_id, mutate)

    def complete_scenario(self, run_id: str, scenario_id: int,
                          *, accepted: bool | None = None,
                          stop_reason: str | None = None) -> dict[str, Any]:
        self.validate_scenario_id(scenario_id)

        def mutate(data: dict[str, Any]) -> None:
            scenario = self._current_scenario(data, scenario_id)
            evaluation = scenario.get("human_evaluation") or {}
            if not scenario.get("expected_answer_note"):
                raise ValueError("expected_answer_note is required before completion")
            if not evaluation.get("overall_rating"):
                raise ValueError("overall_rating is required before completion")
            end = _iso()
            scenario["status"] = "completed"
            scenario["completed_at"] = end
            scenario["timing"]["human_review_completed_at"] = end
            scenario["duration_ms"] = _duration(scenario["started_at"], end)
            scenario["timing"]["human_review_duration_ms"] = _duration(
                scenario["final_result"].get("recorded_at") if scenario.get("final_result") else None, end)
            scenario["final_result"] = {
                **(scenario.get("final_result") or {}),
                "accepted": accepted if accepted is not None else (scenario.get("final_result") or {}).get("accepted"),
                "stop_reason": stop_reason,
            }
            scenario["technical_summary"] = self._technical_summary(scenario)
            if data["run"].get("current_scenario_id") == scenario_id:
                data["run"]["current_scenario_id"] = None

        return self._mutate(run_id, mutate)

    @staticmethod
    def _technical_summary(scenario: dict[str, Any]) -> dict[str, Any]:
        tools = scenario.get("tool_calls", [])
        validation = scenario.get("validation_events", [])
        return {
            "user_message_count": sum(item.get("role") == "user" for item in scenario.get("conversation", [])),
            "assistant_message_count": sum(item.get("role") == "assistant" for item in scenario.get("conversation", [])),
            "clarification_count": sum(item.get("message_type") == "clarification" for item in scenario.get("conversation", [])),
            "tool_call_count": len(tools),
            "tools_used": list(dict.fromkeys(item.get("tool") for item in tools if item.get("tool"))),
            "source_types_attempted": [
                "legislation" if "ccq" in str(item.get("tool", "")) or "cpc" in str(item.get("tool", ""))
                else "jurisprudence" if "jurisprudence" in str(item.get("tool", ""))
                else "other" for item in tools],
            "final_answer_present": bool(scenario.get("final_result")),
            "technical_error_present": any(item.get("event") == "error" for item in validation),
            "accepted": scenario.get("final_result", {}).get("accepted"),
            "total_duration_ms": scenario.get("duration_ms"),
        }

    def complete_run(self, run_id: str) -> dict[str, Any]:
        def mutate(data: dict[str, Any]) -> None:
            if any(item["status"] == "in_progress" for item in data["scenarios"]):
                raise ValueError("an active scenario must be completed or interrupted")
            if any(item["status"] == "not_started" for item in data["scenarios"]):
                raise ValueError("all scenarios must be started before completing the run")
            completed = [item for item in data["scenarios"] if item["status"] == "completed"]
            evaluations = [item.get("human_evaluation", {}) for item in completed]
            durations = [item["duration_ms"] for item in completed if item.get("duration_ms") is not None]
            tools: dict[str, int] = {}
            routes: dict[str, int] = {}
            human_issues: dict[str, int] = {}
            for item in completed:
                for tool in item.get("technical_summary", {}).get("tools_used", []):
                    tools[tool] = tools.get(tool, 0) + 1
                rating = (item.get("human_evaluation") or {}).get("overall_rating")
                if rating:
                    routes[rating] = routes.get(rating, 0) + 1
                for issue in (item.get("human_evaluation") or {}).get("response_quality", []):
                    human_issues[issue] = human_issues.get(issue, 0) + 1
            data["run"]["status"] = "completed"
            data["run"]["completed_at"] = _iso()
            data["run_summary"] = {
                "completed_scenarios": len(completed),
                "successful": sum((item.get("human_evaluation") or {}).get("overall_rating") == "successful" for item in completed),
                "partially_successful": sum((item.get("human_evaluation") or {}).get("overall_rating") == "partially_successful" for item in completed),
                "failed": sum((item.get("human_evaluation") or {}).get("overall_rating") == "failed" for item in completed),
                "unable_to_judge": sum((item.get("human_evaluation") or {}).get("overall_rating") == "unable_to_judge" for item in completed),
                "technical_errors": sum(item.get("technical_summary", {}).get("technical_error_present", False) for item in completed),
                "average_duration_ms": round(sum(durations) / len(durations)) if durations else 0,
                "median_duration_ms": sorted(durations)[len(durations) // 2] if durations else 0,
                "average_clarifications": round(sum(item.get("technical_summary", {}).get("clarification_count", 0) for item in completed) / len(completed), 2) if completed else 0,
                "average_tool_calls": round(sum(item.get("technical_summary", {}).get("tool_call_count", 0) for item in completed) / len(completed), 2) if completed else 0,
                "tools_usage": tools,
                "human_issue_counts": human_issues,
                "observed_route_counts": routes,
            }

        return self._mutate(run_id, mutate)

    def append_manual_event(self, run_id: str, scenario_id: int,
                            event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError("invalid graph event type")
        event = {"type": event_type, **(payload or {})}
        return self.append_backend_event(run_id, scenario_id=scenario_id, event=event)
