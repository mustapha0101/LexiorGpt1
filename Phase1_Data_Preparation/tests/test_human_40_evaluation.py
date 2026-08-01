# -*- coding: utf-8 -*-
"""Persistence and isolation tests for the human 40-situation mode."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from lexior.evaluation.human_40_recorder import Human40Recorder
from lexior.evaluation.human_40_scenarios import load_reference_scenarios


def _recorder(tmp_path: Path) -> Human40Recorder:
    return Human40Recorder(tmp_path, Path(__file__).resolve().parents[1])


def test_run_contains_exactly_40_ordered_categorized_scenarios(tmp_path):
    recorder = _recorder(tmp_path)
    data = recorder.create_run("run-40")
    assert len(data["scenarios"]) == 40
    assert [item["scenario_id"] for item in data["scenarios"]] == list(range(1, 41))
    assert [item["category"] for item in data["scenarios"][:12]] == [1] * 12
    assert [item["category"] for item in data["scenarios"][12:20]] == [2] * 8
    assert [item["category"] for item in data["scenarios"][20:28]] == [3] * 8
    assert [item["category"] for item in data["scenarios"][28:]] == [4] * 12
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_reference_pdf_is_the_40_situation_source():
    items = load_reference_scenarios(Path(__file__).resolve().parents[1])
    assert len(items) == 40
    assert items[0]["scenario_description"].startswith("Le chien de la voisine")
    assert items[2]["scenario_description"] == (
        "Un vieil arbre pourri du voisin est tombé sur ton garage."
    )
    assert items[11]["scenario_description"].startswith("Tu es dans le besoin")
    assert items[12]["scenario_description"].startswith("Tu signes un bail")
    assert items[27]["scenario_description"].startswith("Ta copropriété")
    assert items[28]["scenario_description"].startswith("La ventilation")
    assert items[39]["scenario_description"].startswith("Le règlement")
    assert all("ta question" not in item["scenario_description"].lower() for item in items)
    assert all("Catégorie" not in item["scenario_description"] for item in items)


def test_exact_user_message_is_recorded_without_reference_text(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.create_run("exact")
    recorder.start_scenario("exact", 3)
    exact = "ma formulation humaine, sans reformulation"
    recorder.append_message("exact", scenario_id=3, role="user", content=exact)
    data = recorder.load_run("exact")
    scenario = data["scenarios"][2]
    assert scenario["human_query"] == exact
    assert scenario["conversation"][0]["content"] == exact
    assert scenario["scenario_description"] not in scenario["conversation"][0]["content"]


def test_clarification_and_answer_stay_in_one_scenario_and_tool_is_structured(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.create_run("conversation")
    started = recorder.start_scenario("conversation", 3)
    thread_id = next(item for item in started["scenarios"] if item["scenario_id"] == 3)["thread_id"]
    recorder.append_message("conversation", scenario_id=3, role="user", content="question initial")
    recorder.append_message("conversation", scenario_id=3, role="assistant", content="Dans quelle province?", message_type="clarification")
    recorder.append_message("conversation", scenario_id=3, role="user", content="quebec", message_type="clarification_answer")
    recorder.append_backend_event("conversation", scenario_id=3, event={
        "type": "tool_call", "tool": "semantic_search_ccq",
        "args": {"query": "question initial"}, "schema_correction": [],
    })
    recorder.append_backend_event("conversation", scenario_id=3, event={
        "type": "tool_result", "tool": "semantic_search_ccq", "ok": True,
        "classification": "usable", "reason": "candidate", "result": "article result",
        "metadata": {"candidate_count": 20},
    })
    data = recorder.load_run("conversation")
    scenario = data["scenarios"][2]
    assert scenario["thread_id"] == thread_id
    assert [item["message_type"] for item in scenario["conversation"]] == [
        "initial_query", "clarification", "clarification_answer"]
    assert scenario["tool_calls"][0]["arguments"]["query"] == "question initial"
    assert scenario["tool_calls"][0]["metadata"]["candidate_count"] == 20


def test_new_scenario_gets_new_thread_and_previous_data_is_unchanged(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.create_run("threads")
    first = recorder.start_scenario("threads", 3)
    recorder.append_message("threads", scenario_id=3, role="user", content="Q3")
    recorder.update_human_review("threads", 3, expected_answer_note="attendu", overall_rating="partially_successful")
    recorder.complete_scenario("threads", 3)
    second = recorder.start_scenario("threads", 4)
    data = recorder.load_run("threads")
    assert first["scenarios"][2]["thread_id"] != second["scenarios"][3]["thread_id"]
    assert data["scenarios"][2]["human_query"] == "Q3"
    assert data["scenarios"][3]["human_query"] is None


def test_large_tool_result_has_hash_preview_and_no_silent_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv("HUMAN_EVAL_MAX_INLINE_TOOL_RESULT_CHARS", "1000")
    recorder = _recorder(tmp_path)
    recorder.create_run("large")
    recorder.start_scenario("large", 1)
    recorder.append_backend_event("large", scenario_id=1, event={"type": "tool_call", "tool": "tool", "args": {}})
    recorder.append_backend_event("large", scenario_id=1, event={
        "type": "tool_result", "tool": "tool", "ok": True, "result": "x" * 5000,
    })
    tool = recorder.load_run("large")["scenarios"][0]["tool_calls"][0]
    assert tool["result_truncated"] is True
    assert tool["original_character_count"] == 5000
    assert len(tool["result_sha256"]) == 64
    assert "result" not in tool


def test_concurrent_events_are_not_lost_and_json_remains_valid(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.create_run("concurrent")
    recorder.start_scenario("concurrent", 1)

    def append(index: int) -> None:
        recorder.append_message("concurrent", scenario_id=1, role="system", content=f"event-{index}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(32)))
    path = tmp_path / "concurrent.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["scenarios"][0]["conversation"]) == 32


@pytest.mark.parametrize("bad_id", ["../escape", "C:\\temp\\x", "", "a" * 101])
def test_run_id_path_traversal_is_rejected(tmp_path, bad_id):
    with pytest.raises(ValueError):
        _recorder(tmp_path).create_run(bad_id)


def test_invalid_scenario_id_is_rejected(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.create_run("invalid")
    with pytest.raises(ValueError):
        recorder.start_scenario("invalid", 41)
