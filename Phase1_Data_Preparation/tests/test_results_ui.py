import json

import pytest

from serve_results_ui import ResultsRepository


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def repository(tmp_path):
    run_id = "demo-run"
    accepted = {
        "scenario_id": "accepted-1",
        "request_type": "exact_text_retrieval",
        "messages": [{"role": "user", "content": "Quel est l'article 11?"}],
        "tool_trace": [{"tool_name": "search_ccq"}],
        "quality": {"legal_score": 0.9},
    }
    rejected = {
        "scenario_id": "rejected-1",
        "request_type": "topic_research",
        "stage": "planner",
        "reasons": ["outil incompatible"],
        "timestamp": "2026-07-20T00:00:00+00:00",
        "trajectory": None,
    }
    raw = [
        {"scenario": {"scenario_id": "accepted-1", "user_query": "Question acceptée"}, "accepted": True},
        {
            "scenario": {"scenario_id": "rejected-1", "user_query": "Question rejetée retrouvée"},
            "accepted": False,
        },
    ]
    _write_jsonl(tmp_path / "accepted" / f"{run_id}.jsonl", [accepted])
    _write_jsonl(tmp_path / "rejected" / f"{run_id}.jsonl", [rejected])
    _write_jsonl(tmp_path / "raw" / f"{run_id}.jsonl", raw)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / f"{run_id}.json").write_text(
        json.dumps(
            {
                "teacher_model": "gpt-4o-mini",
                "target_accepted": 100,
                "costs": {
                    "teacher": {"total": {"calls": 12, "cost_usd": 0.04}},
                    "critic": {"total": {"calls": 4, "cost_usd": 0.01}},
                },
            }
        ),
        encoding="utf-8",
    )
    return ResultsRepository(tmp_path)


def test_lists_runs_with_live_counts_and_cost(repository):
    run = repository.runs()[0]
    assert run["run_id"] == "demo-run"
    assert run["accepted"] == 1
    assert run["rejected"] == 1
    assert run["acceptance_rate"] == 0.5
    assert run["cost_usd"] == 0.05
    assert run["api_calls"] == 16


def test_rejected_question_is_recovered_from_raw(repository):
    records = repository.records("demo-run")
    rejected = next(item for item in records if item["status"] == "rejected")
    assert rejected["question"] == "Question rejetée retrouvée"
    detail = repository.detail("demo-run", "rejected", "rejected-1")
    assert detail["scenario"]["user_query"] == "Question rejetée retrouvée"
    assert detail["reasons"] == ["outil incompatible"]


def test_rejects_path_traversal(repository):
    with pytest.raises(ValueError):
        repository.records("../.env")


def test_live_jsonl_refresh_ignores_partial_last_line(repository):
    path = repository.data_root / "accepted" / "demo-run.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"scenario_id":"unfinished"')
    assert len(repository.records("demo-run")) == 2

    with path.open("a", encoding="utf-8") as handle:
        handle.write(',"messages":[{"role":"user","content":"Nouvelle question"}]}\n')
    records = repository.records("demo-run")
    assert len(records) == 3
    assert any(item["question"] == "Nouvelle question" for item in records)
