# -*- coding: utf-8 -*-
"""Two-situation API E2E for the human evaluation envelope."""

from __future__ import annotations

from fastapi.testclient import TestClient

import lexior.api.app as api
from lexior.evaluation.human_40_recorder import Human40Recorder


class _FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    def stream_live(self, *_args, **_kwargs):
        self.calls += 1
        yield {"type": "status", "node": "initialize", "label": "Initializing turn"}
        if self.calls == 1:
            yield {"type": "clarification", "question": "Dans quelle province?"}
            yield {"type": "done", "accepted": True, "pending_clarification": True}
        else:
            yield {"type": "status", "node": "validate_final", "label": "Validating trajectory",
                   "validation_issues": [], "grounding_failures": []}
            yield {"type": "token", "content": "Réponse finale de test."}
            yield {"type": "done", "accepted": True, "pending_clarification": False}


def _scenario(data, scenario_id):
    return next(item for item in data["scenarios"] if item["scenario_id"] == scenario_id)


def test_two_human_situations_share_one_file_but_not_a_thread(tmp_path, monkeypatch):
    recorder = Human40Recorder(tmp_path, tmp_path)
    fake = _FakeRunner()
    monkeypatch.setattr(api, "_HUMAN_40", recorder)
    monkeypatch.setattr(api, "_runner_for", lambda _model: fake)
    client = TestClient(api.app)

    created = client.post("/api/evaluations/human-40/runs", json={"run_id": "e2e-human"}).json()
    started = client.post(
        "/api/evaluations/human-40/runs/e2e-human/scenarios/3/start", json={}).json()
    scenario3 = _scenario(started, 3)
    thread3 = scenario3["thread_id"]

    first = client.post("/api/chat", json={
        "query": "question humaine A", "mode": "human_40",
        "model": "gpt-4o-mini",
        "thread_id": thread3, "evaluation_run_id": "e2e-human",
        "evaluation_scenario_id": 3,
    })
    assert first.status_code == 200
    assert "Dans quelle province?" in first.text
    assert '"type": "request_started"' in first.text
    assert '"model": "gpt-4o-mini"' in first.text
    second = client.post("/api/chat", json={
        "query": "quebec", "mode": "human_40",
        "thread_id": thread3, "evaluation_run_id": "e2e-human",
        "evaluation_scenario_id": 3,
    })
    assert second.status_code == 200
    assert "Réponse finale de test." in second.text
    client.patch(
        "/api/evaluations/human-40/runs/e2e-human/scenarios/3",
        json={"expected_answer_note": "attendu A", "overall_rating": "partially_successful"},
    )
    client.post("/api/evaluations/human-40/runs/e2e-human/scenarios/3/complete")

    started4 = client.post(
        "/api/evaluations/human-40/runs/e2e-human/scenarios/4/start", json={}).json()
    scenario4 = _scenario(started4, 4)
    assert scenario4["thread_id"] != thread3
    third = client.post("/api/chat", json={
        "query": "question humaine B", "mode": "human_40",
        "thread_id": scenario4["thread_id"], "evaluation_run_id": "e2e-human",
        "evaluation_scenario_id": 4,
    })
    assert third.status_code == 200
    client.patch(
        "/api/evaluations/human-40/runs/e2e-human/scenarios/4",
        json={"expected_answer_note": "attendu B", "overall_rating": "successful"},
    )
    client.post("/api/evaluations/human-40/runs/e2e-human/scenarios/4/complete")

    saved = recorder.load_run("e2e-human")
    saved3, saved4 = _scenario(saved, 3), _scenario(saved, 4)
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert saved3["human_query"] == "question humaine A"
    assert [item["content"] for item in saved3["conversation"][:3]] == [
        "question humaine A", "Dans quelle province?", "quebec"]
    assert saved3["human_evaluation"]["overall_rating"] == "partially_successful"
    assert any(
        event.get("node") == "request_started"
        and event.get("model") == "gpt-4o-mini"
        for event in saved3["graph_events"]
    )
    assert saved4["human_query"] == "question humaine B"
    assert all(item["scenario_id"] == 3 for item in saved["scenarios"] if item["scenario_id"] == 3)
    assert saved4["conversation"]
    assert "question humaine A" not in " ".join(
        str(item.get("content", "")) for item in saved4["conversation"])
