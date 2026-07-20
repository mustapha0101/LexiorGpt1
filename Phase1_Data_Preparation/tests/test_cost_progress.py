from types import SimpleNamespace

import pytest

from agentic_generation.cli import _print_progress, _usage_snapshot
from agentic_generation.teacher_client import TeacherClient
from agentic_generation.config import EndpointConfig
from api_cost import CostTracker


def _response(prompt=1000, cached=400, completion=100):
    return SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    ))


def test_gpt_4o_mini_cost_distinguishes_cached_input():
    tracker = CostTracker("gpt-4o-mini")
    tracker.record(_response())
    snapshot = tracker.snapshot()
    assert snapshot["tokens_cached_in"] == 400
    assert snapshot["cost_usd"] == pytest.approx(0.00018)


def test_usage_snapshot_does_not_double_count_shared_client():
    client = TeacherClient(EndpointConfig(model="gpt-4o-mini"))
    client.cost["planner"].record(_response())
    snapshot = _usage_snapshot(client, client)
    assert snapshot["calls"] == 1
    assert snapshot["cost_usd"] == pytest.approx(0.00018)


def test_progress_prints_increment_and_cumulative_cost(capsys):
    before = {"calls": 2, "cost_usd": 0.001}
    after = {"calls": 5, "cost_usd": 0.00225}
    _print_progress(3, 500, {"accepted": 1, "rejected": 2}, 100,
                    "REJETE [validator: source absente]", before, after)
    output = capsys.readouterr().out
    assert "[3/500] REJETE [validator: source absente]" in output
    assert "appels API +3 (cumul 5)" in output
    assert "coût +$0.001250 USD (cumul $0.002250 USD)" in output
