# -*- coding: utf-8 -*-
"""Tests for the evaluation comparison module (Phase 5)."""

import json
from pathlib import Path

import pytest

from lexior.evaluation.comparison import (
    ComparisonReport,
    ScenarioOutcome,
    compare_runs,
    print_report,
)


@pytest.fixture
def tmp_runs(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    for d in (old, new):
        (d / "accepted").mkdir(parents=True)
        (d / "rejected").mkdir(parents=True)
    return old, new


def _write_jsonl(path: Path, records: list[dict]):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )


class TestCompareRuns:
    def test_empty_dirs(self, tmp_runs):
        old, new = tmp_runs
        report = compare_runs(old, new, "r1", "r1")
        assert report.summary["old_total"] == 0
        assert report.summary["new_total"] == 0

    def test_gained_scenario(self, tmp_runs):
        old, new = tmp_runs
        _write_jsonl(old / "rejected" / "r1.jsonl", [
            {"scenario_id": "s1", "request_type": "case_analysis",
             "reasons": ["missing_tool"]},
        ])
        _write_jsonl(new / "accepted" / "r2.jsonl", [
            {"scenario_id": "s1", "request_type": "case_analysis",
             "tool_trace": [{"tool_name": "get_ccq_articles"}],
             "quality": {"repair": {"attempted": False}}},
        ])
        report = compare_runs(old, new, "r1", "r2")
        assert report.summary["gained"] == 1
        assert report.summary["lost"] == 0

    def test_lost_scenario(self, tmp_runs):
        old, new = tmp_runs
        _write_jsonl(old / "accepted" / "r1.jsonl", [
            {"scenario_id": "s1", "request_type": "case_analysis",
             "tool_trace": [], "quality": {"repair": {"attempted": False}}},
        ])
        _write_jsonl(new / "rejected" / "r2.jsonl", [
            {"scenario_id": "s1", "request_type": "case_analysis",
             "reasons": ["route_error"]},
        ])
        report = compare_runs(old, new, "r1", "r2")
        assert report.summary["lost"] == 1
        assert report.summary["gained"] == 0

    def test_stable_accept(self, tmp_runs):
        old, new = tmp_runs
        rec = {"scenario_id": "s1", "request_type": "case_analysis",
               "tool_trace": [], "quality": {"repair": {"attempted": False}}}
        _write_jsonl(old / "accepted" / "r1.jsonl", [rec])
        _write_jsonl(new / "accepted" / "r2.jsonl", [rec])
        report = compare_runs(old, new, "r1", "r2")
        assert report.summary["stable_accept"] == 1

    def test_by_request_type(self, tmp_runs):
        old, new = tmp_runs
        _write_jsonl(old / "accepted" / "r1.jsonl", [
            {"scenario_id": "s1", "request_type": "procedure_guidance",
             "tool_trace": [], "quality": {"repair": {"attempted": False}}},
        ])
        _write_jsonl(new / "accepted" / "r2.jsonl", [
            {"scenario_id": "s1", "request_type": "procedure_guidance",
             "tool_trace": [], "quality": {"repair": {"attempted": False}}},
        ])
        report = compare_runs(old, new, "r1", "r2")
        rt = report.summary["by_request_type"]
        assert "procedure_guidance" in rt
        assert rt["procedure_guidance"]["old_acc"] == 1


class TestPrintReport:
    def test_renders_text(self, tmp_runs):
        old, new = tmp_runs
        report = compare_runs(old, new, "r1", "r1")
        text = print_report(report)
        assert "Comparison Report" in text
        assert "Gained" in text

    def test_to_dict(self, tmp_runs):
        old, new = tmp_runs
        report = compare_runs(old, new, "r1", "r1")
        d = report.to_dict()
        assert "summary" in d
        assert "paired" in d
