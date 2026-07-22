# -*- coding: utf-8 -*-
"""Stratified per-scenario comparison between old orchestrator and new graph.

Generates a structured report comparing acceptance rates, rejection reasons,
and tool usage patterns.  Does NOT require equal acceptance rates — the
comparison surfaces differences for human review.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ScenarioOutcome:
    scenario_id: str
    request_type: str
    accepted: bool
    rejection_reasons: list[str] = field(default_factory=list)
    tool_count: int = 0
    repair_attempted: bool = False
    repair_successful: bool = False


@dataclass
class ComparisonReport:
    old_outcomes: list[ScenarioOutcome] = field(default_factory=list)
    new_outcomes: list[ScenarioOutcome] = field(default_factory=list)
    paired: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "paired_count": len(self.paired),
            "paired": self.paired,
        }


def _load_outcomes(
    accepted_path: Path, rejected_path: Path,
) -> list[ScenarioOutcome]:
    outcomes: list[ScenarioOutcome] = []

    if accepted_path.exists():
        for line in accepted_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                outcomes.append(ScenarioOutcome(
                    scenario_id=data.get("scenario_id", ""),
                    request_type=data.get("request_type", ""),
                    accepted=True,
                    tool_count=len(data.get("tool_trace", [])),
                    repair_attempted=data.get(
                        "quality", {}).get("repair", {}).get(
                        "attempted", False),
                    repair_successful=data.get(
                        "quality", {}).get("repair", {}).get(
                        "status") == "successful",
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    if rejected_path.exists():
        for line in rejected_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                outcomes.append(ScenarioOutcome(
                    scenario_id=data.get("scenario_id", ""),
                    request_type=data.get("request_type", ""),
                    accepted=False,
                    rejection_reasons=data.get("reasons", []),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    return outcomes


def compare_runs(
    old_run_dir: Path,
    new_run_dir: Path,
    old_run_id: str,
    new_run_id: str,
) -> ComparisonReport:
    report = ComparisonReport()

    report.old_outcomes = _load_outcomes(
        old_run_dir / "accepted" / f"{old_run_id}.jsonl",
        old_run_dir / "rejected" / f"{old_run_id}.jsonl",
    )
    report.new_outcomes = _load_outcomes(
        new_run_dir / "accepted" / f"{new_run_id}.jsonl",
        new_run_dir / "rejected" / f"{new_run_id}.jsonl",
    )

    old_by_id = {o.scenario_id: o for o in report.old_outcomes}
    new_by_id = {o.scenario_id: o for o in report.new_outcomes}
    all_ids = sorted(set(old_by_id) | set(new_by_id))

    old_accepted = sum(1 for o in report.old_outcomes if o.accepted)
    new_accepted = sum(1 for o in report.new_outcomes if o.accepted)
    old_total = len(report.old_outcomes)
    new_total = len(report.new_outcomes)

    gained = 0
    lost = 0
    stable_accept = 0
    stable_reject = 0

    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"old_acc": 0, "old_rej": 0, "new_acc": 0, "new_rej": 0})

    old_rejection_counter: Counter[str] = Counter()
    new_rejection_counter: Counter[str] = Counter()

    for sid in all_ids:
        old = old_by_id.get(sid)
        new = new_by_id.get(sid)

        pair: dict[str, Any] = {
            "scenario_id": sid,
            "request_type": (
                (old.request_type if old else None)
                or (new.request_type if new else "")
            ),
        }
        rt = pair["request_type"]

        if old and new:
            pair["old_accepted"] = old.accepted
            pair["new_accepted"] = new.accepted
            if old.accepted and new.accepted:
                stable_accept += 1
                pair["change"] = "stable_accept"
            elif not old.accepted and not new.accepted:
                stable_reject += 1
                pair["change"] = "stable_reject"
                pair["old_reasons"] = old.rejection_reasons
                pair["new_reasons"] = new.rejection_reasons
            elif not old.accepted and new.accepted:
                gained += 1
                pair["change"] = "gained"
            else:
                lost += 1
                pair["change"] = "lost"
                pair["new_reasons"] = new.rejection_reasons

            if old.accepted:
                by_type[rt]["old_acc"] += 1
            else:
                by_type[rt]["old_rej"] += 1
                for r in old.rejection_reasons:
                    old_rejection_counter[r] += 1
            if new.accepted:
                by_type[rt]["new_acc"] += 1
            else:
                by_type[rt]["new_rej"] += 1
                for r in new.rejection_reasons:
                    new_rejection_counter[r] += 1
        elif old and not new:
            pair["old_accepted"] = old.accepted
            pair["new_accepted"] = None
            pair["change"] = "only_old"
        else:
            pair["old_accepted"] = None
            pair["new_accepted"] = new.accepted if new else None
            pair["change"] = "only_new"

        report.paired.append(pair)

    report.summary = {
        "old_total": old_total,
        "new_total": new_total,
        "old_accepted": old_accepted,
        "new_accepted": new_accepted,
        "old_rate": old_accepted / old_total if old_total else 0.0,
        "new_rate": new_accepted / new_total if new_total else 0.0,
        "gained": gained,
        "lost": lost,
        "stable_accept": stable_accept,
        "stable_reject": stable_reject,
        "by_request_type": dict(by_type),
        "old_top_rejections": old_rejection_counter.most_common(10),
        "new_top_rejections": new_rejection_counter.most_common(10),
    }

    return report


def print_report(report: ComparisonReport) -> str:
    s = report.summary
    lines = [
        "=" * 60,
        "Lexior Pipeline Comparison Report",
        "=" * 60,
        "",
        f"Old pipeline: {s['old_accepted']}/{s['old_total']} accepted "
        f"({s['old_rate']:.1%})",
        f"New pipeline: {s['new_accepted']}/{s['new_total']} accepted "
        f"({s['new_rate']:.1%})",
        "",
        f"Gained (old=rejected, new=accepted): {s['gained']}",
        f"Lost   (old=accepted, new=rejected): {s['lost']}",
        f"Stable accepted: {s['stable_accept']}",
        f"Stable rejected: {s['stable_reject']}",
        "",
        "--- By request type ---",
    ]

    for rt, counts in sorted(s.get("by_request_type", {}).items()):
        old_t = counts["old_acc"] + counts["old_rej"]
        new_t = counts["new_acc"] + counts["new_rej"]
        old_r = counts["old_acc"] / old_t if old_t else 0
        new_r = counts["new_acc"] / new_t if new_t else 0
        lines.append(
            f"  {rt}: old={old_r:.0%} ({old_t}), "
            f"new={new_r:.0%} ({new_t})")

    if s.get("new_top_rejections"):
        lines.append("")
        lines.append("--- Top new rejection reasons ---")
        for reason, count in s["new_top_rejections"]:
            lines.append(f"  [{count}] {reason}")

    changes = [p for p in report.paired if p.get("change") == "lost"]
    if changes:
        lines.append("")
        lines.append("--- Lost scenarios (regression) ---")
        for p in changes[:10]:
            lines.append(
                f"  {p['scenario_id']} ({p['request_type']}): "
                f"{', '.join(p.get('new_reasons', ['?']))}")

    lines.append("")
    return "\n".join(lines)
