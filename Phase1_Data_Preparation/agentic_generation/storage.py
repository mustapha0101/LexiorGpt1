# -*- coding: utf-8 -*-
"""Stockage append-only, reprise, cache et manifestes du pipeline agentique."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from .schemas import (
    GenerationManifest, PreferencePair, RejectionRecord, TrainingTrajectory,
)


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JsonCache:
    """Cache fichier simple. Une entrée par fichier évite de corrompre un index global."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        digest = key if len(key) == 64 and all(c in "0123456789abcdef" for c in key) else stable_hash(key)
        return self.root / f"{digest}.json"

    def get(self, key: str) -> Any:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle).get("value")
        except (OSError, ValueError, TypeError):
            return None

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        payload = {"key_hash": stable_hash(key), "value": value}
        with self._lock:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)


class RunStorage:
    SUBDIRS = (
        "raw", "accepted", "rejected", "repaired",
        "preference_pairs", "regression_cases",
        "checkpoints", "manifests", "cache",
    )

    def __init__(self, data_root: str | Path, run_id: str):
        self.root = Path(data_root)
        self.run_id = run_id
        for name in self.SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.raw_path = self.root / "raw" / f"{run_id}.jsonl"
        self.accepted_path = self.root / "accepted" / f"{run_id}.jsonl"
        self.rejected_path = self.root / "rejected" / f"{run_id}.jsonl"
        self.repaired_path = self.root / "repaired" / f"{run_id}.jsonl"
        self.preference_pairs_path = self.root / "preference_pairs" / f"{run_id}.jsonl"
        self.regression_cases_path = self.root / "regression_cases" / f"{run_id}.jsonl"
        self.checkpoint_path = self.root / "checkpoints" / f"{run_id}.json"
        self.manifest_path = self.root / "manifests" / f"{run_id}.json"
        self.summary_path = self.root / "manifests" / f"{run_id}_summary.json"
        self._lock = threading.Lock()

    @staticmethod
    def _append(path: Path, payload: Any) -> None:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_raw(self, payload: Any) -> None:
        with self._lock:
            self._append(self.raw_path, payload)

    def append_accepted(self, record) -> None:
        with self._lock:
            self._append(self.accepted_path, record)

    def append_rejected(self, rejection: RejectionRecord) -> None:
        with self._lock:
            self._append(self.rejected_path, rejection)

    def append_repaired(self, record) -> None:
        with self._lock:
            self._append(self.repaired_path, record)

    def append_preference_pair(self, pair: PreferencePair) -> None:
        with self._lock:
            self._append(self.preference_pairs_path, pair)

    def append_regression_case(self, case) -> None:
        with self._lock:
            self._append(self.regression_cases_path, case)

    def completed_scenario_ids(self) -> set[str]:
        done: set[str] = set()
        for path in (self.accepted_path, self.rejected_path):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                        if item.get("scenario_id"):
                            done.add(item["scenario_id"])
                    except ValueError:
                        continue
        return done

    def accepted_count(self) -> int:
        if not self.accepted_path.exists():
            return 0
        with self.accepted_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    @staticmethod
    def request_type_counts(path: Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        if not path.exists():
            return counts
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                request_type = item.get("request_type")
                if isinstance(request_type, str) and request_type:
                    counts[request_type] = counts.get(request_type, 0) + 1
        return counts

    def rejection_summary(self) -> tuple[int, dict[str, int]]:
        counts: dict[str, int] = {}
        total = 0
        if not self.rejected_path.exists():
            return total, counts
        with self.rejected_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                total += 1
                for reason in item.get("reasons", []):
                    counts[reason] = counts.get(reason, 0) + 1
        return total, counts

    def save_checkpoint(self, payload: dict[str, Any]) -> None:
        self._atomic_json(self.checkpoint_path, payload)

    def save_manifest(self, manifest: GenerationManifest) -> None:
        self._atomic_json(self.manifest_path, manifest.model_dump(mode="json"))

    def save_summary(self, summary: dict[str, Any]) -> None:
        self._atomic_json(self.summary_path, summary)

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def writable(self) -> bool:
        probe = self.root / "checkpoints" / f".{self.run_id}.write-probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
