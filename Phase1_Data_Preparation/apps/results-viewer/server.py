"""Serve the local agentic-generation result inspector.

The server intentionally exposes only the dashboard assets and a small read-only
JSON API over ``data/agentic``.  It has no third-party dependency.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
KINDS = ("accepted", "rejected", "raw")


def _safe_id(value: str, label: str) -> str:
    if not value or not SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} invalide")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _question(trajectory: dict[str, Any] | None, scenario: dict[str, Any] | None) -> str:
    scenario = scenario or {}
    for key in ("user_query", "question", "request"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for message in (trajectory or {}).get("messages", []):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"].strip()
    return "Question indisponible"


def _timestamp(record: dict[str, Any], trajectory: dict[str, Any] | None) -> str | None:
    for value in (
        record.get("timestamp"),
        record.get("created_at"),
        (trajectory or {}).get("generation_metadata", {}).get("generated_at"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _quality_scores(trajectory: dict[str, Any] | None) -> dict[str, Any]:
    quality = (trajectory or {}).get("quality")
    return quality if isinstance(quality, dict) else {}


@dataclass
class _CachedFile:
    signature: tuple[int, int]
    rows: list[dict[str, Any]]


class ResultsRepository:
    """Read and normalize generation artifacts, caching unchanged JSONL files."""

    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self._cache: dict[Path, _CachedFile] = {}
        self._lock = threading.RLock()

    def _jsonl(self, kind: str, run_id: str) -> list[dict[str, Any]]:
        if kind not in KINDS:
            raise ValueError("type de fichier invalide")
        run_id = _safe_id(run_id, "run_id")
        path = self.data_root / kind / f"{run_id}.jsonl"
        if not path.exists():
            return []
        try:
            stat = path.stat()
        except OSError:
            return []
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached.signature == signature:
                return cached.rows
            rows: list[dict[str, Any]] = []
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError:
                            # A live writer may not have finished its last line yet.
                            continue
                        if isinstance(value, dict):
                            rows.append(value)
            except OSError:
                return []
            self._cache[path] = _CachedFile(signature, rows)
            return rows

    def manifest(self, run_id: str) -> dict[str, Any]:
        run_id = _safe_id(run_id, "run_id")
        return _read_json(self.data_root / "manifests" / f"{run_id}.json")

    def _raw_index(self, run_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        by_id: dict[str, dict[str, Any]] = {}
        order: dict[str, int] = {}
        for index, row in enumerate(self._jsonl("raw", run_id)):
            scenario = row.get("scenario") if isinstance(row.get("scenario"), dict) else {}
            trajectory = row.get("trajectory") if isinstance(row.get("trajectory"), dict) else {}
            scenario_id = str(scenario.get("scenario_id") or trajectory.get("scenario_id") or "")
            if scenario_id:
                by_id[scenario_id] = row
                order[scenario_id] = index
        return by_id, order

    def _record_detail(
        self,
        run_id: str,
        status: str,
        record: dict[str, Any],
        raw: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if status == "accepted":
            trajectory = record
        else:
            trajectory = record.get("trajectory") if isinstance(record.get("trajectory"), dict) else None
        raw = raw or {}
        scenario = raw.get("scenario") if isinstance(raw.get("scenario"), dict) else {}
        if trajectory is None and isinstance(raw.get("trajectory"), dict):
            trajectory = raw["trajectory"]
        scenario_id = str(
            record.get("scenario_id")
            or (trajectory or {}).get("scenario_id")
            or scenario.get("scenario_id")
            or ""
        )
        return {
            "run_id": run_id,
            "status": status,
            "scenario_id": scenario_id,
            "question": _question(trajectory, scenario),
            "request_type": record.get("request_type")
            or (trajectory or {}).get("request_type")
            or scenario.get("request_type"),
            "stage": record.get("stage"),
            "reasons": record.get("reasons") if isinstance(record.get("reasons"), list) else [],
            "timestamp": _timestamp(record, trajectory),
            "scenario": scenario,
            "trajectory": trajectory,
            "raw": raw or None,
            "rejection": record if status == "rejected" else None,
        }

    def records(self, run_id: str) -> list[dict[str, Any]]:
        run_id = _safe_id(run_id, "run_id")
        raw_by_id, raw_order = self._raw_index(run_id)
        summaries: list[dict[str, Any]] = []
        fallback_order = len(raw_order)
        for status in ("accepted", "rejected"):
            for offset, record in enumerate(self._jsonl(status, run_id)):
                scenario_id = str(record.get("scenario_id") or "")
                raw = raw_by_id.get(scenario_id)
                detail = self._record_detail(run_id, status, record, raw)
                trajectory = detail["trajectory"] or {}
                tools = [
                    str(item.get("tool_name") or item.get("name") or "outil")
                    for item in trajectory.get("tool_trace", [])
                    if isinstance(item, dict)
                ]
                summaries.append(
                    {
                        "status": status,
                        "scenario_id": detail["scenario_id"],
                        "question": detail["question"],
                        "request_type": detail["request_type"],
                        "stage": detail["stage"],
                        "reasons": detail["reasons"],
                        "timestamp": detail["timestamp"],
                        "quality": _quality_scores(trajectory),
                        "tools": tools,
                        "order": raw_order.get(scenario_id, fallback_order + offset),
                    }
                )
        summaries.sort(key=lambda item: (item["order"], item["status"]))
        return summaries

    def detail(self, run_id: str, status: str, scenario_id: str) -> dict[str, Any] | None:
        run_id = _safe_id(run_id, "run_id")
        scenario_id = _safe_id(scenario_id, "scenario_id")
        if status not in ("accepted", "rejected"):
            raise ValueError("statut invalide")
        raw_by_id, _ = self._raw_index(run_id)
        for record in self._jsonl(status, run_id):
            if str(record.get("scenario_id") or "") == scenario_id:
                return self._record_detail(run_id, status, record, raw_by_id.get(scenario_id))
        return None

    def runs(self) -> list[dict[str, Any]]:
        run_ids: set[str] = set()
        for kind in KINDS:
            directory = self.data_root / kind
            if directory.exists():
                run_ids.update(path.stem for path in directory.glob("*.jsonl") if SAFE_ID.fullmatch(path.stem))
        for kind in ("manifests", "checkpoints"):
            directory = self.data_root / kind
            if directory.exists():
                run_ids.update(path.stem for path in directory.glob("*.json") if SAFE_ID.fullmatch(path.stem))

        runs: list[dict[str, Any]] = []
        for run_id in run_ids:
            accepted = len(self._jsonl("accepted", run_id))
            rejected = len(self._jsonl("rejected", run_id))
            raw_count = len(self._jsonl("raw", run_id))
            manifest = self.manifest(run_id)
            checkpoint = _read_json(self.data_root / "checkpoints" / f"{run_id}.json")
            mtimes = []
            artifacts = [(name, ".jsonl") for name in KINDS]
            artifacts.extend((("manifests", ".json"), ("checkpoints", ".json")))
            for kind, suffix in artifacts:
                path = self.data_root / kind / f"{run_id}{suffix}"
                if path.exists():
                    try:
                        mtimes.append(path.stat().st_mtime)
                    except OSError:
                        pass
            updated_at = datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat() if mtimes else None
            costs = manifest.get("costs") if isinstance(manifest.get("costs"), dict) else {}
            teacher_total = costs.get("teacher", {}).get("total", {}) if isinstance(costs.get("teacher"), dict) else {}
            critic_total = costs.get("critic", {}).get("total", {}) if isinstance(costs.get("critic"), dict) else {}
            total_cost = float(teacher_total.get("cost_usd") or 0) + float(critic_total.get("cost_usd") or 0)
            total_calls = int(teacher_total.get("calls") or 0) + int(critic_total.get("calls") or 0)
            runs.append(
                {
                    "run_id": run_id,
                    "accepted": accepted,
                    "rejected": rejected,
                    "scenarios": max(raw_count, accepted + rejected),
                    "acceptance_rate": accepted / (accepted + rejected) if accepted + rejected else 0,
                    "target_accepted": manifest.get("target_accepted"),
                    "model": manifest.get("teacher_model"),
                    "cost_usd": round(total_cost, 6),
                    "api_calls": total_calls,
                    "updated_at": updated_at,
                    "has_manifest": bool(manifest),
                    "checkpoint": checkpoint.get("counts") if isinstance(checkpoint.get("counts"), dict) else None,
                }
            )
        runs.sort(key=lambda item: item["updated_at"] or "", reverse=True)
        return runs


class ResultsHandler(BaseHTTPRequestHandler):
    repository: ResultsRepository
    static_root: Path

    def _json_response(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static_response(self, filename: str) -> None:
        path = (self.static_root / filename).resolve()
        if path.parent != self.static_root.resolve() or not path.is_file():
            self._json_response({"error": "Fichier introuvable"}, 404)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/runs":
                self._json_response({"runs": self.repository.runs()})
            elif parsed.path == "/api/records":
                run_id = query.get("run", [""])[0]
                self._json_response({"records": self.repository.records(run_id)})
            elif parsed.path == "/api/detail":
                run_id = query.get("run", [""])[0]
                status = query.get("status", [""])[0]
                scenario_id = query.get("id", [""])[0]
                detail = self.repository.detail(run_id, status, scenario_id)
                if detail is None:
                    self._json_response({"error": "Résultat introuvable"}, 404)
                else:
                    self._json_response(detail)
            elif parsed.path == "/api/manifest":
                run_id = query.get("run", [""])[0]
                self._json_response(self.repository.manifest(run_id))
            elif parsed.path in ("/", "/index.html"):
                self._static_response("index.html")
            elif parsed.path in ("/app.js", "/styles.css"):
                self._static_response(parsed.path[1:])
            else:
                self._json_response({"error": "Route introuvable"}, 404)
        except ValueError as exc:
            self._json_response({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # Keep the local inspector useful with a partially written run.
            self._json_response({"error": f"Erreur de lecture: {type(exc).__name__}"}, 500)

    def log_message(self, message: str, *args: Any) -> None:
        print(f"[results-ui] {self.address_string()} {message % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualiser les résultats agentiques locaux.")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (défaut: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port HTTP (défaut: 8765)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "agentic",
        help="Racine des fichiers agentic",
    )
    parser.add_argument("--open", action="store_true", help="Ouvrir automatiquement le navigateur")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    static_root = Path(__file__).resolve().parent / "results_ui"
    if not static_root.is_dir():
        raise SystemExit(f"Interface introuvable: {static_root}")
    ResultsHandler.repository = ResultsRepository(args.data_root)
    ResultsHandler.static_root = static_root
    server = ThreadingHTTPServer((args.host, args.port), ResultsHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"[results-ui] Données: {args.data_root.resolve()}")
    print(f"[results-ui] Ouvrez {url} (Ctrl+C pour arrêter)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[results-ui] Arrêt.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
