# -*- coding: utf-8 -*-
"""Préparation auditée et publication explicite du dataset agentique."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import TrainingTrajectory
from .storage import iter_jsonl
from .validators import compute_metrics, validate_trajectory


def leakage_group(row: TrainingTrajectory) -> str:
    citations = sorted({c for grounding in row.grounding for c in grounding.citations})
    if citations:
        return "citations:" + "|".join(citations)
    articles = sorted({str(obs.arguments.get("start_article")) for obs in row.tool_trace
                       if obs.arguments.get("start_article") is not None})
    if articles:
        return "articles:" + "|".join(articles)
    urls = sorted({u for grounding in row.grounding for u in grounding.source_urls})
    if urls:
        return "sources:" + "|".join(urls)
    return "family:" + row.scenario_family_id


def _clean(row: TrainingTrajectory) -> dict[str, Any]:
    payload = row.model_dump(mode="json")
    for observation in payload.get("tool_trace", []):
        observation.pop("raw_response", None)
        observation.pop("latency_ms", None)
    return payload


def prepare_release(accepted_jsonl: str | Path, output_dir: str | Path,
                    catalog, manifest_path: str | Path, seed: int = 3407,
                    ratios: tuple[float, float, float] = (0.90, 0.05, 0.05),
                    legal_min_score: float = 0.7,
                    agentic_min_score: float = 0.7) -> dict[str, Any]:
    rows = [TrainingTrajectory.model_validate(item) for item in iter_jsonl(accepted_jsonl)]
    if not rows:
        raise ValueError("aucune trajectoire acceptée à publier")
    errors = []
    for row in rows:
        validation = validate_trajectory(row, catalog, allow_mock=False)
        if not validation.valid:
            errors.extend(f"{row.scenario_id}: {error}" for error in validation.errors)
        if row.quality.deterministic_validation is not True:
            errors.append(f"{row.scenario_id}: validation déterministe non confirmée")
        if (row.quality.legal_critic_score is None or
                row.quality.legal_critic_score < legal_min_score):
            errors.append(f"{row.scenario_id}: seuil Legal Critic non atteint")
        if (row.quality.agentic_critic_score is None or
                row.quality.agentic_critic_score < agentic_min_score):
            errors.append(f"{row.scenario_id}: seuil Agentic Critic non atteint")
    if errors:
        raise ValueError("audit de publication échoué: " + "; ".join(errors[:10]))
    groups: dict[str, list[TrainingTrajectory]] = {}
    for row in rows:
        groups.setdefault(leakage_group(row), []).append(row)
    buckets = {"train": [], "validation": [], "test": []}
    train_cut = ratios[0]
    validation_cut = ratios[0] + ratios[1]
    for key, members in sorted(groups.items()):
        value = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "train" if value < train_cut else "validation" if value < validation_cut else "test"
        buckets[split].extend(members)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, members in buckets.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in members:
                handle.write(json.dumps(_clean(row), ensure_ascii=False) + "\n")
    # Évaluation agentique distincte : copie nettoyée du test, jamais remélangée au train.
    with (output / "agentic_eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in buckets["test"]:
            handle.write(json.dumps(_clean(row), ensure_ascii=False) + "\n")
    metrics = compute_metrics(rows, catalog)
    audit = {
        "passed": True, "rows": len(rows),
        "splits": {name: len(items) for name, items in buckets.items()},
        "groups": len(groups), "group_overlap": 0, "metrics": metrics,
    }
    (output / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    (output / "generation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {"dataset_type": "agentic_legal", "schema_version": "agentic-1.0",
            "features": list(_clean(rows[0]).keys()), "splits": audit["splits"]}
    (output / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Lexior agentic legal dataset\n\nTrajectoires ChatML fondées sur des réponses MCP réelles, "
        "validées et groupées sans fuite entre train, validation et test.\n", encoding="utf-8")
    return audit


def push_release(output_dir: str | Path, repo_id: str, allow_remote_calls: bool) -> None:
    if not allow_remote_calls:
        raise PermissionError("publication refusée sans --allow-remote-calls")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN requis pour publier")
    if not repo_id:
        raise ValueError("HF_DATASET_REPO_ID requis pour publier")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output_dir),
                      commit_message="Publish validated agentic Lexior dataset")
