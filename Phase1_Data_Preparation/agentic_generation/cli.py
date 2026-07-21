# -*- coding: utf-8 -*-
"""CLI sûre du pipeline agentique (`generate` et `doctor`)."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .agentic_critic import AgenticCritic
from .config import load_config
from .fixtures import MOCK_MCP_FIXTURES
from .legal_critic import LegalCritic
from .legal_rag import (LegalRAG, OpenAIEmbedder, RAGError, build_index,
                        index_exists)
from .mcp_executor import MCPExecutor, MockMCPTransport, RealMCPTransport
from .orchestrator import AgenticOrchestrator
from .planner_agent import PlannerAgent
from .publisher import prepare_release, push_release
from .anchor_bank import AnchorBank, build_anchor_bank
from .scenario_generator import ScenarioGenerator
from .schemas import GenerationManifest
from .storage import JsonCache, RunStorage
from .teacher_client import TeacherClient
from .taxonomy import target_category_counts
from .tool_catalog import load_catalog
from .trajectory_agent import TrajectoryAgent


def _usage_snapshot(*clients) -> dict[str, float | int]:
    """Totalise les clients distincts (Teacher et Critic peuvent être le même)."""
    totals: dict[str, float | int] = {
        "calls": 0, "failed_calls": 0, "tokens_in": 0,
        "tokens_cached_in": 0, "tokens_out": 0, "cost_usd": 0.0,
    }
    seen: set[int] = set()
    for client in clients:
        if client is None or id(client) in seen:
            continue
        seen.add(id(client))
        report = client.cost_report().get("total", {})
        for key in ("calls", "failed_calls", "tokens_in",
                    "tokens_cached_in", "tokens_out"):
            totals[key] += int(report.get(key, 0))
        totals["cost_usd"] += float(report.get("cost_usd", 0.0))
    totals["cost_usd"] = round(float(totals["cost_usd"]), 6)
    return totals


def _print_progress(index: int, max_scenarios: int, counts: Counter,
                    target: int, status: str, before: dict, after: dict) -> None:
    delta_calls = int(after["calls"]) - int(before["calls"])
    delta_cost = float(after["cost_usd"]) - float(before["cost_usd"])
    print(
        f"[{index}/{max_scenarios}] {status} | "
        f"acceptés {counts['accepted']}/{target} | rejetés {counts['rejected']} | "
        f"appels API +{delta_calls} (cumul {after['calls']}) | "
        f"coût +${delta_cost:.6f} USD (cumul ${float(after['cost_usd']):.6f} USD)",
        flush=True,
    )


_MAX_ATTEMPTS_PER_SLOT = 5


def _next_category(targets: dict[str, int], accepted: Counter,
                   attempted: Counter) -> str:
    """Choisit la catégorie la plus en retard sur son quota d'acceptation.

    Si une catégorie a été tentée trop de fois sans succès, elle est ignorée
    et une catégorie avec quota 0 (overflow) prend le relais.
    """
    pending = [
        name for name, target in targets.items()
        if accepted[name] < target
        and (attempted[name] - accepted[name]) < target * _MAX_ATTEMPTS_PER_SLOT
    ]
    if not pending:
        overflow = [
            name for name, target in targets.items()
            if target == 0 and accepted[name] == 0
            and (attempted[name] - accepted[name]) < _MAX_ATTEMPTS_PER_SLOT
        ]
        if overflow:
            return min(overflow, key=lambda name: (attempted[name], name))
        all_pending = [name for name, target in targets.items() if accepted[name] < target]
        if all_pending:
            return min(all_pending, key=lambda name: (attempted[name], name))
        raise RuntimeError("tous les quotas de catégories sont atteints")
    return min(
        pending,
        key=lambda name: (
            -(targets[name] - accepted[name]) / max(targets[name], 1),
            attempted[name],
            name,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic_generation")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="générer des trajectoires")
    generate.add_argument("--config", default=None)
    generate.add_argument("--target-accepted", type=int, default=None)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--offline", action="store_true")
    generate.add_argument("--max-scenarios", type=int, default=None)
    generate.add_argument("--max-tool-calls", type=int, default=None)
    generate.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    generate.add_argument("--no-critics", action="store_true")
    generate.add_argument("--allow-remote-calls", action="store_true")
    generate.add_argument("--push-to-hf", action="store_true")
    generate.add_argument("--run-id", default=None)
    doctor = sub.add_parser("doctor", help="diagnostic local et, si autorisé, réseau")
    doctor.add_argument("--config", default=None)
    doctor.add_argument("--allow-remote-calls", action="store_true")
    rag = sub.add_parser(
        "build-rag-index", help="construire l'index d'embeddings CCQ/CPC")
    rag.add_argument("--config", default=None)
    rag.add_argument("--allow-remote-calls", action="store_true")
    rag.add_argument("--dataset", default=None)
    rag.add_argument("--split", default=None)
    rag.add_argument("--limit", type=int, default=-1)
    rag.add_argument("--force", action="store_true")
    return parser


def _build(args):
    overrides = {key: value for key, value in {
        "target_accepted": getattr(args, "target_accepted", None),
        "seed": getattr(args, "seed", None),
        "dry_run": getattr(args, "dry_run", None) or None,
        "offline": getattr(args, "offline", None) or getattr(args, "dry_run", False) or None,
        "max_scenarios": getattr(args, "max_scenarios", None),
        "max_tool_calls": getattr(args, "max_tool_calls", None),
        "resume": getattr(args, "resume", None),
        "no_critics": getattr(args, "no_critics", None) or None,
        "allow_remote_calls": getattr(args, "allow_remote_calls", False),
        "push_to_hf": getattr(args, "push_to_hf", False),
    }.items() if value is not None}
    cfg = load_config(getattr(args, "config", None), overrides)
    catalog = load_catalog(cfg.catalog_path)
    return cfg, catalog


def doctor(args) -> int:
    cfg, catalog = _build(args)
    checks: list[tuple[str, str]] = []
    required = {
        "pydantic": "pydantic", "yaml": "yaml", "openai": "openai",
        "mcp": "mcp", "numpy": "numpy", "datasets": "datasets",
    }
    for label, module in required.items():
        checks.append((f"dependency:{label}", "ok" if importlib.util.find_spec(module) else "missing"))
    checks.append(("catalog", f"ok ({len(catalog.tools)} outils, hash {catalog.catalog_hash[:12]})"))
    checks.append(("mcp_config", "ok" if Path(cfg.mcp_config_path).exists() else "missing"))
    storage = RunStorage(cfg.data_root, "doctor")
    checks.append(("write_permissions", "ok" if storage.writable() else "failed"))
    checks.append(("teacher_model", "ok" if cfg.teacher.model else "missing"))
    checks.append(("teacher_base_url", "configured" if cfg.teacher.base_url else "missing"))
    checks.append((
        "rag_index",
        f"ok ({cfg.rag.embedding_model})" if index_exists(cfg.rag.index_dir) else "missing",
    ))
    checks.append((
        "rag_embedding_key",
        "configured" if cfg.rag.embedding_api_key else "missing",
    ))
    if args.allow_remote_calls:
        try:
            models = TeacherClient(cfg.teacher, allow_remote_calls=True).list_models()
            checks.append(("teacher_connection", f"ok ({len(models)} modèles)"))
        except Exception as exc:
            checks.append(("teacher_connection", f"failed ({type(exc).__name__})"))
        try:
            transport = RealMCPTransport(cfg.mcp_config_path)
            executor = MCPExecutor(catalog, transport, allow_remote_calls=True)
            asyncio.run(executor.verify_catalog())
            checks.append(("mcp_schema_match", "ok"))
        except Exception as exc:
            checks.append(("mcp_schema_match", f"failed ({type(exc).__name__})"))
    else:
        checks.append(("teacher_connection", "skipped (ajouter --allow-remote-calls)"))
        checks.append(("mcp_schema_match", "skipped (ajouter --allow-remote-calls)"))
    # Sortie volontairement sans valeur d'URL ni clé.
    for name, status in checks:
        print(f"{name}: {status}")
    failed = any("missing" in status or "failed" in status for _, status in checks
                 if not _.startswith("teacher_") or args.allow_remote_calls)
    return 1 if failed else 0


def build_rag_index(args) -> int:
    cfg, _ = _build(args)
    if not args.allow_remote_calls:
        raise SystemExit("construction de l'index refusée sans --allow-remote-calls")
    if args.dataset:
        cfg.rag.dataset_name = args.dataset
    if args.split:
        cfg.rag.dataset_split = args.split
    try:
        embedder = OpenAIEmbedder(cfg.rag, allow_remote_calls=True)
        manifest = build_index(
            cfg.rag, embedder, force=args.force, limit=args.limit,
            progress=lambda message: print(message, flush=True),
        )
    except RAGError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({
        "index_dir": cfg.rag.index_dir,
        "documents": manifest["documents"],
        "dimensions": manifest["dimensions"],
        "embedding_model": manifest["embedding_model"],
        "usage": manifest["usage"],
    }, ensure_ascii=False), flush=True)
    return 0


def generate(args) -> int:
    cfg, catalog = _build(args)
    if cfg.push_to_hf and not cfg.allow_remote_calls:
        raise SystemExit("--push-to-hf exige --allow-remote-calls")
    if not (cfg.offline or cfg.dry_run) and not cfg.allow_remote_calls:
        raise SystemExit("génération distante refusée : ajouter --allow-remote-calls ou utiliser --offline/--dry-run")
    run_id = args.run_id or f"agentic-{cfg.seed}-{catalog.catalog_hash[:12]}"
    print(
        f"[init] run={run_id} | modèle={cfg.teacher.model} | "
        f"objectif={cfg.target_accepted} | limite={cfg.max_scenarios if cfg.max_scenarios > 0 else cfg.target_accepted * 5}",
        flush=True,
    )
    storage = RunStorage(cfg.data_root, run_id)
    cache_root = Path(cfg.data_root) / "cache"
    teacher_cache = JsonCache(cache_root / "teacher")
    mcp_cache = JsonCache(cache_root / ("mcp-mock" if cfg.offline else "mcp-real"))
    teacher = None if cfg.offline else TeacherClient(
        cfg.teacher, allow_remote_calls=cfg.allow_remote_calls, cache=teacher_cache,
        cache_extra_key=f"{cfg.prompt_version}:{catalog.catalog_hash}")
    critic_client = teacher if cfg.critic == cfg.teacher else (
        None if cfg.offline else TeacherClient(cfg.critic, allow_remote_calls=cfg.allow_remote_calls,
                                               cache=JsonCache(cache_root / "critic"),
                                               cache_extra_key=f"{cfg.prompt_version}:{catalog.catalog_hash}"))
    rag = None
    if cfg.rag.enabled and not cfg.offline:
        try:
            rag = LegalRAG.load(
                cfg.rag,
                OpenAIEmbedder(cfg.rag, allow_remote_calls=cfg.allow_remote_calls),
                reranker=teacher,
            )
        except RAGError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"[init] RAG chargé: {len(rag.documents)} articles | "
            f"embeddings={cfg.rag.embedding_model} | reranking hybride "
            f"dense={cfg.rag.dense_weight:.2f}/BM25={1.0 - cfg.rag.dense_weight:.2f} "
            f"+ LLM={cfg.rag.llm_rerank_enabled}",
            flush=True,
        )
    transport = MockMCPTransport(MOCK_MCP_FIXTURES) if cfg.offline else RealMCPTransport(cfg.mcp_config_path)
    executor = MCPExecutor(catalog, transport, allow_remote_calls=cfg.allow_remote_calls,
                           cache=mcp_cache, max_response_chars=cfg.max_tool_response_chars,
                           rag=rag)
    anchor_bank = None
    if not cfg.offline:
        remote_tools = sum(not catalog.is_local(name) for name in catalog.tools)
        print(f"[init] vérification des schémas des {remote_tools} outils MCP...", flush=True)
        asyncio.run(executor.verify_catalog())
        print("[init] catalogue MCP vérifié.", flush=True)
        print("[init] pré-interrogation MCP pour ancres fédérales...", flush=True)
        anchor_bank = build_anchor_bank(transport)
        print(
            f"[init] ancres: {len(anchor_bank.cases)} décisions fédérales, "
            f"{len(anchor_bank.laws)} lois fédérales.",
            flush=True,
        )
    scenario_generator = ScenarioGenerator(teacher, cfg.seed, cfg.offline, cfg.taxonomy_proportions,
                                           anchor_bank=anchor_bank)
    orchestrator = AgenticOrchestrator(
        cfg, catalog, PlannerAgent(catalog, teacher, cfg.offline), executor,
        TrajectoryAgent(teacher, cfg.offline), LegalCritic(critic_client, cfg.offline),
        AgenticCritic(critic_client, cfg.offline),
        progress=lambda message: print(f"  [agent] {message}", flush=True))
    done = storage.completed_scenario_ids() if cfg.resume else set()
    previous_rejected, previous_reasons = storage.rejection_summary() if cfg.resume else (0, {})
    counts = Counter(accepted=storage.accepted_count() if cfg.resume else 0,
                     rejected=previous_rejected, scenarios=0)
    accepted_by_category = Counter(
        storage.request_type_counts(storage.accepted_path) if cfg.resume else {}
    )
    attempted_by_category = Counter(accepted_by_category)
    if cfg.resume:
        attempted_by_category.update(storage.request_type_counts(storage.rejected_path))
    category_targets = target_category_counts(cfg.target_accepted, cfg.taxonomy_proportions)
    rejections = Counter(previous_reasons)
    max_scenarios = cfg.max_scenarios if cfg.max_scenarios and cfg.max_scenarios > 0 else max(cfg.target_accepted * 5, 1)
    if cfg.resume and (counts["accepted"] or counts["rejected"]):
        print(
            f"[init] reprise: {counts['accepted']} accepté(s), "
            f"{counts['rejected']} rejeté(s) déjà enregistrés.",
            flush=True,
        )
    while counts["accepted"] < cfg.target_accepted and counts["scenarios"] < max_scenarios:
        usage_before = _usage_snapshot(teacher, critic_client, rag)
        next_index = counts["scenarios"] + 1
        print(f"[{next_index}/{max_scenarios}] génération de la question...", flush=True)
        category = _next_category(
            category_targets, accepted_by_category, attempted_by_category)
        scenario = scenario_generator.generate(category)
        counts["scenarios"] += 1
        if scenario.scenario_id in done:
            print(f"[{counts['scenarios']}/{max_scenarios}] scénario déjà traité, ignoré.", flush=True)
            continue
        attempted_by_category[scenario.request_type] += 1
        print(
            f"[{counts['scenarios']}/{max_scenarios}] question prête "
            f"({scenario.request_type}); planification, outils et critiques...",
            flush=True,
        )
        result = orchestrator.run(scenario)
        storage.append_raw({"scenario": scenario.model_dump(mode="json"),
                            "accepted": result.accepted,
                            "trajectory": result.trajectory.model_dump(mode="json") if result.trajectory else None})
        if result.accepted and result.trajectory:
            storage.append_accepted(result.trajectory)
            counts["accepted"] += 1
            accepted_by_category[result.trajectory.request_type] += 1
        elif result.rejection:
            storage.append_rejected(result.rejection)
            counts["rejected"] += 1
            rejections.update(result.rejection.reasons)
        usage_after = _usage_snapshot(teacher, critic_client, rag)
        if result.accepted:
            status = "ACCEPTE"
        elif result.rejection:
            reason = " ".join(result.rejection.reasons[0].split())[:120]
            status = f"REJETE [{result.rejection.stage}: {reason}]"
        else:
            status = "REJETE"
        _print_progress(counts["scenarios"], max_scenarios, counts,
                        cfg.target_accepted, status, usage_before, usage_after)
        storage.save_checkpoint({"run_id": run_id, "counts": dict(counts),
                                 "accepted_by_category": dict(accepted_by_category),
                                 "category_targets": category_targets,
                                 "last_scenario_id": scenario.scenario_id})
    manifest = GenerationManifest(
        run_id=run_id, seed=cfg.seed, prompt_version=cfg.prompt_version,
        tool_catalog_hash=catalog.catalog_hash, teacher_model=cfg.teacher.model,
        teacher_base_url_hash=cfg.teacher.base_url_hash, critic_model=cfg.critic.model,
        target_accepted=cfg.target_accepted, counts=dict(counts),
        rejection_reasons=dict(rejections),
        accepted_by_category=dict(accepted_by_category),
        category_targets=category_targets,
        costs={
            "teacher": teacher.cost_report() if teacher else {},
            "critic": (critic_client.cost_report()
                       if critic_client is not None and critic_client is not teacher else {}),
            "rag_queries": rag.cost_report() if rag else {},
        },
        taxonomy_proportions=cfg.taxonomy_proportions, mix=cfg.mix,
        files={"accepted": str(storage.accepted_path), "rejected": str(storage.rejected_path)},
        config_snapshot=cfg.redacted(),
    )
    storage.save_manifest(manifest)
    final_usage = _usage_snapshot(teacher, critic_client, rag)
    print(json.dumps({"run_id": run_id, "counts": dict(counts),
                      "api_usage": final_usage,
                      "manifest": str(storage.manifest_path)}, ensure_ascii=False))
    if cfg.push_to_hf:
        if counts["accepted"] < cfg.target_accepted:
            raise SystemExit("publication refusée : objectif de trajectoires acceptées non atteint")
        split_cfg = cfg.split
        ratios = (float(split_cfg.get("train", 0.90)),
                  float(split_cfg.get("validation", 0.05)),
                  float(split_cfg.get("test", 0.05)))
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise SystemExit("publication refusée : les ratios train/validation/test ne totalisent pas 1")
        release_dir = Path(cfg.data_root) / "release" / run_id
        prepare_release(storage.accepted_path, release_dir, catalog,
                        storage.manifest_path, cfg.seed, ratios,
                        cfg.legal_min_score, cfg.agentic_min_score)
        push_release(release_dir, cfg.hf_dataset_repo_id, cfg.allow_remote_calls)
        print(f"Dataset publié dans {cfg.hf_dataset_repo_id}")
    return 0 if counts["accepted"] >= cfg.target_accepted else 2


def main() -> int:
    args = _parser().parse_args()
    if args.command == "doctor":
        return doctor(args)
    if args.command == "build-rag-index":
        return build_rag_index(args)
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
