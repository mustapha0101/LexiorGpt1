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

from .config import load_config
from .fixtures import MOCK_MCP_FIXTURES
from .legal_rag import (LegalRAG, OpenAIEmbedder, RAGError, build_index,
                        index_exists)
from .mcp_executor import MCPExecutor, MockMCPTransport, RealMCPTransport
from .publisher import (DEFAULT_GROUP_BY, PublicationError, prepare_release,
                        push_release)
from .anchor_bank import AnchorBank, build_anchor_bank
from .scenario_generator import ScenarioGenerator
from .schemas import GenerationManifest
from .storage import JsonCache, RunStorage
from .teacher_client import TeacherClient
from .taxonomy import target_request_type_counts
from .tool_catalog import load_catalog

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[4] / ".env")
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


def _next_request_type(targets: dict[str, int], accepted: Counter,
                       attempted: Counter) -> str:
    """Choisit le type de requête le plus en retard sur son quota."""
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
        raise RuntimeError("tous les quotas de types de requêtes sont atteints")
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
    publish = sub.add_parser(
        "publish", help="produire train/validation/test.jsonl audités")
    publish.add_argument("--config", default=None)
    publish.add_argument("--run-id", required=True,
                         help="run dont les trajectoires acceptées sont publiées")
    publish.add_argument("--output-dir", default=None)
    publish.add_argument("--push-to-hf", action="store_true")
    publish.add_argument("--allow-remote-calls", action="store_true")
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


def _distribution_deviations(
    targets: dict[str, int], actual: dict[str, int],
) -> list[dict[str, object]]:
    total_target = max(sum(targets.values()), 1)
    total_actual = max(sum(actual.values()), 1)
    deviations: list[dict[str, object]] = []
    for name in sorted(set(targets) | set(actual)):
        expected_pct = targets.get(name, 0) / total_target * 100
        actual_pct = actual.get(name, 0) / total_actual * 100
        deviation = actual_pct - expected_pct
        if abs(deviation) > 2.0:
            deviations.append({
                "request_type": name,
                "expected_pct": round(expected_pct, 1),
                "actual_pct": round(actual_pct, 1),
                "deviation_pct": round(deviation, 1),
            })
    return deviations


def generate(args) -> int:
    cfg, catalog = _build(args)
    if cfg.push_to_hf:
        raise SystemExit(
            "--push-to-hf non disponible pour les enregistrements intermédiaires "
            "(schema agentic-2.0). Utilisez le step de projection déterministe.")
    if not (cfg.offline or cfg.dry_run) and not cfg.allow_remote_calls:
        raise SystemExit("génération distante refusée : ajouter --allow-remote-calls ou utiliser --offline/--dry-run")
    run_id = args.run_id or f"agentic-{cfg.seed}-{catalog.catalog_hash[:12]}"
    print(
        f"[init] run={run_id} | modèle={cfg.teacher.model} | "
        f"objectif={cfg.target_accepted} | limite={cfg.max_scenarios if cfg.max_scenarios > 0 else cfg.target_accepted * 5}",
        flush=True,
    )
    for warning in cfg.warnings():
        print(f"[avert] {warning}", flush=True)
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
    scenario_generator = ScenarioGenerator(
        teacher, cfg.seed, cfg.offline,
        request_type_weights=cfg.request_type_weights or None,
        jurisdiction_weights=cfg.jurisdiction_weights or None,
        clarification_stage_weights=cfg.clarification_stage_weights or None,
        failure_mode_weights=cfg.failure_mode_weights or None,
        failure_injection_rate=cfg.failure_injection_rate,
        anchor_bank=anchor_bank,
    )
    # Génération via le graphe LangGraph central — l'unique moteur.
    from lexior.agent_graph import GraphRunner, build_context
    from lexior.services import build_services

    services = build_services(
        cfg, catalog, executor=executor,
        teacher=teacher, critic_client=critic_client)
    graph_runner = GraphRunner(build_context(cfg, catalog, services))
    done = storage.completed_scenario_ids() if cfg.resume else set()
    previous_rejected, previous_reasons = storage.rejection_summary() if cfg.resume else (0, {})
    counts = Counter(accepted=storage.accepted_count() if cfg.resume else 0,
                     rejected=previous_rejected, scenarios=0)
    accepted_by_type = Counter(
        storage.request_type_counts(storage.accepted_path) if cfg.resume else {}
    )
    attempted_by_type = Counter(accepted_by_type)
    if cfg.resume:
        attempted_by_type.update(storage.request_type_counts(storage.rejected_path))
    type_targets = target_request_type_counts(
        cfg.target_accepted, cfg.request_type_weights or None)
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
        request_type_name = _next_request_type(
            type_targets, accepted_by_type, attempted_by_type)
        scenario = scenario_generator.generate(request_type_name=request_type_name)
        counts["scenarios"] += 1
        if scenario.scenario_id in done:
            print(f"[{counts['scenarios']}/{max_scenarios}] scénario déjà traité, ignoré.", flush=True)
            continue
        attempted_by_type[scenario.request_type] += 1
        print(
            f"[{counts['scenarios']}/{max_scenarios}] question prête "
            f"({scenario.request_type}); planification, outils et critiques...",
            flush=True,
        )
        result = graph_runner.run_dataset(
            scenario,
            progress=lambda message: print(f"  [agent] {message}",
                                           flush=True))
        storage.append_raw({"scenario": scenario.model_dump(mode="json"),
                            "accepted": result.accepted,
                            "trajectory": result.trajectory.model_dump(mode="json") if result.trajectory else None})
        if result.accepted and result.trajectory:
            storage.append_accepted(result.trajectory)
            counts["accepted"] += 1
            accepted_by_type[result.trajectory.request_type] += 1
            if result.trajectory.quality.repaired:
                counts["repaired_accepted"] += 1
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
                                 "accepted_by_type": dict(accepted_by_type),
                                 "type_targets": type_targets,
                                 "last_scenario_id": scenario.scenario_id})
    manifest = GenerationManifest(
        run_id=run_id, seed=cfg.seed, prompt_version=cfg.prompt_version,
        tool_catalog_hash=catalog.catalog_hash, teacher_model=cfg.teacher.model,
        teacher_base_url_hash=cfg.teacher.base_url_hash, critic_model=cfg.critic.model,
        target_accepted=cfg.target_accepted, counts=dict(counts),
        rejection_reasons=dict(rejections),
        accepted_by_category=dict(accepted_by_type),
        category_targets=type_targets,
        costs={
            "teacher": teacher.cost_report() if teacher else {},
            "critic": (critic_client.cost_report()
                       if critic_client is not None and critic_client is not teacher else {}),
            "rag_queries": rag.cost_report() if rag else {},
        },
        taxonomy_proportions=cfg.request_type_weights,
        files={
            "accepted": str(storage.accepted_path),
            "rejected": str(storage.rejected_path),
            "repaired": str(storage.repaired_path),
            "preference_pairs": str(storage.preference_pairs_path),
            "regression_cases": str(storage.regression_cases_path),
        },
        config_snapshot=cfg.redacted(),
    )
    storage.save_manifest(manifest)
    deviations = _distribution_deviations(type_targets, dict(accepted_by_type))
    summary = {
        "run_id": run_id,
        "schema_version": "agentic-2.0",
        "counts": {
            "total": counts["scenarios"],
            "accepted": counts["accepted"],
            "repaired_and_accepted": counts.get("repaired_accepted", 0),
            "rejected": counts["rejected"],
        },
        "accepted_by_type": dict(accepted_by_type),
        "type_targets": type_targets,
        "rejection_reasons": dict(rejections),
        "distribution_deviations": deviations,
    }
    storage.save_summary(summary)
    final_usage = _usage_snapshot(teacher, critic_client, rag)
    print(json.dumps({"run_id": run_id, "counts": dict(counts),
                      "api_usage": final_usage,
                      "manifest": str(storage.manifest_path)}, ensure_ascii=False))
    if deviations:
        print(f"[distribution] {len(deviations)} type(s) avec déviation > 2%:",
              flush=True)
        for d in deviations:
            print(f"  {d['request_type']}: cible {d['expected_pct']}% "
                  f"→ réel {d['actual_pct']}% ({d['deviation_pct']:+.1f}%)",
                  flush=True)
    return 0 if counts["accepted"] >= cfg.target_accepted else 2


def publish(args) -> int:
    """Produit train/validation/test.jsonl à partir des acceptées d'un run."""
    cfg, catalog = _build(args)
    storage = RunStorage(cfg.data_root, args.run_id)
    if not storage.accepted_path.exists():
        raise SystemExit(
            f"aucune trajectoire acceptée pour le run {args.run_id} "
            f"({storage.accepted_path})")
    if not storage.manifest_path.exists():
        raise SystemExit(f"manifeste absent : {storage.manifest_path}")

    for warning in cfg.warnings():
        print(f"[avert] {warning}", flush=True)

    split_cfg = cfg.split or {}
    ratios = (
        float(split_cfg.get("train", 0.90)),
        float(split_cfg.get("validation", 0.05)),
        float(split_cfg.get("test", 0.05)),
    )
    output_dir = Path(args.output_dir or
                      (Path(cfg.data_root) / "releases" / args.run_id))
    try:
        audit = prepare_release(
            storage.accepted_path, output_dir, catalog,
            storage.manifest_path, seed=cfg.seed, ratios=ratios,
            legal_min_score=cfg.legal_min_score,
            agentic_min_score=cfg.agentic_min_score,
            group_by=split_cfg.get("group_by") or DEFAULT_GROUP_BY,
            agentic_eval_ratio=float(
                split_cfg.get("agentic_eval", 0.05)),
            separate_agentic_evaluation=bool(
                split_cfg.get("separate_agentic_evaluation", True)),
        )
    except PublicationError as error:
        raise SystemExit(str(error)) from error

    print(f"[publish] {audit['rows']} trajectoires | "
          f"{audit['groups']} groupes de fuite", flush=True)
    for name, count in audit["splits"].items():
        print(f"[publish]   {name:11} {count:5d} "
              f"({audit['achieved_ratios'][name]:.1%} — visé "
              f"{audit['target_ratios'][name]:.1%})", flush=True)
    print(f"[publish]   agentic_eval {audit['agentic_eval']:4d} "
          f"({len(audit['held_out_families'])} familles réservées)",
          flush=True)
    print(f"[publish] chevauchement mesuré entre splits : "
          f"{audit['group_overlap']}", flush=True)
    if not audit["passed"]:
        print("[publish] ÉCHEC : des sources sont partagées entre splits, "
              f"détail dans {output_dir / 'audit_report.json'}", flush=True)
        return 1
    print(f"[publish] écrit dans {output_dir}", flush=True)

    if args.push_to_hf:
        push_release(output_dir, cfg.hf_dataset_repo_id,
                     args.allow_remote_calls)
        print(f"[publish] publié sur {cfg.hf_dataset_repo_id}", flush=True)
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.command == "doctor":
        return doctor(args)
    if args.command == "build-rag-index":
        return build_rag_index(args)
    if args.command == "publish":
        return publish(args)
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
