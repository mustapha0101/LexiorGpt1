#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Déploie uniquement le Teacher Qwen/vLLM sur RunPod et vérifie /v1/models."""

from __future__ import annotations

import argparse
import os
import time

TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"


def parse_args():
    parser = argparse.ArgumentParser(description="Déploiement mono-pod du Teacher Lexior")
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--gpu-type", default="NVIDIA A100 80GB PCIe")
    parser.add_argument("--model", default=TEACHER_MODEL)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--server-api-key", default=os.environ.get("VLLM_API_KEY", ""),
                        help="Clé imposée par vLLM; vide = serveur sans authentification")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--health-timeout", type=float, default=10.0)
    parser.add_argument("--health-retries", type=int, default=90)
    return parser.parse_args()


def wait_until_ready(base_url: str, api_key: str, timeout: int,
                     request_timeout: float, retries: int) -> list[str]:
    import httpx
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last = "aucune réponse"
    for attempt in range(1, retries + 1):
        if time.monotonic() >= deadline:
            break
        try:
            response = httpx.get(f"{base_url}/models", headers=headers,
                                 timeout=request_timeout, follow_redirects=True)
            response.raise_for_status()
            models = [item.get("id", "") for item in response.json().get("data", [])]
            if models:
                return models
            last = "liste de modèles vide"
        except Exception as exc:
            last = type(exc).__name__
        print(f"Teacher pas encore prêt ({attempt}/{retries}: {last})", flush=True)
        time.sleep(min(5 + attempt // 10, 15))
    raise TimeoutError(f"Teacher indisponible après {timeout}s ({last}); consulter les logs du pod")


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("RUNPOD_API_KEY est requis")
    try:
        import runpod
    except ImportError as exc:
        raise SystemExit("installer runpod: pip install runpod") from exc
    runpod.api_key = args.api_key
    command = [
        "--model", args.model, "--quantization", "awq", "--port", "8000",
        "--host", "0.0.0.0", "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.92",
    ]
    if args.server_api_key:
        command.extend(["--api-key", args.server_api_key])
    pod = runpod.create_pod(
        name="lexior-teacher-vllm", image_name="vllm/vllm-openai:latest",
        gpu_type_id=args.gpu_type, gpu_count=1, volume_in_gb=80,
        container_disk_in_gb=30, ports="8000/http,22/tcp",
        env={"HF_TOKEN": args.hf_token, "HF_HOME": "/runpod-volume/hf_cache"},
        docker_args=" ".join(command),
    )
    pod_id = pod["id"]
    base_url = f"https://{pod_id}-8000.proxy.runpod.net/v1"
    print(f"Pod Teacher créé: {pod_id}. Vérification HTTP réelle en cours...", flush=True)
    models = wait_until_ready(base_url, args.server_api_key, args.timeout,
                              args.health_timeout, args.health_retries)
    if args.model not in models:
        raise SystemExit(f"Teacher répond, mais le modèle demandé n'est pas annoncé ({len(models)} modèle(s))")
    print("Teacher prêt. Variables à définir localement:")
    print(f"TEACHER_BASE_URL={base_url}")
    print(f"TEACHER_MODEL={args.model}")
    print("TEACHER_API_KEY=<VLLM_API_KEY configurée, ou valeur factice sans authentification>")
    print("L'orchestrateur n'a pas été lancé; exécutez-le explicitement depuis une machine locale/CPU.")


if __name__ == "__main__":
    main()
