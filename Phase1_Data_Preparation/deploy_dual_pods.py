#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mode LEGACY/optionnel dual-pod.

Le Teacher vLLM est déployé et vérifié par HTTP. Le coordinateur GPU n'est
créé que si `--deploy-coordinator` est fourni; l'architecture recommandée est
le Teacher seul, avec l'orchestrateur sur une machine locale ou CPU.
"""

import os
import sys
import time
import argparse
import json
import urllib.request

try:
    import runpod
except ImportError:
    print("La bibliothèque 'runpod' n'est pas installée.")
    print("Veuillez l'installer localement via : pip install runpod")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="LEGACY/optionnel : déploiement dual-pod sur RunPod.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("RUNPOD_API_KEY", None),
        help="Votre clé API RunPod."
    )
    parser.add_argument(
        "--git_repo",
        type=str,
        required=True,
        help="URL de votre dépôt Git contenant le pipeline."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Token d'écriture Hugging Face."
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default=os.environ.get("HF_DATASET_REPO_ID", ""),
        help="Dépôt cible pour le dataset sur Hugging Face."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1500,
        help="Nombre maximum de documents A2AJ à générer."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Nombre de threads d'appels API parallèles (vLLM supporte facilement 16+)."
    )
    parser.add_argument(
        "--inference_gpu",
        type=str,
        default="NVIDIA A100 80GB PCIe",
        help="Type de GPU pour le serveur d'inférence vLLM (ex: 'NVIDIA A100 80GB PCIe', 'NVIDIA GeForce RTX 4090')."
    )
    parser.add_argument(
        "--generation_gpu",
        type=str,
        default="NVIDIA GeForce RTX 3070",
        help="Type de GPU économique pour le coordinateur (ex: 'NVIDIA GeForce RTX 3070', 'NVIDIA GeForce RTX 3060')."
    )
    parser.add_argument("--server-api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--ready-timeout", type=int, default=900)
    parser.add_argument("--deploy-coordinator", action="store_true",
                        help="Créer explicitement le second pod legacy. Sans ce drapeau, seul le Teacher est créé.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # ----------------------------------------------------
    # ÉTAPE 1 : DÉPLOIEMENT DU POD INFERENCE (vLLM)
    # ----------------------------------------------------
    docker_image_inf = "vllm/vllm-openai:v0.5.2"
    
    # Commande de démarrage vLLM avec modèle Qwen 2.5 32B quantifié AWQ
    # Limiter max-model-len à 8192 et gpu-memory-utilization à 0.85 pour tenir sur 24 Go de VRAM
    container_command_inf = (
        "serve Qwen/Qwen2.5-32B-Instruct-AWQ "
        "--quantization awq "
        "--port 8000 "
        "--max-model-len 4096 "
        "--gpu-memory-utilization 0.90 "
        "--kv-cache-dtype fp8"
    )
    
    inference_gpu_preferences = [
        args.inference_gpu,
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100 SXM4 80GB",
        "NVIDIA A30",
        "NVIDIA RTX 6000 Ada",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 3090"
    ]
    pod_inf = None
    
    print("--- ÉTAPE 1 : RÉCUPÉRATION OU CRÉATION DU SERVEUR D'INFÉRENCE vLLM ---", flush=True)
    existing_pods = runpod.get_pods()
    for p in existing_pods:
        if p.get("name") == "lexior-phase1-inference-vllm" and p.get("desiredStatus") == "RUNNING":
            pod_inf = p
            print(f"Pod d'inférence existant trouvé : {pod_inf['id']}", flush=True)
            break
            
    if not pod_inf:
        for gpu_type in inference_gpu_preferences:
            print(f"Tentative d'allocation de l'inférence sur GPU : {gpu_type}...", flush=True)
            # Paramètres vLLM optimisés pour le GPU alloué (vitesse maximale vs stabilité)
            if "A100" in gpu_type or "H100" in gpu_type:
                # Mode Performance Ultime : pas de compression de cache, grand contexte, grand parallélisme
                cmd = (
                    "--model Qwen/Qwen2.5-32B-Instruct-AWQ "
                    "--quantization awq "
                    "--port 8000 "
                    "--max-model-len 8192 "
                    "--gpu-memory-utilization 0.95 "
                    "--max-num-seqs 256"
                )
            else:
                # Mode Optimisé 24Go/48Go : FP8 KV cache, contexte modéré pour éviter OOM
                cmd = (
                    "--model Qwen/Qwen2.5-32B-Instruct-AWQ "
                    "--quantization awq "
                    "--port 8000 "
                    "--max-model-len 4096 "
                    "--gpu-memory-utilization 0.90 "
                    "--kv-cache-dtype fp8"
                )
            try:
                pod_inf = runpod.create_pod(
                    name="lexior-phase1-inference-vllm",
                    image_name=docker_image_inf,
                    gpu_type_id=gpu_type,
                    gpu_count=1,
                    volume_in_gb=50,
                    container_disk_in_gb=20,
                    ports="8000/http,22/tcp",
                    env={"HF_TOKEN": args.hf_token},
                    docker_args=cmd + ((" --api-key " + args.server_api_key) if args.server_api_key else "")
                )
                print(f"Succès ! Serveur alloué sur GPU : {gpu_type}", flush=True)
                break
            except Exception as e:
                print(f"Erreur d'allocation pour {gpu_type} : {e}", flush=True)
            
    if not pod_inf:
        print("Erreur critique : Impossible d'allouer le Pod d'Inférence vLLM.", flush=True)
        sys.exit(1)
        
    pod_id_inf = pod_inf["id"]
    print(f"ID du Pod d'Inférence : {pod_id_inf}", flush=True)
    
    # ----------------------------------------------------
    # ÉTAPE 2 : ATTENTE DU DÉMARRAGE ET RÉCUPÉRATION DU PROXY URL
    # ----------------------------------------------------
    print("\n--- ÉTAPE 2 : ATTENTE DE LA MISE EN LIGNE DU SERVEUR D'INFÉRENCE ---")
    openai_url = f"https://{pod_id_inf}-8000.proxy.runpod.net/v1"
    ready_deadline = time.monotonic() + args.ready_timeout
    while True:
        if time.monotonic() >= ready_deadline:
            print(f"Erreur: /v1/models indisponible après {args.ready_timeout}s.", flush=True)
            sys.exit(1)
        time.sleep(10)
        try:
            status_data = runpod.get_pod(pod_id_inf)
            if not status_data:
                print("Erreur : Le pod d'inférence s'est éteint brusquement.", flush=True)
                sys.exit(1)
            desired_status = status_data.get("desiredStatus")
            runtime = status_data.get("runtime")
            print(f"Statut actuel - desiredStatus: {desired_status}, runtime: {bool(runtime)} (attente du démarrage...)", flush=True)
            if desired_status == "RUNNING" and runtime is not None:
                try:
                    request = urllib.request.Request(openai_url + "/models")
                    if args.server_api_key:
                        request.add_header("Authorization", "Bearer " + args.server_api_key)
                    with urllib.request.urlopen(request, timeout=10) as response:
                        models = json.loads(response.read().decode("utf-8")).get("data", [])
                    if models:
                        print("Le serveur vLLM répond réellement sur /v1/models.", flush=True)
                        break
                except Exception as health_error:
                    print(f"Runtime présent, endpoint HTTP pas prêt: {type(health_error).__name__}", flush=True)
        except Exception as e:
            print(f"Erreur de suivi : {e}", flush=True)
            
    if not args.deploy_coordinator:
        print("Mode recommandé: exécutez l'orchestrateur localement/CPU.")
        print(f"TEACHER_BASE_URL={openai_url}")
        print("TEACHER_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ")
        print("TEACHER_API_KEY=<VLLM_API_KEY ou valeur factice sans authentification>")
        return
    
    # ----------------------------------------------------
    # ÉTAPE 3 : DÉPLOIEMENT DU POD GENERATION (CLIENT)
    # ----------------------------------------------------
    print("\n--- ÉTAPE 3 : CRÉATION DU COORDINATEUR DE GÉNÉRATION ---")
    docker_image_gen = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
    
    env_vars_gen = {
        "TEACHER_API_KEY": args.server_api_key or "not-needed",
        "TEACHER_BASE_URL": openai_url,
        "TEACHER_MODEL": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "OPENAI_API_KEY": args.server_api_key or "not-needed",
        "OPENAI_BASE_URL": openai_url,
        "GEN_MODEL": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "HF_TOKEN": args.hf_token,
        "HF_DATASET_REPO_ID": args.hf_dataset_repo,
        "GEN_LIMIT": str(args.limit),
        "GEN_WORKERS": str(args.workers),
        "USE_LOCAL_OLLAMA": "false"  # vLLM tourne sur le serveur distant
    }
    
    container_command_gen = (
        f"bash -c 'ssh-keygen -A && service ssh start || true; /usr/sbin/sshd || true; "
        f"rm -rf /workspace/DistillationModeles && git clone {args.git_repo} /workspace/DistillationModeles && "
        f"cd /workspace/DistillationModeles/Phase1_Data_Preparation && chmod +x run_generation.sh && "
        f"./run_generation.sh; sleep infinity'"
    )
    
    generation_gpu_preferences = [
        args.generation_gpu,
        "NVIDIA GeForce RTX 3070",
        "NVIDIA GeForce RTX 3060",
        "NVIDIA GeForce RTX 4070 Ti",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 3080",
        "NVIDIA L4",
        "NVIDIA A30"
    ]
    pod_gen = None
    
    for gpu_type in generation_gpu_preferences:
        print(f"Tentative d'allocation du coordinateur sur GPU économique : {gpu_type}...")
        try:
            pod_gen = runpod.create_pod(
                name="lexior-phase1-generation-coordinator",
                image_name=docker_image_gen,
                gpu_type_id=gpu_type,
                gpu_count=1,
                volume_in_gb=30,
                container_disk_in_gb=20,
                ports="8888/http,22/tcp",
                env=env_vars_gen,
                docker_args=container_command_gen
            )
            print(f"Succès ! Coordinateur alloué sur GPU : {gpu_type}")
            break
        except Exception as e:
            print(f"Erreur d'allocation pour {gpu_type} : {e}")
            
    if not pod_gen:
        print("Erreur critique : Impossible de créer le Pod de génération.")
        sys.exit(1)
        
    print(f"\n======================================================================")
    print(f" DEPLOIEMENT DUAL-POD TERMINE AVEC SUCCÈS !")
    print(f"======================================================================")
    print(f"1. Pod Inférence vLLM (RTX 4090) : ID: {pod_id_inf} | URL API: {openai_url}")
    print(f"2. Pod Génération (Coordinateur)  : ID: {pod_gen['id']}")
    print(f"\nLes deux pods sont visibles sur votre console : https://www.runpod.io/console/pods")
    print(f"Une fois le chargement initial du modèle terminé sur le pod vLLM, le coordinateur")
    print(f"téléchargera vos 95 exemples de HF et générera les suivants à vitesse maximale.")
    print(f"======================================================================\n")

if __name__ == "__main__":
    main()
