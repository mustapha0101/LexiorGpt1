#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de déploiement automatique de la Phase 1 en mode Dual-Pod sur RunPod.
Il installe :
1. Un pod d'inférence rapide exécutant vLLM (sur RTX 4090 ou RTX 3090) avec le modèle Qwen 2.5 32B AWQ.
2. Un pod coordinateur de génération (sur GPU RTX 3070 / RTX 3060 économique) connecté à l'API vLLM du premier pod.
"""

import os
import sys
import time
import argparse

try:
    import runpod
except ImportError:
    print("La bibliothèque 'runpod' n'est pas installée.")
    print("Veuillez l'installer localement via : pip install runpod")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Déploiement de la génération dual-pod sur RunPod.")
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
        default="NVIDIA GeForce RTX 4090",
        help="Type de GPU pour le serveur d'inférence vLLM (ex: 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 3090')."
    )
    parser.add_argument(
        "--generation_gpu",
        type=str,
        default="NVIDIA GeForce RTX 3070",
        help="Type de GPU économique pour le coordinateur (ex: 'NVIDIA GeForce RTX 3070', 'NVIDIA GeForce RTX 3060')."
    )
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
    docker_image_inf = "vllm/vllm-openai:latest"
    
    # Commande de démarrage vLLM avec modèle Qwen 2.5 32B quantifié AWQ
    # Limiter max-model-len à 8192 et gpu-memory-utilization à 0.85 pour tenir sur 24 Go de VRAM
    container_command_inf = (
        "serve Qwen/Qwen2.5-14B-Instruct-AWQ "
        "--quantization awq "
        "--port 8000 "
        "--max-model-len 8192 "
        "--gpu-memory-utilization 0.85"
    )
    
    inference_gpu_preferences = [args.inference_gpu, "NVIDIA GeForce RTX 3090", "NVIDIA A100 80GB PCIe"]
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
            try:
                pod_inf = runpod.create_pod(
                    name="lexior-phase1-inference-vllm",
                    image_name=docker_image_inf,
                    gpu_type_id=gpu_type,
                    gpu_count=1,
                    volume_in_gb=50, # 50 Go pour stocker les poids AWQ (environ 20 Go)
                    container_disk_in_gb=20,
                    ports="8000/http,22/tcp",
                    env={"HF_TOKEN": args.hf_token},
                    docker_args=container_command_inf
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
    while True:
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
                print("Le serveur vLLM est en ligne et initialisé !", flush=True)
                break
        except Exception as e:
            print(f"Erreur de suivi : {e}", flush=True)
            
    # Laisser 10 secondes supplémentaires pour s'assurer que le port proxy est mappé au niveau réseau
    time.sleep(10)
    
    # ----------------------------------------------------
    # ÉTAPE 3 : DÉPLOIEMENT DU POD GENERATION (CLIENT)
    # ----------------------------------------------------
    print("\n--- ÉTAPE 3 : CRÉATION DU COORDINATEUR DE GÉNÉRATION ---")
    docker_image_gen = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
    
    env_vars_gen = {
        "OPENAI_API_KEY": "vllm-key",
        "OPENAI_BASE_URL": openai_url,
        "GEN_MODEL": "Qwen/Qwen2.5-14B-Instruct-AWQ",
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
    
    generation_gpu_preferences = [args.generation_gpu, "NVIDIA GeForce RTX 3060", "NVIDIA A30"]
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
