#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de déploiement d'une instance vLLM de production AWQ pour LexiorGPT.
Ce script lance un pod sur RunPod (recommandé : RTX 4090 ou L40 économique)
servant le modèle quantifié LexiorGPT-AWQ avec un contexte de 32 768 tokens (32k).
"""

import os
import argparse
import sys
import time

try:
    import runpod
except ImportError:
    print("La bibliothèque 'runpod' n'est pas installée.")
    print("Veuillez l'installer localement via : pip install runpod")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Déploiement de vLLM de production AWQ sur RunPod.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("RUNPOD_API_KEY", None),
        help="Votre clé API RunPod."
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="NVIDIA GeForce RTX 4090",
        help="Type de GPU économique (ex: 'NVIDIA GeForce RTX 4090', 'NVIDIA L40', 'NVIDIA RTX 6000 Ada')."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Token d'écriture Hugging Face pour télécharger le modèle privé."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="intelliwork/LexiorGpt1-merged-AWQ",
        help="ID du modèle quantifié AWQ sur Hugging Face."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Image officielle vLLM
    docker_image = "vllm/vllm-openai:latest"
    
    # Cache HF sur le grand volume persistant /runpod-volume
    env_vars = {
        "HF_TOKEN": args.hf_token,
        "HF_HOME": "/runpod-volume/hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    }
    
    # Arguments optimisés pour le modèle AWQ avec un contexte de 32k (32768 tokens)
    vllm_cmd = [
        "--model", args.model_id,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--quantization", "awq",
        "--max-model-len", "32768",      # Contexte étendu à 32k
        "--gpu-memory-utilization", "0.90", # Marger 10% pour les activations à 32k
        "--enable-auto-tool-choice",      # Appel d'outils automatique MCP
        "--tool-call-parser", "hermes"    # Parseur compatible avec le format Hermes
    ]
        
    container_command = " ".join(vllm_cmd)
    
    print("==================================================")
    print(" DÉPLOIEMENT DU POD DE PRODUCTION LEXIORGPT AWQ")
    print("==================================================")
    print(f"GPU Cible       : {args.gpu_type}")
    print(f"Modèle AWQ      : {args.model_id}")
    print(f"Longueur Max    : 32768 (32k)")
    print("==================================================")
    
    try:
        pod = runpod.create_pod(
            name="lexior-vllm-prod-awq",
            image_name=docker_image,
            gpu_type_id=args.gpu_type,
            gpu_count=1,
            volume_in_gb=100, # 100 Go suffisent largement pour le modèle AWQ (18 Go) + cache
            container_disk_in_gb=30,
            ports="8000/http,22/tcp",
            env=env_vars,
            docker_args=container_command
        )
    except Exception as e:
        print(f"Erreur lors de la création du Pod vLLM AWQ : {e}")
        sys.exit(1)
        
    pod_id = pod["id"]
    print(f"\n==================================================")
    print(f" Pod vLLM AWQ créé avec succès ! ID : {pod_id}")
    print(f"==================================================")
    print(f"Suivi de l'instance : https://www.runpod.io/console/pods")
    
    print("\nEn attente de l'adresse IP publique et de l'initialisation...")
    for _ in range(10):
        time.sleep(5)
        pod_status = runpod.get_pod(pod_id)
        if pod_status and pod_status.get("runtime"):
            ports = pod_status["runtime"].get("ports") or []
            for p in ports:
                if p["privatePort"] == 8000:
                    public_url = f"https://{pod_id}-8000.proxy.runpod.net/v1"
                    print(f"\n🚀 Votre API de Production AWQ (32k) est prête !")
                    print(f"👉 URL de Base : {public_url}")
                    print(f"👉 Modèle : {args.model_id}")
                    print(f"\nExemple de connexion en Python (OpenAI SDK) :")
                    print(f"----------------------------------------")
                    print(f"from openai import OpenAI")
                    print(f"client = OpenAI(base_url='{public_url}', api_key='none')")
                    print(f"response = client.chat.completions.create(")
                    print(f"    model='{args.model_id}',")
                    print(f"    messages=[{{'role': 'user', 'content': 'Quelle est la règle d'exclusion de la preuve en vertu de l'article 24(2) de la Charte ?'}}]")
                    print(f")")
                    print(f"print(response.choices[0].message.content)")
                    print(f"----------------------------------------")
                    return
            break
            
    print("\nLe pod démarre en tâche de fond. Retrouvez l'URL de proxy du port 8000 sur votre console RunPod.")

if __name__ == "__main__":
    main()
