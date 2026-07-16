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
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Longueur maximale du contexte (ex: 32768, 65536, 131072)."
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=1,
        help="Nombre de GPU à allouer (nécessaire >1 pour les contextes très longs comme 128k sur des GPU de 24G/48G)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Image officielle vLLM (v0.5.2 utilise CUDA 12.1, ce qui évite les erreurs CUDA 13 incompatibles)
    docker_image = "vllm/vllm-openai:v0.5.2"
    
    # Cache HF sur le grand volume persistant /runpod-volume
    env_vars = {
        "HF_TOKEN": args.hf_token,
        "HF_HOME": "/runpod-volume/hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1" # Autorise à dépasser la longueur de contexte de base dans vLLM
    }
    
    # Arguments optimisés pour le modèle AWQ avec contexte dynamique de 128k
    vllm_cmd = [
        "--model", args.model_id,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--quantization", "awq",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", "0.90", # Conserve 10% pour les activations de contexte
        "--kv-cache-dtype", "fp8",          # Indispensable pour stocker le KV cache 128k en VRAM
        "--rope-scaling", "'{\\\"type\\\":\\\"yarn\\\",\\\"factor\\\":4.0,\\\"original_max_position_embeddings\\\":32768}'",
        "--tokenizer-mode", "slow"
    ]
    
    # Configuration Tensor Parallel si multi-GPU
    if args.gpu_count > 1:
        vllm_cmd.extend(["--tensor-parallel-size", str(args.gpu_count)])
        
    container_command = " ".join(vllm_cmd)
    
    print("==================================================")
    print(" DÉPLOIEMENT DU POD DE PRODUCTION LEXIORGPT AWQ")
    print("==================================================")
    print(f"GPU Cible       : {args.gpu_type} (x{args.gpu_count})")
    print(f"Modèle AWQ      : {args.model_id}")
    print(f"Longueur Max    : {args.max_model_len} tokens")
    if args.gpu_count > 1:
        print(f"Tensor Parallel : {args.gpu_count}")
    print("==================================================")
    
    try:
        pod = runpod.create_pod(
            name="lexior-vllm-prod-awq",
            image_name=docker_image,
            gpu_type_id=args.gpu_type,
            gpu_count=args.gpu_count,
            volume_in_gb=150, # 150 Go pour être totalement à l'aise (modèle de 18 Go + KV cache)
            container_disk_in_gb=50,
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
