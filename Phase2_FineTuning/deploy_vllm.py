#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de déploiement d'une instance vLLM de production pour LexiorGpt.
Ce script lance un pod sur RunPod utilisant l'image officielle vllm-openai
pour servir le modèle fusionné sous forme d'API compatible OpenAI.
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
    parser = argparse.ArgumentParser(description="Déploiement de vLLM de production sur RunPod.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("RUNPOD_API_KEY", None),
        help="Votre clé API RunPod."
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="NVIDIA A100-SXM4-80GB",
        help="Type de GPU à allouer (ex: 'NVIDIA A100-SXM4-80GB', 'NVIDIA A40')."
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
        default="intelliwork/LexiorGpt1-merged",
        help="ID du modèle fusionné sur Hugging Face."
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Type de quantification à appliquer pour l'inférence (facultatif)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Image officielle vLLM optimisée pour servir les API OpenAI
    docker_image = "vllm/vllm-openai:latest"
    
    # Injection des variables d'environnement
    # Rediriger le cache HF sur le grand volume /workspace (100 Go)
    env_vars = {
        "HF_TOKEN": args.hf_token,
        "HF_HOME": "/workspace/hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1" # Accélérer drastiquement le téléchargement du modèle
    }
    
    # Argument de commande pour vLLM (l'image Docker a déjà l'entrypoint vllm serve)
    vllm_cmd = [
        "--model", args.model_id,
        "--port", "8000",
        "--host", "0.0.0.0",
        "--dtype", "float16",
        "--max-model-len", "8192" # Taille maximale du contexte (ajustable)
    ]
    
    if args.quantization:
        vllm_cmd.extend(["--quantization", args.quantization])
        
    container_command = " ".join(vllm_cmd)
    
    print(f"Déploiement du modèle {args.model_id} avec vLLM sur {args.gpu_type}...")
    
    try:
        pod = runpod.create_pod(
            name="lexior-vllm-production",
            image_name=docker_image,
            gpu_type_id=args.gpu_type,
            gpu_count=1,
            volume_in_gb=100, # Espace suffisant pour stocker les 65 Go du modèle
            container_disk_in_gb=40,
            ports="8000/http,22/tcp",
            env=env_vars,
            docker_args=container_command
        )
    except Exception as e:
        print(f"Erreur lors de la création du Pod vLLM : {e}")
        sys.exit(1)
        
    pod_id = pod["id"]
    print(f"\n==================================================")
    print(f" Pod vLLM créé avec succès ! ID : {pod_id}")
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
                    print(f"\n🚀 Votre API de Production compatible OpenAI est prête !")
                    print(f"👉 URL de Base : {public_url}")
                    print(f"👉 Modèle : {args.model_id}")
                    print(f"\nExemple de connexion en Python (OpenAI SDK) :")
                    print(f"----------------------------------------")
                    print(f"from openai import OpenAI")
                    print(f"client = OpenAI(base_url='{public_url}', api_key='none')")
                    print(f"response = client.chat.completions.create(")
                    print(f"    model='{args.model_id}',")
                    print(f"    messages=[{{'role': 'user', 'content': 'Bonjour !'}}]")
                    print(f")")
                    print(f"print(response.choices[0].message.content)")
                    print(f"----------------------------------------")
                    return
            break
            
    print("\nLe pod démarre en tâche de fond. Retrouvez l'URL de proxy du port 8000 sur votre console RunPod.")

if __name__ == "__main__":
    main()
