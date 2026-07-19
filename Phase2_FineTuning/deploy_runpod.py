#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de déploiement automatique de la Phase 2 (Fine-Tuning) sur RunPod.
Ce script utilise l'API de RunPod pour louer une instance GPU, configurer les clés
et cloner votre projet pour lancer directement l'entraînement.
"""

import os
import time
import argparse
import sys

try:
    import runpod
except ImportError:
    print("La bibliothèque 'runpod' n'est pas installée.")
    print("Veuillez l'installer localement via : pip install runpod")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Déploiement automatique du Fine-Tuning sur RunPod.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("RUNPOD_API_KEY", None),
        help="Votre clé API RunPod."
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="NVIDIA GeForce RTX 3090",
        help="Type de GPU à allouer (ex: 'NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 4090')."
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
        "--hf_repo_id",
        type=str,
        default=os.environ.get("HF_REPO_ID", ""),
        help="ID du dépôt de destination sur Hugging Face."
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=os.environ.get("WANDB_API_KEY", ""),
        help="Clé API Weights & Biases."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Nom du modèle de base à fine-tuner."
    )
    parser.add_argument(
        "--only_merge",
        action="store_true",
        help="Déployer uniquement pour exécuter le script de fusion et d'upload."
    )
    return parser.parse_args()
 
def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Image officielle RunPod PyTorch pré-configurée avec Git, Curl et SSH
    docker_image = "runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
    
    # Injection des variables d'environnement
    env_vars = {
        "HF_TOKEN": args.hf_token,
        "HF_REPO_ID": args.hf_repo_id,
        "WANDB_API_KEY": args.wandb_key,
        "WANDB_PROJECT": "huggingface",
        "WANDB_ENTITY": "lexiorgpt-intelliwork",
        "TRACKING_REPORT_TO": "wandb" if args.wandb_key else "none",
        "TRACKING_RUN_NAME": "qwen25-canadian-cot",
        "MODEL_NAME": args.model_name
    }

    # Ajustement des configurations selon le mode (entraînement ou fusion simple)
    if args.only_merge:
        container_command = f"bash -c 'ssh-keygen -A && service ssh start || true; /usr/sbin/sshd || true; rm -rf /workspace/DistillationModeles && git clone {args.git_repo} /workspace/DistillationModeles && cd /workspace/DistillationModeles/Phase2_FineTuning && pip uninstall -y torchvision torchaudio && pip install --no-cache-dir huggingface_hub transformers peft accelerate bitsandbytes sentencepiece protobuf && python3 merge_and_upload.py; sleep infinity'"
        volume_size = 300
        pod_name = "lexior-phase2-merge"
    else:
        container_command = f"bash -c 'ssh-keygen -A && service ssh start || true; /usr/sbin/sshd || true; rm -rf /workspace/DistillationModeles && git clone {args.git_repo} /workspace/DistillationModeles && cd /workspace/DistillationModeles/Phase2_FineTuning && chmod +x run_training.sh && ./run_training.sh; sleep infinity'"
        volume_size = 300
        pod_name = "lexior-phase2-finetuning"
    
    print(f"Création d'un pod sur RunPod ({args.gpu_type})...")
    
    try:
        pod = runpod.create_pod(
            name=pod_name,
            image_name=docker_image,
            gpu_type_id=args.gpu_type,
            gpu_count=1,
            volume_in_gb=volume_size,
            container_disk_in_gb=40,
            ports="8888/http,22/tcp",
            env=env_vars,
            docker_args=container_command
        )
    except Exception as e:
        print(f"Erreur lors de la création du Pod : {e}")
        sys.exit(1)
        
    pod_id = pod["id"]
    print(f"Pod créé avec succès ! ID : {pod_id}")
    print(f"Vous pouvez suivre l'état du pod sur : https://www.runpod.io/console/pods")
    
    print("\nDéploiement terminé. Pensez à arrêter le Pod manuellement après l'entraînement ou la fusion.")

if __name__ == "__main__":
    main()
