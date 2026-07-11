#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de déploiement automatique de la Phase 1 (Génération de Données) sur RunPod.
Il loue une instance CPU/GPU légère et économique, clone le projet et lance
le script maître run_generation.sh, qui va générer les données et les envoyer
directement sur Hugging Face Hub.
"""

import os
import argparse
import sys

try:
    import runpod
except ImportError:
    print("La bibliothèque 'runpod' n'est pas installée.")
    print("Veuillez l'installer localement via : pip install runpod")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Déploiement de la génération CoT sur RunPod.")
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
        "--openai_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="Clé API du modèle Teacher (ex: OpenAI, Groq, etc.)."
    )
    parser.add_argument(
        "--openai_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="URL de base de l'API Teacher."
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="gpt-4o-mini",
        help="Modèle Teacher à utiliser (ex: gpt-4o-mini, llama3-70b-instruct)."
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
        help="Dépôt cible pour le dataset sur Hugging Face (ex: 'username/dataset-name')."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Nombre de documents A2AJ à traiter."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Nombre de threads parallèles d'appels API."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Image PyTorch légère (pas besoin de CUDA lourd pour de la génération d'API)
    docker_image = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
    
    # Définition des variables d'environnement pour le conteneur
    env_vars = {
        "OPENAI_API_KEY": args.openai_key,
        "OPENAI_BASE_URL": args.openai_url,
        "GEN_MODEL": args.teacher_model,
        "HF_TOKEN": args.hf_token,
        "HF_DATASET_REPO_ID": args.hf_dataset_repo,
        "GEN_LIMIT": str(args.limit),
        "GEN_WORKERS": str(args.workers)
    }
    
    # Commande de démarrage
    container_command = f"bash -c 'echo Demarrage_de_la_Phase1_Generation && git clone {args.git_repo} /workspace/DistillationModeles && cd /workspace/DistillationModeles/Phase1_Data_Preparation && chmod +x run_generation.sh && ./run_generation.sh'"
    
    # CPU léger ou GPU économique (ex: RTX 3070 ou CPU-only si supporté, RTX 3070 est parfaite pour rester pas cher)
    gpu_type = "NVIDIA GeForce RTX 3070"
    
    print(f"Création d'un pod de génération de données sur RunPod ({gpu_type})...")
    
    try:
        pod = runpod.create_pod(
            name="lexior-phase1-generation",
            image_name=docker_image,
            gpu_type_id=gpu_type,
            gpu_count=1,
            volume_in_gb=30,
            container_disk_in_gb=20,
            ports="8888/http,22/tcp",
            env=env_vars,
            docker_args=container_command
        )
    except Exception as e:
        print(f"Erreur lors de la création du Pod : {e}")
        sys.exit(1)
        
    print(f"Pod de génération créé avec succès ! ID : {pod['id']}")
    print(f"Vous pouvez suivre les logs de génération sur : https://www.runpod.io/console/pods")
    print(f"Une fois terminé, le dataset sera poussé sur Hugging Face : {args.hf_dataset_repo}")

if __name__ == "__main__":
    main()
