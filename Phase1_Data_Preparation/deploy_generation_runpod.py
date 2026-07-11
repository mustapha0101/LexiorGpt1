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
        default=os.environ.get("OPENAI_API_KEY", "ollama"),
        help="Clé API du modèle Teacher."
    )
    parser.add_argument(
        "--openai_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        help="URL de base de l'API Teacher (local Ollama par défaut)."
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="qwen2.5:32b-instruct-q4_K_M",
        help="Modèle Teacher à utiliser (Ollama)."
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
        default=1000,
        help="Nombre de documents A2AJ à traiter."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Nombre de threads parallèles d'appels API (8-10 max pour Ollama en local)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Image PyTorch officielle avec CUDA
    docker_image = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
    
    # Définition des variables d'environnement pour le conteneur
    env_vars = {
        "OPENAI_API_KEY": args.openai_key,
        "OPENAI_BASE_URL": args.openai_url,
        "GEN_MODEL": args.teacher_model,
        "HF_TOKEN": args.hf_token,
        "HF_DATASET_REPO_ID": args.hf_dataset_repo,
        "GEN_LIMIT": str(args.limit),
        "GEN_WORKERS": str(args.workers),
        "USE_LOCAL_OLLAMA": "true"
    }
    
    # Commande de démarrage (installe git/curl, nettoie et clone le repo)
    container_command = f"bash -c 'apt-get update && apt-get install -y git curl && rm -rf /workspace/DistillationModeles && git clone {args.git_repo} /workspace/DistillationModeles && cd /workspace/DistillationModeles/Phase1_Data_Preparation && chmod +x run_generation.sh && ./run_generation.sh'"
    
    # Ordre de préférence des GPU : RTX 4090 pour la vitesse, puis RTX 3090, puis A40
    gpu_preferences = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA A40"]
    pod = None
    
    for gpu_type in gpu_preferences:
        print(f"Tentative de création du pod sur RunPod avec le GPU : {gpu_type}...")
        try:
            pod = runpod.create_pod(
                name="lexior-phase1-generation",
                image_name=docker_image,
                gpu_type_id=gpu_type,
                gpu_count=1,
                volume_in_gb=40, # Augmenté à 40 Go pour loger le modèle 32B et Ollama confortablement
                container_disk_in_gb=20,
                ports="8888/http,22/tcp",
                env=env_vars,
                docker_args=container_command
            )
            print(f"Succès ! Pod alloué sur GPU : {gpu_type}")
            break
        except Exception as e:
            error_msg = str(e)
            if "resources" in error_msg.lower() or "availability" in error_msg.lower():
                print(f"Ressources insuffisantes pour {gpu_type}, tentative avec le GPU suivant...")
            else:
                print(f"Erreur lors de la tentative avec {gpu_type} : {e}")
                
    if not pod:
        print("Erreur critique : Impossible de créer le Pod sur l'un des GPU configurés.")
        sys.exit(1)
        
    print(f"Pod de génération créé avec succès ! ID : {pod['id']}")
    print(f"Vous pouvez suivre les logs de génération sur : https://www.runpod.io/console/pods")
    print(f"Une fois terminé, le dataset sera poussé sur Hugging Face : {args.hf_dataset_repo}")

if __name__ == "__main__":
    main()
