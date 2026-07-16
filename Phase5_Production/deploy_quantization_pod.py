#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de déploiement automatique sur RunPod pour effectuer la quantification AWQ.
Ce script lance un pod GPU de calcul (RTX 4090), installe AutoAWQ, effectue le
calibrage avec notre jeu de données juridiques en français, et pousse le modèle
quantifié finalisé sur Hugging Face.
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
    parser = argparse.ArgumentParser(description="Déploiement RunPod pour la quantification AWQ de LexiorGPT.")
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
        help="Type de GPU de calcul à utiliser (ex: 'NVIDIA GeForce RTX 4090')."
    )
    parser.add_argument(
        "--git_repo",
        type=str,
        default="https://github.com/mustapha0101/LexiorGpt1.git",
        help="URL de votre dépôt Git contenant le projet."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Token d'écriture Hugging Face pour uploader le modèle."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="intelliwork/LexiorGpt1-merged",
        help="ID du modèle source FP16 sur Hugging Face."
    )
    parser.add_argument(
        "--repo_id_awq",
        type=str,
        default="intelliwork/LexiorGpt1-merged-AWQ",
        help="ID du dépôt de destination AWQ sur Hugging Face."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est requise.")
        sys.exit(1)
    if not args.hf_token:
        print("Erreur : Le token d'écriture Hugging Face (--hf_token) est requis.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Image PyTorch officielle sur RunPod contenant CUDA 12.4 et PyTorch >= 2.4
    docker_image = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    
    # Injection des tokens nécessaires dans les variables d'environnement du pod
    env_vars = {
        "HF_TOKEN": args.hf_token,
        "RUNPOD_API_KEY": args.api_key
    }
    
    # Commande du conteneur :
    # 1. Active SSH
    # 2. Clone le dépôt Git
    # 3. Installe AutoAWQ et les dépendances requises
    # 4. Exécute le script quantize_awq.py avec calibration juridique et upload HF

    container_command = (
        "bash -c '"
        "mkdir -p ~/.ssh && echo \"$PUBLIC_KEY\" > ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys; "
        "ssh-keygen -A && service ssh start || true; /usr/sbin/sshd || true; "
        "rm -rf /workspace/DistillationModeles && "
        f"git clone {args.git_repo} /workspace/DistillationModeles && "
        "cd /workspace/DistillationModeles && "
        "pip uninstall -y torchvision torchaudio && "
        "pip install --no-cache-dir transformers torch huggingface_hub && "
        "pip install git+https://github.com/casper-hansen/AutoAWQ.git && "
        "python3 Phase5_Production/quantize_awq.py "
        f"--model_path {args.model_id} "
        f"--quant_path /runpod-volume/outputs/final_model/quantized_awq_4bit "
        "--push_to_hub "
        f"--repo_id {args.repo_id_awq} "
        f"--hf_token {args.hf_token}; "
        "echo ==================================================; "
        "echo QUANTIFICATION-AWQ-TERMINEE-ET-ENVOYEE-SUR-HF; "
        "echo ==================================================; "
        "sleep infinity'"
    )
    
    print("==================================================")
    print(" LANCEMENT DU POD DE QUANTIFICATION AWQ SUR RUNPOD")
    print("==================================================")
    print(f"GPU d'exécution : {args.gpu_type}")
    print(f"Modèle FP16     : {args.model_id}")
    print(f"Destination AWQ : {args.repo_id_awq}")
    print(f"Dépôt Git       : {args.git_repo}")
    print("==================================================")
    
    try:
        pod = runpod.create_pod(
            name="lexior-awq-quantization-job",
            image_name=docker_image,
            gpu_type_id=args.gpu_type,
            gpu_count=1,
            volume_in_gb=200, # 200 Go pour être totalement serein (65 Go FP16 + 18 Go AWQ + Cache)
            container_disk_in_gb=50,
            ports="22/tcp",
            env=env_vars,
            docker_args=container_command
        )
    except Exception as e:
        print(f"Erreur lors du démarrage du Pod RunPod : {e}")
        sys.exit(1)
        
    pod_id = pod["id"]
    print(f"\n==================================================")
    print(f" Pod de quantification créé ! ID : {pod_id}")
    print(f"==================================================")
    print(f"Suivi de l'instance : https://www.runpod.io/console/pods")
    print(f"Logs en direct : Vous pouvez inspecter les logs de démarrage via l'interface RunPod.")
    print(f"\nUne fois le modèle AWQ uploadé sur Hugging Face, vous pourrez détruire ce pod et lancer deploy_vllm_awq.py.")

if __name__ == "__main__":
    main()
