#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de déploiement automatique pour RunPod.
Ce script utilise l'API de RunPod pour :
1. Créer une instance GPU (ex: RTX 4090 ou RTX 3090).
2. Lancer un conteneur avec l'image Docker optimisée d'Unsloth.
3. Injecter les variables d'environnement (Git, Hugging Face, Wandb).
4. Cloner le projet et lancer le pipeline de distillation automatiquement.
5. (Optionnel) Arrêter le pod à la fin de l'entraînement pour éviter les coûts inutiles.
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
    parser = argparse.ArgumentParser(description="Déploiement et orchestration automatique sur RunPod.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("RUNPOD_API_KEY", None),
        help="Votre clé API RunPod (peut aussi être définie via la variable RUNPOD_API_KEY)."
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="NVIDIA GeForce RTX 4090",
        choices=["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA A40", "NVIDIA A100-SXM4-80GB"],
        help="Type de GPU à allouer."
    )
    parser.add_argument(
        "--git_repo",
        type=str,
        required=True,
        help="URL de votre dépôt Git contenant le pipeline (ex: https://github.com/mon-username/DistillationModeles.git)."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Token d'écriture Hugging Face (optionnel, pour pousser le modèle sur le Hub)."
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default=os.environ.get("HF_REPO_ID", ""),
        help="ID du dépôt de destination sur Hugging Face (ex: username/llama3-juridique-cot)."
    )
    parser.add_argument(
        "--wandb_key",
        type=str,
        default=os.environ.get("WANDB_API_KEY", ""),
        help="Clé API Weights & Biases (optionnel, pour le tracking)."
    )
    parser.add_argument(
        "--autostop",
        action="store_true",
        help="Arrêter/Terminer automatiquement le Pod une fois l'entraînement fini pour économiser les coûts."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API RunPod est obligatoire. Définissez RUNPOD_API_KEY ou passez --api_key.")
        sys.exit(1)
        
    runpod.api_key = args.api_key
    
    # Configuration du conteneur
    # On utilise l'image officielle d'Unsloth pour éviter de recompiler les kernels Triton à chaque boot
    docker_image = "unslothdev/unsloth:latest"
    
    # Commande de démarrage : Cloner le dépôt et lancer run_pipeline.sh
    # On injecte les variables pour que le script d'automatisation les lise directement
    env_vars = {
        "HF_TOKEN": args.hf_token,
        "HF_REPO_ID": args.hf_repo_id,
        "WANDB_API_KEY": args.wandb_key,
        "TRACKING_REPORT_TO": "wandb" if args.wandb_key else "none",
        "DATASET_NAME": "SuperMust/irac-thinking",
        "MODEL_NAME": "unsloth/llama-3-8b-Instruct-bnb-4bit"
    }
    
    # Script exécuté au démarrage du conteneur Docker (mis sur une seule ligne pour éviter les erreurs GraphQL)
    container_command = f"bash -c 'echo \"Démarrage du conteneur RunPod...\" && git clone {args.git_repo} /workspace/DistillationModeles && cd /workspace/DistillationModeles && chmod +x run_pipeline.sh && ./run_pipeline.sh'"
    
    print(f"Création d'un pod sur RunPod ({args.gpu_type})...")
    
    try:
        # Lancement de l'instance
        pod = runpod.create_pod(
            name="lexior-distillation-cot",
            image_name=docker_image,
            gpu_type_id=args.gpu_type,
            gpu_count=1,
            volume_in_gb=50,
            container_disk_in_gb=30,
            ports="8888/http,22/tcp", # Ports pour Jupyter / SSH si besoin
            env=env_vars,
            docker_args=container_command
        )
    except Exception as e:
        print(f"Erreur lors de la création du Pod : {e}")
        sys.exit(1)
        
    pod_id = pod["id"]
    print(f"Pod créé avec succès ! ID du Pod : {pod_id}")
    print(f"Vous pouvez suivre l'état du pod sur : https://www.runpod.io/console/pods")
    
    # Boucle de suivi si l'autostop est activé
    if args.autostop:
        print("Suivi de l'avancement activé. Le Pod sera supprimé une fois l'entraînement terminé...")
        
        while True:
            time.sleep(60) # Vérification toutes les minutes
            try:
                pod_status = runpod.get_pod(pod_id)
                if not pod_status:
                    print("Le pod n'existe plus ou a été arrêté manuellement.")
                    break
                    
                status = pod_status.get("status")
                print(f"Statut actuel du Pod : {status}")
                
                # Vous pouvez implémenter une logique de lecture de logs pour détecter la fin,
                # mais le plus simple est de vérifier si le processus principal de commande est terminé.
                # Dans notre cas, si run_pipeline.sh se termine, le conteneur a fini sa tâche.
                # Si le statut passe à COMPLETED ou si la machine s'éteint :
                if status in ["COMPLETED", "STOPPED"]:
                    print("Entraînement et exports terminés !")
                    print(f"Suppression du Pod {pod_id} pour arrêter la facturation...")
                    runpod.terminate_pod(pod_id)
                    print("Pod supprimé avec succès.")
                    break
            except Exception as e:
                print(f"Erreur lors de la vérification du statut : {e}")
                
    else:
        print("\nDéploiement terminé. N'oubliez pas d'éteindre le Pod manuellement sur la console RunPod après l'entraînement.")

if __name__ == "__main__":
    main()
