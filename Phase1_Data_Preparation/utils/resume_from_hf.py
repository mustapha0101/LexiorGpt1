#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de coordination pour la reprise de génération depuis Hugging Face.
Il télécharge le dernier checkpoint JSONL depuis votre dépôt Hugging Face,
l'installe localement, puis lance le générateur parallèle en sautant les exemples déjà faits.
"""

import os
import sys
import argparse
import subprocess
import shutil
from huggingface_hub import hf_hub_download

def parse_args():
    parser = argparse.ArgumentParser(description="Télécharge le dernier checkpoint depuis Hugging Face et lance la génération.")
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=os.environ.get("HF_DATASET_REPO_ID", ""),
        help="Dépôt Hugging Face du dataset."
    )
    parser.add_argument(
        "--hf_file",
        type=str,
        default="data/backup_95_examples.jsonl",
        help="Chemin du fichier JSONL de backup dans le dépôt HF."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/generated_a2aj_cot.jsonl",
        help="Chemin de sortie local pour la génération."
    )
    # Récupérer les autres arguments non définis pour les transmettre à generator_a2aj.py
    return parser.parse_known_args()

def main():
    args, unknown_args = parse_args()
    
    token = os.environ.get("HF_TOKEN")
    
    if not args.hf_repo:
        print("Erreur : La variable d'environnement HF_DATASET_REPO_ID ou l'argument --hf_repo est requis.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 1. Tenter de télécharger le fichier de backup depuis HF si le fichier local n'existe pas encore ou est vide
    if not os.path.exists(args.output_file) or os.path.getsize(args.output_file) == 0:
        print(f"Téléchargement du checkpoint depuis HF ({args.hf_repo}/{args.hf_file})...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=args.hf_repo,
                filename=args.hf_file,
                repo_type="dataset",
                token=token
            )
            # Copier le fichier téléchargé vers l'emplacement de travail attendu par generator_a2aj.py
            shutil.copy(downloaded_path, args.output_file)
            print(f"Fichier de checkpoint téléchargé et copié avec succès vers '{args.output_file}'")
        except Exception as e:
            print(f"\nAvertissement : Impossible de charger le checkpoint depuis HF ({e}).")
            print("La génération démarrera à zéro (aucun fichier local détecté).")
    else:
        print(f"Fichier de travail local détecté à '{args.output_file}' ({os.path.getsize(args.output_file)} octets). Reprise locale.")

    # 2. Exécuter generator_a2aj.py en propageant le fichier de sortie et tous les autres arguments
    cmd = [sys.executable, "generator_a2aj.py", "--output_file", args.output_file] + unknown_args
    print(f"Lancement de la génération : {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nErreur pendant l'exécution de generator_a2aj.py (code de retour {e.returncode}).")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
