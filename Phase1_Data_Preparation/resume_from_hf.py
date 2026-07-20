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
        default=os.environ.get("HF_REPO_FEDERAL")
                or os.environ.get("HF_DATASET_REPO_ID", "")
                or "intelliwork/canadian-cot-dataset-federal-french",
        help="Dépôt Hugging Face servant de point de reprise."
    )
    parser.add_argument(
        "--hf_file",
        type=str,
        default=None,
        help="Chemin d'un JSONL brut dans le dépôt HF. Par défaut, le dépôt est "
             "lu comme un dataset (parquet) — c'est le format écrit par push_to_hf.py."
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
    
    # 1. Tenter de télécharger le checkpoint depuis HF si le fichier local n'existe pas encore ou est vide
    if not os.path.exists(args.output_file) or os.path.getsize(args.output_file) == 0:
        print(f"Aucun fichier local. Récupération du checkpoint depuis HF ({args.hf_repo})...")
        try:
            if args.hf_file:
                # Chemin explicite vers un JSONL brut (ancien fonctionnement).
                downloaded_path = hf_hub_download(
                    repo_id=args.hf_repo,
                    filename=args.hf_file,
                    repo_type="dataset",
                    token=token
                )
                shutil.copy(downloaded_path, args.output_file)
            else:
                # Le dépôt est un dataset écrit par push_to_hf.py (parquet) : on
                # le relit et on le remet à plat en JSONL, format attendu par
                # l'index de reprise de generator_a2aj.py.
                from datasets import load_dataset
                ds = load_dataset(args.hf_repo, split="train", token=token)
                ds.to_json(args.output_file, orient="records", lines=True, force_ascii=False)
            n = sum(1 for _ in open(args.output_file, encoding="utf-8"))
            print(f"Checkpoint récupéré : {n} lignes écrites dans '{args.output_file}'.")
        except Exception as e:
            print(f"\nAvertissement : Impossible de charger le checkpoint depuis HF ({e}).")
            print("La génération démarrera à zéro (aucun fichier local détecté).")
    else:
        n = sum(1 for _ in open(args.output_file, encoding="utf-8"))
        print(f"Fichier de travail local détecté à '{args.output_file}' ({n} lignes). Reprise locale.")

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
