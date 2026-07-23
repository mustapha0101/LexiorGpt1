#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour pousser un dataset généré de la Phase 1 vers le Hugging Face Hub.

Deux usages :

1. Pousser un corpus généré, par source (dépôts privés). Le dépôt est déduit du
   fichier si --repo_id est omis (cf. DEFAULT_REPOS) :

     python3 push_to_hf.py --input_file data/processed/generated_a2aj_cot.jsonl
         -> intelliwork/canadian-cot-dataset-federal-french

     python3 push_to_hf.py --input_file data/processed/generated_ccq_cot.jsonl
         -> intelliwork/canadian-cot-dataset-quebec-french

     python3 push_to_hf.py --input_file data/processed/generated_identity_cot.jsonl
         -> intelliwork/canadian-cot-dataset-identity-french

2. Sans argument : comportement historique — pousse le couple
   train/test formaté de data/processed/ vers $HF_DATASET_REPO_ID.
"""

import argparse
import os
import sys

from datasets import load_dataset

# Dépôts par défaut, par source de génération.
DEFAULT_REPOS = {
    "data/processed/generated_ccq_cot.jsonl": "intelliwork/canadian-cot-dataset-quebec-french",
    "data/processed/generated_a2aj_cot.jsonl": "intelliwork/canadian-cot-dataset-federal-french",
    "data/processed/generated_identity_cot.jsonl": "intelliwork/canadian-cot-dataset-identity-french",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Téléversement d'un dataset de la Phase 1 sur le Hugging Face Hub."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Fichier JSONL à téléverser. Par défaut : le couple train/test formaté."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Dépôt cible. Par défaut : déduit de --input_file, sinon $HF_DATASET_REPO_ID."
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Publier en public. Par défaut le dépôt est PRIVÉ."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Erreur : la variable d'environnement HF_TOKEN n'est pas définie.")
        sys.exit(1)

    # --- Résolution du dépôt cible ---------------------------------------
    repo_id = args.repo_id
    if not repo_id and args.input_file:
        repo_id = DEFAULT_REPOS.get(args.input_file.replace("\\", "/"))
    if not repo_id:
        repo_id = os.environ.get("HF_DATASET_REPO_ID")
    if not repo_id:
        print("Erreur : aucun dépôt cible. Utiliser --repo_id, ou définir "
              "HF_DATASET_REPO_ID, ou passer un --input_file connu de DEFAULT_REPOS.")
        sys.exit(1)

    # --- Résolution des fichiers -----------------------------------------
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Erreur : le fichier '{args.input_file}' est introuvable.")
            sys.exit(1)
        data_files = {"train": args.input_file}
    else:
        train_path = "data/processed/train_dataset.jsonl"
        test_path = "data/processed/test_dataset.jsonl"
        if not os.path.exists(train_path):
            print(f"Erreur : Le fichier d'entraînement '{train_path}' est introuvable.")
            sys.exit(1)
        data_files = {"train": train_path}
        if os.path.exists(test_path):
            data_files["test"] = test_path

    print(f"Chargement des fichiers locaux : {data_files}")
    dataset = load_dataset("json", data_files=data_files)
    for split in dataset:
        print(f"  {split} : {len(dataset[split]):,} lignes")

    private = not args.public
    print(f"Téléversement sur Hugging Face Hub : {repo_id} "
          f"({'privé' if private else 'PUBLIC'})...")
    dataset.push_to_hub(repo_id, token=token, private=private)
    print("Dataset téléversé avec succès sur Hugging Face !")


if __name__ == "__main__":
    main()
