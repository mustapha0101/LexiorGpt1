#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour pousser le dataset généré de la Phase 1 vers le Hugging Face Hub.
"""

import os
import sys
from datasets import load_dataset

def main():
    train_path = "data/processed/train_dataset.jsonl"
    test_path = "data/processed/test_dataset.jsonl"
    
    repo_id = os.environ.get("HF_DATASET_REPO_ID")
    token = os.environ.get("HF_TOKEN")
    
    if not repo_id:
        print("Erreur : La variable d'environnement HF_DATASET_REPO_ID n'est pas définie.")
        sys.exit(1)
        
    if not os.path.exists(train_path):
        print(f"Erreur : Le fichier d'entraînement '{train_path}' est introuvable.")
        sys.exit(1)
        
    print(f"Chargement des fichiers locaux depuis 'data/processed'...")
    data_files = {"train": train_path}
    if os.path.exists(test_path):
        data_files["test"] = test_path
        
    dataset = load_dataset("json", data_files=data_files)
    
    print(f"Téléversement du dataset sur Hugging Face Hub : {repo_id}...")
    dataset.push_to_hub(repo_id, token=token, private=True)
    print("Dataset téléversé avec succès sur Hugging Face !")

if __name__ == "__main__":
    main()
