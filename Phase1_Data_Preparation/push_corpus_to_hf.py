#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de téléversement du corpus de lois structurées sur Hugging Face Hub.
Lit le corpus local généré dans data/law_corpus et le pousse sous forme de Dataset HF.
"""

import os
import json
import argparse
from datasets import Dataset

DEFAULT_REPO_ID = "intelliwork/canadian-quebec-law-corpus"

def parse_args():
    parser = argparse.ArgumentParser(description="Pousse le corpus juridique sur Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"ID du dépôt du dataset sur Hugging Face (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Jeton d'authentification Hugging Face (HF Write Token)"
    )
    return parser.parse_args()

def push_to_hub():
    args = parse_args()
    
    if not args.token:
        print("❌ Erreur : Jeton Hugging Face manquant. Exportez-le (export HF_TOKEN='...') ou passez-le en paramètre (--token).")
        return
        
    index_path = "data/law_corpus/index.json"
    
    if not os.path.exists(index_path):
        print(f"❌ Erreur : Fichier index.json introuvable à '{index_path}'. Lancez d'abord compile_local_corpus.py.")
        return
        
    print(f"Chargement de l'index du corpus depuis {index_path}...")
    with open(index_path, "r", encoding="utf-8") as f:
        corpus_index = json.load(f)
        
    records = []
    print(f"Lecture et conversion de {len(corpus_index)} articles de lois...")
    
    for idx, item in enumerate(corpus_index):
        file_path = item["path"]
        if not os.path.exists(file_path):
            print(f"⚠ Fichier d'article manquant : {file_path}")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f_art:
            art_data = json.load(f_art)
            
        records.append({
            "id": item["id"],
            "title": item["title"],
            "article": art_data["article"],
            "code": art_data["code"],
            "jurisdiction": art_data["juridiction"],
            "texte": art_data["texte"],
            "chemin_taxonomy": art_data["chemin_taxonomy"]
        })
        
    # Créer le Dataset Hugging Face
    print("Création du Dataset Hugging Face...")
    hf_dataset = Dataset.from_list(records)
    
    # Téléverser sur Hugging Face Hub
    print(f"Téléversement du dataset sur Hugging Face : '{args.repo_id}'...")
    try:
        hf_dataset.push_to_hub(
            repo_id=args.repo_id,
            token=args.token,
            private=True # Nous le créons en privé par défaut pour la confidentialité
        )
        print(f"🎉 Succès ! Le corpus juridique canadien-québécois est en ligne : https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"❌ Échec du téléversement sur le Hugging Face Hub : {e}")

if __name__ == "__main__":
    push_to_hub()
