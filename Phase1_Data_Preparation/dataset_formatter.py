#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de formatage du dataset pour la distillation Chain-of-Thought (CoT).
Il télécharge le dataset SuperMust/irac-thinking ou lit un fichier local généré par generator_a2aj.py,
fusionne le champ 'thinking' et le champ 'content' de l'assistant dans des balises <thinking>,
et applique le chat template.
"""

import os
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Formatage du dataset pour la distillation CoT.")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="SuperMust/irac-thinking",
        help="Nom du dataset sur Hugging Face Hub (ou chemin local vers un fichier JSON/JSONL)."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Nom du modèle pour extraire le tokenizer et le chat template."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed",
        help="Dossier de sauvegarde des fichiers formatés."
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.05, 
        help="Proportion de données pour le jeu de test (0.0 pour désactiver le split)."
    )
    parser.add_argument(
        "--local_file",
        type=str,
        default=None,
        help="Chemin local vers un fichier JSONL brut à formater."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Création des répertoires de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Chargement du tokenizer pour appliquer le template
    print(f"Chargement du tokenizer pour '{args.model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Erreur lors du chargement du tokenizer : {e}")
        print("Tentative de chargement d'un tokenizer générique Qwen...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    # 3. Chargement du dataset
    if args.local_file:
        print(f"Chargement du dataset local depuis {args.local_file}...")
        dataset = load_dataset("json", data_files=args.local_file, split="train")
    else:
        print(f"Chargement du dataset depuis Hugging Face Hub : '{args.dataset_name}'...")
        try:
            dataset = load_dataset(args.dataset_name, split="train")
        except Exception as e:
            print(f"Erreur lors du chargement de '{args.dataset_name}' : {e}")
            print("Tentative de chargement en tant que jeu brut sans spécifier de split...")
            dataset = load_dataset(args.dataset_name)
            if hasattr(dataset, "keys"):
                first_key = list(dataset.keys())[0]
                print(f"Utilisation du split : {first_key}")
                dataset = dataset[first_key]

    print(f"Nombre d'exemples initiaux : {len(dataset)}")

    # 4. Fonction de formatage CoT
    def format_cot_dataset(example):
        messages_bruts = example.get("messages", [])
        
        if not messages_bruts:
            for key in ["conversations", "dialogue", "chat"]:
                if key in example:
                    messages_bruts = example[key]
                    break
        
        if not messages_bruts:
            instruction = example.get("instruction", example.get("prompt", ""))
            input_context = example.get("input", "")
            output = example.get("output", example.get("response", ""))
            thinking = example.get("thinking", "")
            
            if input_context:
                user_content = f"{instruction}\n\nContexte :\n{input_context}"
            else:
                user_content = instruction
                
            messages_bruts = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output, "thinking": thinking}
            ]

        messages_formates = []
        for msg in messages_bruts:
            role = msg.get("role", msg.get("from", ""))
            if role in ["developer", "system"]:
                role = "system"
            elif role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant"]:
                role = "assistant"
                
            content = msg.get("content", msg.get("value", ""))
            thinking = msg.get("thinking", "")
            
            # Injection du contexte de droit canadien/québécois dans le prompt système
            if role == "system":
                content = (
                    "Tu es un assistant juridique Lexior, spécialisé en droit canadien et québécois. "
                    "Raisonne en français selon le format IRAC. Tu dois obligatoirement baser tes analyses "
                    "sur la législation et la jurisprudence canadienne/québécoise (ex: Code civil du Québec, CanLII). "
                    "Lorsque tu as fini de raisonner dans tes balises <thinking>, formate tes citations de bas de page "
                    "strictement sous la forme [^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/...\",\"title\":\"Titre\"}."
                )
            
            # Injection de la réflexion pour l'assistant
            if role == "assistant":
                if thinking:
                    content_final = f"<thinking>\n{thinking}\n</thinking>\n\n{content}"
                else:
                    content_final = content
                messages_formates.append({"role": role, "content": content_final})
            else:
                messages_formates.append({"role": role, "content": content})
                
        text = tokenizer.apply_chat_template(
            messages_formates,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": text}

    # 5. Mapping du dataset
    print("Application du formatage CoT...")
    mapped_dataset = dataset.map(
        format_cot_dataset,
        remove_columns=dataset.column_names,
        desc="Formatting dataset to Llama-3/Qwen CoT format"
    )

    # 6. Split Train / Test
    if args.test_size > 0.0:
        print(f"Division du dataset (Test size = {args.test_size})...")
        split_dataset = mapped_dataset.train_test_split(test_size=args.test_size, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
    else:
        train_dataset = mapped_dataset
        test_dataset = None

    # 7. Sauvegarde locale
    train_path = os.path.join(args.output_dir, "train_dataset.jsonl")
    print(f"Sauvegarde du jeu d'entraînement dans {train_path} ({len(train_dataset)} exemples)...")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    if test_dataset:
        test_path = os.path.join(args.output_dir, "test_dataset.jsonl")
        print(f"Sauvegarde du jeu de test dans {test_path} ({len(test_dataset)} exemples)...")
        with open(test_path, "w", encoding="utf-8") as f:
            for item in test_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Formatage terminé avec succès !")

if __name__ == "__main__":
    main()
