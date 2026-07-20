#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de fine-tuning QLoRA pour Llama 3 / Qwen 2.5 avec Unsloth.
Prend en charge l'entraînement CoT (Chain-of-Thought) et l'exportation automatique
des adapters LoRA et du modèle au format GGUF ou fusionné 16-bit.
"""

import os
import argparse
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from assistant_loss import AssistantOnlyDataCollator, STRICT_CHATML_TEMPLATE, encode_example_assistant_only

# Importations Unsloth
try:
    from unsloth import FastLanguageModel
    import torch
except ImportError:
    raise ImportError(
        "Unsloth ou PyTorch n'est pas installé. Ce script doit être exécuté sur un environnement avec GPU NVIDIA "
        "et les bibliothèques CUDA appropriées installées."
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement QLoRA avec Unsloth.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        help="Modèle de base Unsloth (ex. unsloth/Qwen2.5-32B-Instruct-bnb-4bit)."
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="../Phase1_Data_Preparation/data/processed/train_dataset.jsonl",
        help="Chemin vers le fichier JSONL de train formaté."
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="../Phase1_Data_Preparation/data/processed/test_dataset.jsonl",
        help="Chemin vers le fichier JSONL de test formaté (facultatif)."
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=4096,
        help="Longueur maximale de séquence (context length)."
    )
    parser.add_argument("--lora_r", type=int, default=16, help="Rang de LoRA (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha de LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout pour LoRA (0.0 est recommandé par Unsloth).")
    parser.add_argument("--epochs", type=int, default=3, help="Nombre d'époques d'entraînement.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size par GPU.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Nombre d'étapes d'accumulation de gradient.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints", help="Dossier de sortie pour les checkpoints.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Intervalle d'affichage des logs.")
    parser.add_argument("--save_steps", type=int, default=100, help="Intervalle de sauvegarde des checkpoints.")
    parser.add_argument(
        "--export_dir", 
        type=str, 
        default="outputs/final_model",
        help="Dossier final pour sauvegarder les adapters LoRA."
    )
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Téléverser le modèle sur Hugging Face Hub."
    )
    parser.add_argument(
        "--hf_repo_id", 
        type=str, 
        default=None,
        help="Identifiant du dépôt Hugging Face (ex. 'mon-username/llama-3-8b-juridique')."
    )
    parser.add_argument(
        "--export_gguf", 
        action="store_true", 
        help="Activer l'exportation du modèle final au format GGUF pour Ollama/Llama.cpp."
    )
    parser.add_argument(
        "--gguf_quantization", 
        type=str, 
        default="q4_k_m", 
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="Méthode de quantification GGUF."
    )
    parser.add_argument(
        "--export_merged_16bit", 
        action="store_true", 
        help="Exporter le modèle fusionné en float16 (Poids complets)."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Framework à utiliser pour suivre l'entraînement."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="qwen25-juridique-cot",
        help="Nom de l'expérience de fine-tuning."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Chargement du modèle de base : {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Détection automatique de float16/bfloat16
        load_in_4bit=True,
    )
    
    tokenizer.padding_side = "right"
    tokenizer.chat_template = STRICT_CHATML_TEMPLATE

    print("Configuration des modules LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )
    
    print(f"Chargement des données d'entraînement depuis {args.train_file}...")
    if args.train_file.startswith("intelliwork/"):
        dataset_dict = {
            "train": load_dataset(args.train_file, split="train", token=os.environ.get("HF_TOKEN")),
            "test": load_dataset(args.train_file, split="test", token=os.environ.get("HF_TOKEN"))
        }
    else:
        dataset_dict = {"train": load_dataset("json", data_files=args.train_file, split="train")}
        if args.test_file and os.path.exists(args.test_file):
            print(f"Chargement des données de test depuis {args.test_file}...")
            dataset_dict["test"] = load_dataset("json", data_files=args.test_file, split="train")
    
    print("Construction explicite des labels assistant-only...")
    for split_name, split in list(dataset_dict.items()):
        dataset_dict[split_name] = split.map(
            lambda row: encode_example_assistant_only(row, tokenizer, args.max_seq_length),
            remove_columns=split.column_names,
            desc=f"Assistant-only labels ({split_name})",
        )
    print("Initialisation du Trainer avec labels explicites...")
    trainer = Trainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("test"),
        data_collator=AssistantOnlyDataCollator(tokenizer),
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.save_steps if "test" in dataset_dict else None,
            evaluation_strategy="steps" if "test" in dataset_dict else "no",
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=args.output_dir,
            report_to=args.report_to,
            run_name=args.run_name if args.report_to != "none" else None,
            logging_dir="outputs/logs"
        ),
    )
    
    print("Début du Fine-Tuning...")
    trainer_stats = trainer.train()
    print("Entraînement terminé !")
    
    os.makedirs(args.export_dir, exist_ok=True)
    lora_dir = os.path.join(args.export_dir, "lora_adapters")
    print(f"Sauvegarde locale des adapters LoRA dans : {lora_dir}")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    
    if args.push_to_hub and args.hf_repo_id:
        print(f"Téléversement des adapters LoRA sur le Hugging Face Hub ({args.hf_repo_id})...")
        model.push_to_hub(args.hf_repo_id, commit_message="Ajout des adapters LoRA spécialisés")
        tokenizer.push_to_hub(args.hf_repo_id)

    if args.export_merged_16bit:
        merged_dir = os.path.join(args.export_dir, "merged_16bit")
        print(f"Fusion des poids et sauvegarde du modèle complet 16-bit dans : {merged_dir}...")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        if args.push_to_hub and args.hf_repo_id:
            model.push_to_hub_merged(f"{args.hf_repo_id}-merged", tokenizer, save_method="merged_16bit")
            
    if args.export_gguf:
        gguf_dir = os.path.join(args.export_dir, f"gguf_{args.gguf_quantization}")
        print(f"Exportation directe au format GGUF ({args.gguf_quantization}) dans : {gguf_dir}...")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=args.gguf_quantization)
        if args.push_to_hub and args.hf_repo_id:
            model.push_to_hub_gguf(f"{args.hf_repo_id}-gguf", tokenizer, quantization_method=args.gguf_quantization)
            
    print("Processus d'entraînement et d'export terminé !")

if __name__ == "__main__":
    main()
