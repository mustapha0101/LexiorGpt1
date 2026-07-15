#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de fine-tuning QLoRA standard utilisant l'écosystème Hugging Face (transformers, peft, trl).
Bypasse l'utilisation d'Unsloth pour éviter les conflits de dépendances système et pilotes.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement QLoRA standard avec Hugging Face.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        help="Modèle de base (ex. unsloth/Qwen2.5-32B-Instruct-bnb-4bit)."
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
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout pour LoRA.")
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
        help="Activer l'exportation du modèle final au format GGUF."
    )
    parser.add_argument(
        "--gguf_quantization", 
        type=str, 
        default="q4_k_m", 
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
    
    # Chargement du modèle. Le modèle est chargé via device_map="auto".
    # Si le modèle est déjà pré-quantisé (comme unsloth/Qwen2.5-32B-Instruct-bnb-4bit),
    # AutoModelForCausalLM le charge directement en utilisant bitsandbytes.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Préparation du modèle pour l'entraînement K-Bit...")
    model = prepare_model_for_kbit_training(model)
    
    print("Configuration des modules LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
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
            
    print("Initialisation du SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("test"),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
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
    resume_checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [os.path.join(args.output_dir, d) for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            resume_checkpoint = checkpoints[-1]
            print(f"Checkpoint trouvé, reprise de l'entraînement depuis : {resume_checkpoint}")
            
    trainer.train(resume_from_checkpoint=resume_checkpoint)

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
        print(f"Fusion des poids et sauvegarde du modèle complet dans : {merged_dir}...")
        
        # Pour fusionner, on charge le modèle de base non-quantifié 16-bit (Qwen/Qwen2.5-32B-Instruct)
        base_model_name = args.model_name
        if "bnb-4bit" in base_model_name:
            base_model_name = base_model_name.replace("unsloth/", "Qwen/").replace("-bnb-4bit", "")
            
        print(f"Chargement du modèle de base pour la fusion : {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        from peft import PeftModel
        model_to_merge = PeftModel.from_pretrained(base_model, lora_dir)
        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print("Fusion du modèle terminée avec succès !")
        
        if args.push_to_hub and args.hf_repo_id:
            print(f"Téléversement du modèle fusionné sur le Hugging Face Hub ({args.hf_repo_id}-merged)...")
            merged_model.push_to_hub(f"{args.hf_repo_id}-merged")
            tokenizer.push_to_hub(f"{args.hf_repo_id}-merged")
            
    if args.export_gguf:
        print("L'export GGUF nécessite llama.cpp. Vous pouvez convertir le modèle fusionné 16-bit en GGUF via llama.cpp localement.")
        
    print("Processus d'entraînement et d'export terminé !")

if __name__ == "__main__":
    main()
