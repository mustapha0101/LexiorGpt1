#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de quantification AWQ 4-bit pour LexiorGPT.
Ce script charge le modèle fusionné en FP16 complet et applique la quantification AWQ
en calibrant le modèle sur un échantillon représentatif de notre corpus juridique québécois/canadien.
"""

import os
import sys
import json
import argparse
import time

try:
    import torch
    from transformers import AutoTokenizer
except ImportError:
    print("Erreur : La bibliothèque 'transformers' et 'torch' doivent être installées.")
    print("Veuillez installer les dépendances requises : pip install transformers torch")
    sys.exit(1)

try:
    from awq import AutoAWQForCausalLM
except ImportError as e:
    import traceback
    print("Erreur lors de l'importation de 'autoawq' :")
    traceback.print_exc()
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantification AWQ 4-bit pour LexiorGPT")
    
    # Chemins des modèles
    volume_path = "/runpod-volume" if os.path.exists("/runpod-volume") else "/workspace"
    default_model_path = os.path.join(volume_path, "outputs/final_model/merged_16bit")
    
    if not os.path.exists(default_model_path):
        # Si le dossier local n'existe pas, cibler par défaut le repo Hugging Face
        default_model_path = "intelliwork/LexiorGpt1-merged"
        
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_model_path,
        help="Chemin local ou identifiant Hugging Face du modèle FP16 fusionné."
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default=os.path.join(volume_path, "outputs/final_model/quantized_awq_4bit"),
        help="Chemin de sauvegarde du modèle quantifié."
    )
    
    # Jeu de données de calibration
    # Tenter de trouver le jeu de données par défaut
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    default_dataset = os.path.join(project_root, "data/processed/generated_a2aj_cot_reconstructed.jsonl")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_dataset,
        help="Chemin vers le jeu de données de calibration JSONL."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Nombre d'exemples de calibration à utiliser (128 par défaut, recommandé 128-256)."
    )
    
    # Options Hugging Face Hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Activer le téléversement direct du modèle quantifié sur le Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="intelliwork/LexiorGpt1-merged-AWQ",
        help="ID du dépôt cible sur Hugging Face (utilisé si --push_to_hub est activé)."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Token d'écriture Hugging Face pour l'upload."
    )
    
    return parser.parse_args()


def load_calibration_data(dataset_path, tokenizer, num_samples):
    """
    Charge les exemples de calibration juridiques et applique le template de chat Qwen-2.5.
    """
    if not os.path.exists(dataset_path):
        print(f"Attention : Le jeu de données de calibration n'a pas été trouvé à '{dataset_path}'.")
        print("Recherche dans d'autres répertoires courants...")
        # Tentative de recherche dans les chemins typiques RunPod
        runpod_path = "/runpod-volume/DistillationModeles/data/processed/generated_a2aj_cot_reconstructed.jsonl"
        workspace_path = "/workspace/DistillationModeles/data/processed/generated_a2aj_cot_reconstructed.jsonl"
        
        if os.path.exists(runpod_path):
            dataset_path = runpod_path
        elif os.path.exists(workspace_path):
            dataset_path = workspace_path
        else:
            print("Erreur : Impossible de localiser le jeu de données juridique.")
            print("Veuillez spécifier le bon chemin via --dataset_path.")
            sys.exit(1)
            
    print(f"Chargement de {num_samples} exemples de calibration depuis {dataset_path}...")
    calibration_texts = []
    
    # Assigner un template de chat par défaut (ChatML de Qwen) si absent
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\\n'}}"
            "{% endif %}"
        )
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(calibration_texts) >= num_samples:
                break
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                if not messages:
                    continue
                # Appliquer le template de chat officiel (conserve les balises de structure)
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
                calibration_texts.append(formatted_text)
            except Exception as e:
                print(f"Ligne {i} ignorée en raison d'une erreur de lecture : {e}")
                
    print(f"Calibration préparée avec succès ! ({len(calibration_texts)} dialogues chargés)")
    return calibration_texts


def main():
    args = parse_args()
    
    # Configuration du cache Hugging Face
    volume_path = "/runpod-volume" if os.path.exists("/runpod-volume") else "/workspace"
    os.environ["HF_HOME"] = os.path.join(volume_path, "hf_cache")
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    
    # Désactiver hf_xet pour éviter le bug de téléchargement "Background writer channel closed"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    print("==================================================")
    # Titre en français et esthétique propre
    print("     QUANTIFICATION HORS LIGNE LEXIORGPT AWQ")
    print("==================================================")
    print(f"Modèle source   : {args.model_path}")
    print(f"Modèle cible    : {args.quant_path}")
    print(f"Calibration     : {args.dataset_path}")
    print(f"Échantillons    : {args.num_samples}")
    print("==================================================")
    
    # 1. Chargement du Tokenizer
    print("\n1/4. Chargement du Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 2. Préparation du jeu de données de calibration
    print("\n2/4. Préparation du jeu de données de calibration...")
    calibration_data = load_calibration_data(args.dataset_path, tokenizer, args.num_samples)
    
    # 3. Chargement du modèle de base
    print("\n3/4. Chargement du modèle en précision FP16...")
    start_time = time.time()
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    print(f"Modèle chargé en {time.time() - start_time:.2f} secondes.")
    
    # 4. Quantification AWQ
    print("\n4/4. Lancement de la quantification AWQ (4-bit, groupe 128)...")
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
    
    start_quant = time.time()
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_data
    )
    print(f"Quantification terminée en {time.time() - start_quant:.2f} secondes.")
    
    # 5. Sauvegarde
    print(f"\nSauvegarde locale du modèle quantifié dans : {args.quant_path}...")
    model.save_quantized(args.quant_path)
    tokenizer.save_pretrained(args.quant_path)
    print("Sauvegarde locale terminée !")
    
    # 6. Téléversement optionnel sur Hugging Face
    if args.push_to_hub:
        if not args.hf_token:
            print("Erreur : Le token Hugging Face (--hf_token) est requis pour l'upload.")
            sys.exit(1)
            
        print(f"\nTéléversement du modèle quantifié sur le Hub Hugging Face : {args.repo_id}...")
        model.push_to_hub(args.repo_id, token=args.hf_token)
        tokenizer.push_to_hub(args.repo_id, token=args.hf_token)
        print("Téléversement terminé avec succès !")
        
    print("\n==================================================")
    print("       PROCESSUS COMPLET TERMINÉ AVEC SUCCÈS")
    print("==================================================")


if __name__ == "__main__":
    main()
