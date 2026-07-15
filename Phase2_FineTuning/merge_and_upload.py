# -*- coding: utf-8 -*-
import os
# Configurer le cache Hugging Face sur le volume persistant disponible
volume_path = "/runpod-volume" if os.path.exists("/runpod-volume") else "/workspace"
os.environ["HF_HOME"] = os.path.join(volume_path, "hf_cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    # Définition des chemins et dépôts
    lora_dir = "outputs/final_model/lora_adapters"
    merged_dir = "outputs/final_model/merged_16bit"
    
    hf_token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID", "intelliwork/LexiorGpt1")
    merged_repo_id = f"{repo_id}-merged"
    
    print("==================================================")
    print("   FUSION DU MODÈLE ET EXPORT HUGGING FACE")
    print("==================================================")
    
    # 1. Chargement du modèle de base Qwen-32B
    print("1/4. Chargement du modèle de base Qwen-32B (float16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-32B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    
    # 2. Application et fusion des adapters LoRA
    print("2/4. Chargement et fusion des adapters LoRA...")
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"Dossier des adapters introuvable à : {lora_dir}")
        
    model_to_merge = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = model_to_merge.merge_and_unload()
    
    # 3. Sauvegarde locale du modèle complet
    print(f"3/4. Sauvegarde locale du modèle complet dans : {merged_dir}...")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    # 4. Téléversement sur le Hub Hugging Face
    api = HfApi(token=hf_token)
    
    # A. Téléversement des adapters LoRA finaux (Étape 2703)
    print(f"4a/4. Téléversement des adapters LoRA finaux sur : {repo_id}...")
    api.upload_folder(
        folder_path=lora_dir,
        repo_id=repo_id,
        commit_message="Ajout des adapters LoRA finalisés (Étape 2703)"
    )
    print("Adapters LoRA finaux envoyés avec succès !")
    
    # B. Téléversement du modèle complet fusionné
    print(f"4b/4. Téléversement du modèle fusionné sur : {merged_repo_id}...")
    merged_model.push_to_hub(merged_repo_id, token=hf_token)
    tokenizer.push_to_hub(merged_repo_id, token=hf_token)

    
    print("==================================================")
    print(" FUSION ET TÉLÉVERSEMENT TERMINÉS AVEC SUCCÈS !")
    print("==================================================")

if __name__ == "__main__":
    main()
