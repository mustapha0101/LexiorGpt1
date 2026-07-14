#!/bin/bash

# Phase 2 : Script de lancement du Fine-Tuning QLoRA
# S'exécute sur le Pod GPU de RunPod.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}    Phase 2 : Fine-Tuning QLoRA avec Unsloth (RunPod)                 ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# 1. Installation d'Unsloth et dépendances de base
echo -e "\n${YELLOW}[1/3] Installation des dépendances et d'Unsloth...${NC}"
export PATH=$PATH:/root/.local/bin:/usr/local/bin
pip install --upgrade pip

# Vérifier la compatibilité CUDA du pilote hôte
python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"
if [ $? -ne 0 ]; then
    echo -e "${RED}Le pilote NVIDIA du système hôte est trop ancien pour CUDA 12.1. Réinstallation de PyTorch avec CUDA 11.8...${NC}"
    pip uninstall -y torch torchvision torchaudio
    pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
fi

echo -e "\n${YELLOW}Installation de Unsloth et des dépendances optimisées...${NC}"
# Installer d'abord les packages système de base avec leurs dépendances (absents par défaut sur RunPod)
pip install datasets tqdm sentencepiece protobuf packaging ninja triton jinja2 pydantic "numpy<2.0" rich tensorboard wandb huggingface_hub
# Installer Unsloth (Colab tag)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Installer/Ecraser les versions requises sans les dépendances récursives pour éviter le backtracking
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes







# 2. Hugging Face Login (Programmation Python Robuste)
echo -e "\n${YELLOW}[2/3] Connexion au Hugging Face Hub...${NC}"
if [ -n "$HF_TOKEN" ]; then
    python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$HF_TOKEN')"
fi

# 3. Lancement du Fine-Tuning
echo -e "\n${YELLOW}[3/3] Lancement de l'entraînement...${NC}"
# Détection automatique du fichier de données généré dans la Phase 1
# On charge en priorité depuis le dépôt Hugging Face pour s'assurer d'avoir les 7500 exemples
DATASET_PATH=${TRAIN_DATASET_FILE:-"intelliwork/canadian-cot-dataset"}
MODEL_BASE=${MODEL_NAME:-"unsloth/Qwen2.5-32B-Instruct-bnb-4bit"}

# Options d'export
PUSH_FLAG=""
REPO_FLAG=""
if [ -n "$HF_REPO_ID" ]; then
    PUSH_FLAG="--push_to_hub"
    REPO_FLAG="--hf_repo_id $HF_REPO_ID"
fi

EPOCHS=${TRAIN_EPOCHS:-3}
BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
GRAD_ACCUM=${TRAIN_GRAD_ACCUM:-4}
LR=${TRAIN_LR:-2e-4}

python3 train_unsloth.py \
    --model_name "$MODEL_BASE" \
    --train_file "$DATASET_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --report_to "${TRACKING_REPORT_TO:-none}" \
    --run_name "${TRACKING_RUN_NAME:-qwen25-canadian-cot}" \
    $PUSH_FLAG \
    $REPO_FLAG \
    --export_gguf \
    --export_merged_16bit

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}======================================================================${NC}"
    echo -e "${GREEN}    PHASE 2 TERMINÉE ! LE MODÈLE EST ENTRAÎNÉ ET EXPORTÉ !            ${NC}"
    echo -e "${GREEN}======================================================================${NC}"
else
    echo -e "${RED}Erreur pendant l'entraînement.${NC}"
    exit 1
fi
