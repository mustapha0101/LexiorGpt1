#!/bin/bash

# Pipeline de Distillation CoT automatique pour RunPod
# Ce script prépare l'environnement, formate le dataset et lance l'entraînement avec Unsloth.

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}       LEXIOR - Pipeline de Distillation CoT avec Unsloth (RunPod)    ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# 1. Vérification de l'environnement GPU
echo -e "\n${YELLOW}[1/5] Vérification de l'environnement GPU (CUDA)...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Erreur : Aucun GPU NVIDIA détecté ou nvidia-smi n'est pas disponible.${NC}"
    echo -e "${YELLOW}Ce script est conçu pour s'exécuter sur une instance RunPod GPU.${NC}"
    exit 1
fi
nvidia-smi

# 2. Installation des dépendances standard
echo -e "\n${YELLOW}[2/5] Installation des dépendances (requirements.txt)...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# 3. Installation spécifique d'Unsloth (compatible CUDA / Triton)
echo -e "\n${YELLOW}[3/5] Installation d'Unsloth optimisé pour GPU...${NC}"
# Unsloth recommande cette commande d'installation universelle pour les environnements CUDA récents
pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothdev/unsloth.git"

# 4. Connexion au Hugging Face Hub si un token est fourni
echo -e "\n${YELLOW}[4/5] Configuration de Hugging Face...${NC}"
if [ -n "$HF_TOKEN" ]; then
    echo "Connexion à Hugging Face à l'aide du token fourni dans \$HF_TOKEN..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo -e "${YELLOW}Aucun token \$HF_TOKEN détecté en variable d'environnement.${NC}"
    echo -e "Si vous devez pousser des modèles vers le Hub, lancez la commande suivante au préalable :"
    echo -e "  export HF_TOKEN=votre_token_hugging_face"
fi

# 5. Formatage des données (SuperMust/irac-thinking par défaut)
echo -e "\n${YELLOW}[5/5] Formatage du dataset pour la distillation CoT (IRAC)...${NC}"
DATASET=${DATASET_NAME:-"SuperMust/irac-thinking"}
MODEL_BASE=${MODEL_NAME:-"unsloth/llama-3-8b-Instruct-bnb-4bit"}

python data/dataset_formatter.py \
    --dataset_name "$DATASET" \
    --model_name "$MODEL_BASE" \
    --output_dir "data/processed" \
    --test_size 0.05

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur lors du formatage des données. Arrêt du pipeline.${NC}"
    exit 1
fi

# 6. Lancement du Fine-Tuning
echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}           LANCEMENT DE L'ENTRAÎNEMENT AVEC UNSLOTH                   ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# Variables par défaut (surchargées par les variables d'environnement)
EPOCHS=${TRAIN_EPOCHS:-3}
BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
GRAD_ACCUM=${TRAIN_GRAD_ACCUM:-4}
LR=${TRAIN_LR:-2e-4}
LORA_R=${TRAIN_LORA_R:-16}
LORA_ALPHA=${TRAIN_LORA_ALPHA:-32}

# Options de téléversement et d'export GGUF
PUSH_FLAG=""
REPO_FLAG=""
if [ -n "$HF_REPO_ID" ]; then
    PUSH_FLAG="--push_to_hub"
    REPO_FLAG="--hf_repo_id $HF_REPO_ID"
    echo -e "${YELLOW}Le modèle et les adapters seront téléversés vers : $HF_REPO_ID${NC}"
fi

GGUF_FLAG="--export_gguf"
MERGED_FLAG="--export_merged_16bit"

# Options de tracking (Weights & Biases par défaut si token présent)
REPORT_TO=${TRACKING_REPORT_TO:-"none"}
RUN_NAME=${TRACKING_RUN_NAME:-"llama3-juridique-cot"}

if [ -n "$WANDB_API_KEY" ]; then
    REPORT_TO="wandb"
    echo -e "${YELLOW}Jeton Weights & Biases détecté. Le tracking est activé.${NC}"
fi

python src/training/train_unsloth.py \
    --model_name "$MODEL_BASE" \
    --train_file "data/processed/train_dataset.jsonl" \
    --test_file "data/processed/test_dataset.jsonl" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" \
    $PUSH_FLAG \
    $REPO_FLAG \
    $GGUF_FLAG \
    $MERGED_FLAG

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}======================================================================${NC}"
    echo -e "${GREEN}       PIPELINE EXÉCUTÉ AVEC SUCCÈS ! LES MODÈLES SONT PRÊTS.         ${NC}"
    echo -e "${GREEN}======================================================================${NC}"
else
    echo -e "\n${RED}Erreur pendant le fine-tuning. Veuillez consulter les logs ci-dessus.${NC}"
    exit 1
fi
