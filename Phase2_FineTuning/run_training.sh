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

# Configurer le répertoire de cache Hugging Face sur le volume workspace
VOLUME_PATH="/workspace"
if [ -d "/runpod-volume" ]; then
    VOLUME_PATH="/runpod-volume"
fi
mkdir -p "$VOLUME_PATH/hf_cache"
export HF_HOME="$VOLUME_PATH/hf_cache"


# 1. Installation des dépendances standard de Hugging Face

echo -e "\n${YELLOW}[1/3] Installation des dépendances standard de Hugging Face...${NC}"
export PATH=$PATH:/root/.local/bin:/usr/local/bin
pip install --upgrade pip

# Installer les versions stables et adaptées à PyTorch 2.2.0 de l'image de base RunPod
pip install --no-cache-dir "torch==2.2.0" "torchvision==0.17.0" "torchaudio==2.2.0"
pip install --no-cache-dir "huggingface_hub<0.25.0" "transformers<4.46.0" "peft<0.13.0" "trl<0.10.0" "accelerate<0.35.0" "bitsandbytes==0.43.3" datasets tqdm sentencepiece protobuf packaging ninja jinja2 pydantic "numpy<2.0" rich tensorboard wandb




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

# Téléchargement automatique de tout checkpoint sauvegardé sur Hugging Face
echo -e "\n${YELLOW}Recherche de checkpoints existants sur Hugging Face...${NC}"
python3 -c "
import os
import json
import shutil
from huggingface_hub import HfApi, snapshot_download
repo_id = os.environ.get('HF_REPO_ID')
token = os.environ.get('HF_TOKEN')
current_model = os.environ.get('MODEL_NAME', 'unsloth/Qwen2.5-32B-Instruct-bnb-4bit')
if repo_id and token:
    try:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id)
        checkpoints = set()
        for f in files:
            if '/' in f and f.startswith('checkpoint-'):
                checkpoints.add(f.split('/')[0])
        if checkpoints:
            print('Checkpoints trouvés sur Hugging Face :', list(checkpoints))
            os.makedirs('outputs/checkpoints', exist_ok=True)
            for cp in checkpoints:
                config_path = f'{cp}/adapter_config.json'
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=config_path,
                        local_dir='outputs/checkpoints',
                        token=token
                    )
                    local_config = os.path.join('outputs/checkpoints', config_path)
                    with open(local_config, 'r') as f:
                        config_data = json.load(f)
                    base_model_in_cp = config_data.get('base_model_name_or_path', '')
                    def get_model_id(name):
                        return name.split('/')[-1].replace('-bnb-4bit', '').replace('-Instruct', '').lower()
                    if get_model_id(base_model_in_cp) != get_model_id(current_model):
                        print(f'Ignoré {cp} : Incompatibilité du modèle de base ({base_model_in_cp} vs {current_model})')
                        shutil.rmtree(os.path.join('outputs/checkpoints', cp), ignore_errors=True)
                        continue
                except Exception as ex:
                    print(f'Erreur de vérification de config pour {cp} : {ex}')
                    continue

                print(f'Téléchargement de {cp}...')
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=f'{cp}/*',
                    local_dir='outputs/checkpoints',
                    token=token
                )
            print('Téléchargement des checkpoints validés terminé !')
        else:
            print('Aucun checkpoint trouvé sur le Hub.')
    except Exception as e:
        print('Erreur ou aucun checkpoint trouvé sur le Hub :', e)
"

python3 train_hf.py \
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
    --export_merged_16bit

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}======================================================================${NC}"
    echo -e "${GREEN}    PHASE 2 TERMINÉE ! LE MODÈLE EST ENTRAÎNÉ ET EXPORTÉ !            ${NC}"
    echo -e "${GREEN}======================================================================${NC}"
else
    echo -e "${RED}Erreur pendant l'entraînement.${NC}"
    exit 1
fi
