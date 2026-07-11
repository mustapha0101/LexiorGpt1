#!/bin/bash

# Phase 1 : Script de génération et formatage du dataset CoT Juridique
# Ce script s'exécute sur une machine CPU/GPU légère pour créer le dataset avant l'entraînement.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}    Phase 1 : Génération et Formatage du Dataset Canadien CoT         ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# 1. Installation des dépendances de préparation
echo -e "\n${YELLOW}[1/3] Installation des packages de base (datasets, openai)...${NC}"
pip install --upgrade pip
pip install datasets openai tqdm transformers jinja2

# 2. Exécution du générateur parallèle (A2AJ)
echo -e "\n${YELLOW}[2/3] Génération de données synthétiques à partir de A2AJ...${NC}"
# Récupération des variables d'environnement
DATASET=${DATASET_NAME:-"a2aj/canadian-laws"}
LIMIT=${GEN_LIMIT:-1000}
WORKERS=${GEN_WORKERS:-15}
MODEL=${GEN_MODEL:-"gpt-4o-mini"}

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Erreur : La variable \$OPENAI_API_KEY n'est pas définie.${NC}"
    echo -e "Veuillez la définir dans setup_env.sh ou dans votre terminal."
    exit 1
fi

python3 generator_a2aj.py \
    --dataset "$DATASET" \
    --limit "$LIMIT" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    --output_file "data/processed/generated_a2aj_cot.jsonl"

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur pendant la génération A2AJ. Arrêt du script.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[2b/3] Génération de cas pratiques basés sur le Code civil du Québec (CCQ)...${NC}"
python3 generate_ccq_data.py \
    --model "$MODEL" \
    --scenarios_per_article 10 \
    --output_file "data/processed/generated_ccq_cot.jsonl"

# Concaténer les deux fichiers de génération
echo -e "\n${YELLOW}[2c/3] Fusion des datasets (A2AJ + CCQ)...${NC}"
cat data/processed/generated_a2aj_cot.jsonl data/processed/generated_ccq_cot.jsonl > data/processed/combined_raw_cot.jsonl

# 3. Exécution du formatage de chat
echo -e "\n${YELLOW}[3/3] Application du template conversationnel sur le corpus combiné...${NC}"
python3 dataset_formatter.py \
    --local_file "data/processed/combined_raw_cot.jsonl" \
    --output_dir "data/processed" \
    --test_size 0.05

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}======================================================================${NC}"
    echo -e "${GREEN}  PHASE 1 TERMINÉE ! Le dataset combiné est dans data/processed/    ${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    
    # Étape facultative : Pousser sur Hugging Face Hub si la variable est configurée
    if [ -n "$HF_DATASET_REPO_ID" ]; then
        echo -e "\n${YELLOW}[Facultatif] Téléversement du dataset sur Hugging Face Hub...${NC}"
        python3 push_to_hf.py
    fi
else
    echo -e "${RED}Erreur pendant le formatage des données.${NC}"
    exit 1
fi
