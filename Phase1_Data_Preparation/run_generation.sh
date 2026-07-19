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

# Optionnel : Démarrer et configurer Ollama localement si demandé
if [ "$USE_LOCAL_OLLAMA" = "true" ]; then
    if command -v ollama >/dev/null 2>&1; then
        echo -e "\n${YELLOW}[1b/3] Ollama est déjà installé. Vérification du serveur...${NC}"
        # Démarrer le serveur uniquement s'il ne tourne pas déjà
        if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo -e "${YELLOW}Démarrage du serveur Ollama...${NC}"
            ollama serve > ollama.log 2>&1 &
            sleep 10
        fi
    else
        echo -e "\n${YELLOW}[1b/3] Installation et démarrage de Ollama (modèle local)...${NC}"
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update && apt-get install -y zstd pciutils lshw
            curl -fsSL https://ollama.com/install.sh | sh
            ollama serve > ollama.log 2>&1 &
            sleep 10
        else
            echo -e "${RED}Erreur : apt-get introuvable et Ollama n'est pas installé. Veuillez l'installer manuellement.${NC}"
            exit 1
        fi
    fi
    echo -e "${YELLOW}Vérification du modèle Teacher local (${GEN_MODEL:-qwen2.5:32b})...${NC}"
    ollama pull ${GEN_MODEL:-qwen2.5:32b-instruct-q4_K_M}
fi

# 2. Exécution du générateur parallèle (A2AJ)
echo -e "\n${YELLOW}[2/3] Génération de données synthétiques à partir de A2AJ...${NC}"
# Récupération des variables d'environnement
DATASET=${DATASET_NAME:-"a2aj/canadian-laws"}
LIMIT=${GEN_LIMIT:-1000}
WORKERS=${GEN_WORKERS:-8}
# Définir le modèle par défaut pour Ollama si non spécifié
if [ "$USE_LOCAL_OLLAMA" = "true" ]; then
    MODEL=${GEN_MODEL:-"qwen2.5:32b-instruct-q4_K_M"}
    export OPENAI_API_KEY="ollama"
    export OPENAI_BASE_URL="http://localhost:11434/v1"
else
    MODEL=${GEN_MODEL:-"gpt-4o-mini"}
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Erreur : La variable \$OPENAI_API_KEY n'est pas définie.${NC}"
    echo -e "Veuillez la définir dans setup_env.sh ou dans votre terminal."
    exit 1
fi

python3 utils/resume_from_hf.py \
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
python3 provincial_quebec/generate_ccq_data.py \
    --model "$MODEL" \
    --scenarios_per_article 10 \
    --output_file "data/processed/generated_ccq_cot.jsonl"

echo -e "\n${YELLOW}[2c/3] Exécution du scraper LégisQuébec et Fédéral pour extraire les articles à jour...${NC}"
python3 provincial_quebec/legisquebec_scraper.py

echo -e "\n${YELLOW}[2d/3] Génération du jeu d'alignement d'identité (LexiorGPT Branding & DPO)...${NC}"
python3 identity/generate_identity_data.py

echo -e "\n${YELLOW}[2e/3] Génération du jeu d'appels d'outils (Tool Calling / MCP)...${NC}"
python3 tool_calling/generate_tool_calling_data.py

echo -e "\n${YELLOW}[2f/3] Génération du jeu de citations exactes du Code civil du Québec (CCQ)...${NC}"
python3 provincial_quebec/generate_ccq_citations_dataset.py

echo -e "\n${YELLOW}[2f-2/3] Génération du jeu Détective Sherlock Holmes (Multi-tours Graphe)...${NC}"
python3 identity/generate_sherlock_data.py

# Concaténer les fichiers de génération
echo -e "\n${YELLOW}[2g/3] Fusion des datasets (A2AJ + CCQ + Identité SFT + Outils + Citations CCQ + Sherlock)...${NC}"
cat data/processed/generated_a2aj_cot.jsonl data/processed/generated_ccq_cot.jsonl data/processed/generated_identity_cot.jsonl data/processed/generated_tool_calling_cot.jsonl data/processed/ccq_citations_sft.jsonl data/processed/generated_sherlock_cot.jsonl > data/processed/combined_raw_cot.jsonl

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
        python3 utils/push_to_hf.py
    fi
else
    echo -e "${RED}Erreur pendant le formatage des données.${NC}"
    exit 1
fi
