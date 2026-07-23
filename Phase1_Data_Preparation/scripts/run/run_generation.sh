#!/bin/bash

# Le comportement historique reste le défaut tant que le pipeline agentique
# n'est pas explicitement sélectionné.
GENERATION_MODE=${GENERATION_MODE:-legacy}
if [ "$GENERATION_MODE" = "agentic" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    exec "$SCRIPT_DIR/run_agentic_generation.sh" "$@"
elif [ "$GENERATION_MODE" != "legacy" ]; then
    echo "GENERATION_MODE invalide: $GENERATION_MODE (legacy|agentic)" >&2
    exit 2
fi
echo "[legacy] Pipeline one-shot conservé pour reproductibilité; utilisez GENERATION_MODE=agentic pour le nouveau pipeline."

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
    echo -e "\n${YELLOW}[1b/3] Installation et démarrage de Ollama (modèle local)...${NC}"
    apt-get update && apt-get install -y zstd pciutils lshw
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve > ollama.log 2>&1 &
    sleep 10
    echo -e "${YELLOW}Téléchargement du modèle Teacher local (${GEN_MODEL:-qwen2.5:32b})...${NC}"
    ollama pull ${GEN_MODEL:-qwen2.5:32b-instruct-q4_K_M}
fi

# 2. Exécution du générateur parallèle (A2AJ)
echo -e "\n${YELLOW}[2/3] Génération de données synthétiques à partir de A2AJ...${NC}"
# Récupération des variables d'environnement
DATASET=${DATASET_NAME:-"a2aj/canadian-laws"}
LIMIT=${GEN_LIMIT:-1000}
WORKERS=${GEN_WORKERS:-8}
# Nombre de LIGNES visé par juridiction, lignes déjà présentes comprises.
# Premier passage : 1000 de chaque côté, à valider. Pour continuer ensuite,
# relancer avec un plafond plus élevé (MAX_ROWS_FED=5000) ou sans plafond (-1) :
# les index de reprise sautent ce qui est déjà fait, sans repartir de zéro.
MAX_ROWS_FED=${MAX_ROWS_FED:-1000}
MAX_ROWS_QC=${MAX_ROWS_QC:-1000}
# Premier test : uniquement les lois fédérales tenant entièrement dans le budget
# de contexte (2 413 sur 4 727). Les longues lois attendent un découpage par
# article — tronquées, il n'en resterait que les définitions d'ouverture.
# Mettre WHOLE_LAWS_ONLY=false pour les réintégrer (tronquées).
WHOLE_LAWS_FLAG=""
if [ "${WHOLE_LAWS_ONLY:-true}" = "true" ]; then
    WHOLE_LAWS_FLAG="--whole_laws_only"
fi

# --- Jeu d'identité -------------------------------------------------------
# IDENTITY_POOL_SIZE : taille du vivier produit (conversations uniques).
# IDENTITY_COUNT     : nombre ABSOLU de lignes d'identité dans le mélange final.
#                      Le vivier est fixe : la proportion baisse d'elle-même à
#                      mesure que le corpus juridique grandit (500 sur 2 000
#                      lignes = 20 % des lignes mais ~5 % des tokens ; les mêmes
#                      500 sur 20 000 = 2,4 %).
# IDENTITY_RATIO     : proportion CIBLE (0.02 / 0.05 / 0.08). Si elle est
#                      définie, elle prend le pas sur IDENTITY_COUNT — c'est le
#                      mode des trois expériences comparatives.
IDENTITY_POOL_SIZE=${IDENTITY_POOL_SIZE:-1000}
IDENTITY_COUNT=${IDENTITY_COUNT:-500}
IDENTITY_RATIO=${IDENTITY_RATIO:-}
IDENTITY_SEED=${IDENTITY_SEED:-3407}
LEGAL_SYSTEM_PROMPT_DROPOUT=${LEGAL_SYSTEM_PROMPT_DROPOUT:-0.15}
TEST_SIZE=${TEST_SIZE:-0.05}

# Le mélangeur accepte l'un OU l'autre : compte absolu, ou proportion cible.
if [ -n "$IDENTITY_RATIO" ]; then
    MIX_IDENTITY_FLAG="--identity_ratio $IDENTITY_RATIO"
    AUDIT_EXPECT_FLAG="--expect_identity_ratio $IDENTITY_RATIO"
else
    MIX_IDENTITY_FLAG="--identity_count $IDENTITY_COUNT"
    AUDIT_EXPECT_FLAG="--expect_identity_count $IDENTITY_COUNT"
fi
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
    echo -e "Veuillez la définir dans votre environnement ou dans un fichier .env chargé explicitement."
    exit 1
fi

# Resolve project root (two levels up from scripts/run/).
PHASE1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 "$PHASE1_DIR/scripts/huggingface/resume_generation.py" \
    --dataset "$DATASET" \
    --limit "$LIMIT" \
    --max_rows "$MAX_ROWS_FED" \
    --workers "$WORKERS" \
    --model "$MODEL" \
    $WHOLE_LAWS_FLAG \
    --output_file "data/processed/generated_a2aj_cot.jsonl"

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur pendant la génération A2AJ. Arrêt du script.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[2b/3] Génération de cas pratiques basés sur le droit québécois (CCQ + CPC)...${NC}"
python3 "$PHASE1_DIR/scripts/dataset_generation/generate_ccq.py" \
    --model "$MODEL" \
    --scenarios_per_article 10 \
    --max_rows "$MAX_ROWS_QC" \
    --output_file "data/processed/generated_ccq_cot.jsonl"

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur pendant la génération québécoise. Arrêt du script.${NC}"
    exit 1
fi

echo -e "\n${YELLOW}[3/6] Génération du vivier d'identité (aucun appel API)...${NC}"
python3 "$PHASE1_DIR/scripts/dataset_generation/generate_identity.py" \
    --count "$IDENTITY_POOL_SIZE" \
    --seed "$IDENTITY_SEED" \
    --output_file "data/processed/generated_identity_cot.jsonl"

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur pendant la génération du jeu d'identité "
    echo -e "(violation probable de la politique — cf. identity_policy.py).${NC}"
    exit 1
fi

# Téléversement de chaque corpus dans SON dépôt privé, avant la fusion.
# C'est cette version brute qui sert de point de reprise et de validation.
if [ -n "$HF_TOKEN" ]; then
    echo -e "\n${YELLOW}[4/6] Téléversement des trois corpus sources (dépôts privés)...${NC}"
    python3 "$PHASE1_DIR/scripts/huggingface/push_dataset.py" \
        --input_file "data/processed/generated_a2aj_cot.jsonl" \
        --repo_id "${HF_REPO_FEDERAL:-intelliwork/canadian-cot-dataset-federal-french}"
    python3 "$PHASE1_DIR/scripts/huggingface/push_dataset.py" \
        --input_file "data/processed/generated_ccq_cot.jsonl" \
        --repo_id "${HF_REPO_QUEBEC:-intelliwork/canadian-cot-dataset-quebec-french}"
    python3 "$PHASE1_DIR/scripts/huggingface/push_dataset.py" \
        --input_file "data/processed/generated_identity_cot.jsonl" \
        --repo_id "${HF_REPO_IDENTITY:-intelliwork/canadian-cot-dataset-identity-french}"
else
    echo -e "\n${YELLOW}[4/6] HF_TOKEN absent — téléversement des corpus sources ignoré.${NC}"
fi

# --- Mélange -------------------------------------------------------------
# Remplace l'ancien « cat ». Un cat rendait la proportion d'identité
# accidentelle : elle valait ce que les fichiers contenaient. Ici elle est
# demandée, vérifiée, et consignée dans un manifeste.
echo -e "\n${YELLOW}[5/6] Mélange des trois sources (mix_datasets.py)...${NC}"
python3 "$PHASE1_DIR/scripts/dataset_processing/mix_datasets.py" \
    --federal_file "data/processed/generated_a2aj_cot.jsonl" \
    --quebec_file "data/processed/generated_ccq_cot.jsonl" \
    --identity_file "data/processed/generated_identity_cot.jsonl" \
    --output_file "data/processed/combined_raw_cot.jsonl" \
    --manifest_file "data/processed/mix_manifest.json" \
    --no-include_agentic \
    --include_legacy_legal \
    --include_identity_data \
    --seed "$IDENTITY_SEED" \
    $MIX_IDENTITY_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur pendant le mélange des datasets. Arrêt du script.${NC}"
    exit 1
fi

# --- Formatage -----------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Formatage conversationnel du corpus mélangé...${NC}"
python3 "$PHASE1_DIR/scripts/dataset_processing/format_dataset.py" \
    --local_file "data/processed/combined_raw_cot.jsonl" \
    --output_dir "data/processed" \
    --test_size "$TEST_SIZE" \
    --legal_system_prompt_dropout "$LEGAL_SYSTEM_PROMPT_DROPOUT" \
    --seed "$IDENTITY_SEED"

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur pendant le formatage des données.${NC}"
    exit 1
fi

# --- Audit : barrière avant tout téléversement ----------------------------
# Ne jamais affirmer que le jeu d'identité est présent : le vérifier.
# Sortie non nulle => on n'envoie rien.
echo -e "\n${YELLOW}[Audit] Vérification du dataset d'entraînement...${NC}"
python3 "$PHASE1_DIR/scripts/dataset_processing/audit_dataset.py" \
    --files "data/processed/train_dataset.jsonl" "data/processed/test_dataset.jsonl" \
    --require_identity_in_test \
    --report_file "data/processed/audit_report.json" \
    $AUDIT_EXPECT_FLAG

if [ $? -ne 0 ]; then
    echo -e "\n${RED}======================================================================${NC}"
    echo -e "${RED}  AUDIT EN ÉCHEC — le dataset n'est PAS téléversé.                    ${NC}"
    echo -e "${RED}  Voir data/processed/audit_report.json                               ${NC}"
    echo -e "${RED}======================================================================${NC}"
    exit 1
fi

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}  PHASE 1 TERMINÉE — dataset vérifié dans data/processed/            ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# Téléversement du dataset combiné formaté (facultatif).
if [ -n "$HF_DATASET_REPO_ID" ] && [ -n "$HF_TOKEN" ]; then
    echo -e "\n${YELLOW}[Facultatif] Téléversement du dataset combiné formaté...${NC}"
    python3 "$PHASE1_DIR/scripts/huggingface/push_dataset.py" --repo_id "$HF_DATASET_REPO_ID"
fi
