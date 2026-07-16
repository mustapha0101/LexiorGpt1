#!/bin/bash

# Phase 3 : Script d'exécution maître pour l'évaluation et la visualisation
# Ce script lance l'évaluation locale et ouvre le tableau de bord dans le navigateur.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}          Phase 3 : Évaluation & Tableau de Bord LexiorGPT           ${NC}"
echo -e "${GREEN}======================================================================${NC}"

# 1. Vérification des dépendances
echo -e "\n${YELLOW}[1/3] Vérification des prérequis Python...${NC}"
pip install -q datasets openai tqdm jinja2

# 2. Exécution du moteur d'évaluation
echo -e "\n${YELLOW}[2/3] Analyse et calcul des scores de fiabilité...${NC}"
python3 eval_engine.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Erreur pendant l'exécution du moteur d'évaluation.${NC}"
    exit 1
fi

# 3. Lancement du tableau de bord dans le navigateur
echo -e "\n${YELLOW}[3/3] Lancement du tableau de bord dans votre navigateur...${NC}"

# Vérifier quel OS est utilisé pour ouvrir l'URL
if [[ "$OSTYPE" == "darwin"* ]]; then
    open dashboard/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open > /dev/null; then
        xdg-open dashboard/index.html
    else
        echo -e "${YELLOW}Navigateur non détecté. Ouvrez manuellement le fichier :${NC}"
        echo -e "${GREEN}Phase3_Evaluation/dashboard/index.html${NC}"
    fi
else
    echo -e "${YELLOW}Système d'exploitation non pris en charge pour l'ouverture automatique.${NC}"
    echo -e "Veuillez ouvrir manuellement le fichier dans votre navigateur :"
    echo -e "${GREEN}Phase3_Evaluation/dashboard/index.html${NC}"
fi

echo -e "\n${GREEN}======================================================================${NC}"
echo -e "${GREEN}            Évaluation terminée ! Dashboard ouvert avec succès.       ${NC}"
echo -e "${GREEN}======================================================================${NC}"
