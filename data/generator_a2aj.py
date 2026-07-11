#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération synthétique CoT (Chain-of-Thought) Juridique.
Il extrait des textes depuis les corpus bruts de l'A2AJ :
- a2aj/canadian-laws
- a2aj/canadian-case-law

Puis il interroge un modèle de langage Teacher (via API OpenAI/Groq/Ollama)
pour générer des paires d'entraînement (Prompt + Réflexion CoT + Réponse/Outil) en français québécois/canadien.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="Générateur de données synthétiques CoT A2AJ.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="a2aj/canadian-laws",
        choices=["a2aj/canadian-laws", "a2aj/canadian-case-law"],
        help="Nom du dataset brut A2AJ à charger depuis Hugging Face."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split du dataset à charger."
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="URL de l'API Teacher (Ollama, Groq, Together, OpenAI)."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="Clé API du modèle Teacher."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Nom du modèle Teacher (ex. gpt-4o, llama3-70b-instruct, qwen-2.5-72b)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Nombre de textes juridiques bruts à traiter pour la génération."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/generated_a2aj_cot.jsonl",
        help="Fichier de sortie pour stocker le dataset généré."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API (--api_key ou variable d'environnement) est requise.")
        sys.exit(1)
        
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    
    # 1. Chargement du dataset brut A2AJ
    print(f"Chargement du dataset de base A2AJ : {args.dataset} (split: {args.split})...")
    try:
        raw_ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Erreur lors du chargement du dataset HF : {e}")
        sys.exit(1)
        
    print(f"Dataset chargé ! Taille totale : {len(raw_ds)} exemples.")
    
    # 2. Préparation du répertoire de sortie
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 3. Boucle de génération synthétique
    success_count = 0
    
    # Récupération des n premiers éléments
    sample_range = min(args.limit, len(raw_ds))
    print(f"Lancement de la génération CoT sur les {sample_range} premiers textes juridiques...")
    
    with open(args.output_file, "a", encoding="utf-8") as f_out:
        for idx in tqdm(range(sample_range)):
            item = raw_ds[idx]
            
            # Extraction du contenu du texte brut selon le dataset
            raw_text = ""
            context_meta = {}
            
            if args.dataset == "a2aj/canadian-laws":
                # Le dataset canadian-laws contient généralement les colonnes 'text', 'title', 'section'
                raw_text = item.get("text", "")
                context_meta = {
                    "title": item.get("title", "Loi canadienne"),
                    "section": item.get("section", "N/A")
                }
            elif args.dataset == "a2aj/canadian-case-law":
                # Le dataset canadian-case-law contient généralement le texte de la décision ('text' ou 'content')
                raw_text = item.get("text", item.get("content", ""))
                context_meta = {
                    "citation": item.get("citation", "Jurisprudence"),
                    "court": item.get("court", "Tribunal canadien")
                }
                
            if not raw_text or len(raw_text.strip()) < 150:
                continue # Passer les textes trop courts
                
            # Limiter la taille du texte brut envoyé pour éviter de saturer le contexte
            truncated_text = raw_text[:4000]
            
            # 4. Prompt système et utilisateur pour forcer la CoT IRAC et le format Lexior
            system_prompt = (
                "Tu es un expert en droit canadien et québécois. Ta tâche est d'analyser le texte juridique fourni "
                "et de générer une question juridique complexe en français, puis de fournir une réponse structurée.\n\n"
                "Tu dois obligatoirement formater ta réponse en générant :\n"
                "1. Un bloc de réflexion <thinking>...</thinking> contenant le raisonnement juridique IRAC en français :\n"
                "   - Issue (Question de droit)\n"
                "   - Rule (Règle de droit et articles précis du Code civil du Québec ou lois fédérales cités)\n"
                "   - Application (Raisonnement ou application des faits)\n"
                "2. La réponse finale (Conclusion) en français.\n"
                "3. Une citation de bas de page contenant l'URL au format JSON LexiorGPT à la toute fin :\n"
                "   [^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/...\",\"title\":\"Titre du document\"}"
            )
            
            user_prompt = (
                f"Voici l'extrait juridique de référence (Métadonnées : {json.dumps(context_meta, ensure_ascii=False)}) :\n\n"
                f"<context>\n{truncated_text}\n</context>\n\n"
                f"Génère une interaction utilisateur/assistant réaliste pour LexiorNotebook. "
                f"L'utilisateur pose une question complexe en français relative à ce texte juridique, et tu y réponds selon les consignes strictes."
            )
            
            # Appel au Teacher LLM
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                
                generated_output = response.choices[0].message.content
                
                # Validation minimale de la présence du CoT
                if "<thinking>" in generated_output and "</thinking>" in generated_output:
                    # Extraction du prompt généré (la question simulée de l'utilisateur)
                    # Pour simplifier, nous divisons la réponse du modèle :
                    # On demande au modèle de générer une question et une réponse.
                    # Si le modèle renvoie directement la réponse assistant, on simule une question utilisateur correspondante.
                    
                    # Nettoyage et séparation de la réflexion et de la réponse
                    parts = generated_output.split("</thinking>")
                    thinking_part = parts[0].replace("<thinking>", "").strip()
                    content_part = parts[1].strip()
                    
                    # On crée une question réaliste basée sur l'Issue identifiée dans la CoT
                    issue_match = re.search(r"Issue\s*:\s*(.*)", thinking_part, re.IGNORECASE)
                    simulated_question = "Analyse cette situation : " + context_meta.get("title", "Droit canadien")
                    if issue_match:
                        simulated_question = issue_match.group(1).strip()
                        
                    # Structure de message finale
                    message_data = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": simulated_question},
                            {"role": "assistant", "content": content_part, "thinking": thinking_part}
                        ]
                    }
                    
                    f_out.write(json.dumps(message_data, ensure_ascii=False) + "\n")
                    success_count += 1
            except Exception as e:
                print(f"Erreur lors de l'appel API pour l'index {idx} : {e}")
                continue
                
    print(f"Génération terminée ! {success_count} exemples valides enregistrés dans '{args.output_file}'.")

if __name__ == "__main__":
    main()
