#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération du dataset d'entraînement SFT pour la mémorisation et la citation exacte
de l'ensemble des articles du Code civil du Québec (CCQ).
Parcourt les 3 489 articles du CCQ et génère des paires de questions/réponses de citation directe.
"""

import os
import json
import random

def generate_ccq_citations():
    ccq_db_path = "data/ccqDb.json"
    output_file = "data/processed/ccq_citations_sft.jsonl"
    
    if not os.path.exists(ccq_db_path):
        print(f"❌ Erreur : Le fichier {ccq_db_path} est introuvable. Veuillez d'abord importer ccqDb.json.")
        return
        
    print(f"Lecture des articles depuis {ccq_db_path}...")
    with open(ccq_db_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    system_prompt = (
        "Tu es LexiorGPT, un assistant d'intelligence juridique spécialisé en droit canadien et québécois. "
        "Raisonne en français selon le format IRAC. Tu as été conçu et développé par l'équipe d'intelliwork."
    )
    
    # Formats de questions de citation
    prompt_templates = [
        "Quel est le texte exact de l'article {num} du Code civil du Québec ?",
        "Cite mot pour mot l'article {num} du CCQ.",
        "Peux-tu me donner le contenu officiel de l'article {num} du Code civil du Québec ?",
        "Comment est rédigé l'article {num} du CCQ ?",
        "Affiche le texte légal de l'article {num} du Code civil du Québec."
    ]
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(articles):
            num = item["numero"]
            text = item["texte"]
            
            if not num or not text:
                continue
                
            # Générer 2 formulations de questions différentes par article pour la robustesse
            chosen_templates = random.sample(prompt_templates, 2)
            
            for tpl in chosen_templates:
                user_query = tpl.format(num=num)
                
                # Appliquer 20% de System Prompt Dropout
                include_system = ((idx + count) % 5 != 0)
                
                messages = []
                if include_system:
                    messages.append({"role": "system", "content": system_prompt})
                    
                messages.append({"role": "user", "content": user_query})
                
                # Message de l'assistant structuré avec pensée et texte officiel exact
                assistant_message = {
                    "role": "assistant",
                    "content": f"Voici le texte officiel de l'**article {num} du Code civil du Québec** :\n\n> « {text} »",
                    "thinking": f"L'utilisateur me demande de citer l'article {num} du Code civil du Québec. Je dois fournir le texte exact sans modification pour garantir la rigueur juridique."
                }
                
                messages.append(assistant_message)
                
                # Écrire l'exemple d'entraînement au format JSONL
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
                
    print(f"🎉 Dataset de citation du Code civil généré avec succès ! {count} exemples créés dans {output_file}.")

if __name__ == "__main__":
    generate_ccq_citations()
