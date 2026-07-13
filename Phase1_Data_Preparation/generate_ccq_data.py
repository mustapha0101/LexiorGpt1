#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération de scénarios juridiques basés sur le Code civil du Québec (CCQ).
Il utilise des articles clés du CCQ pour générer des paires d'entraînement CoT (IRAC)
en français québécois, permettant au modèle d'apprendre la substance du Code civil.
"""

import os
import sys
import json
import re
import argparse
from tqdm import tqdm
from openai import OpenAI

# Articles majeurs du Code civil du Québec (CCQ) pour la recherche/distillation
CCQ_ARTICLES_SAMPLES = [
    {
        "article": "Article 1457",
        "texte": "Toute personne a le devoir de respecter les règles de conduite qui, suivant les circonstances, l'usage ou la loi, s'imposent à elle, de manière à ne pas causer de préjudice à autrui. Elle est, lorsqu'elle manque à ce devoir, responsable du préjudice causé par sa faute à autrui et tenue de réparer ce préjudice, qu'il soit corporel, moral ou matériel. Elle est aussi tenue, en certains cas, de réparer le préjudice causé à autrui par le fait ou la faute d'une autre personne ou par le fait des biens qu'elle a sous sa garde.",
        "domaine": "Responsabilité civile extracontractuelle"
    },
    {
        "article": "Article 1458",
        "texte": "Toute personne est tenue d'honorer les engagements qu'elle a contractés envers une autre. Elle est, lorsqu'elle manque à ce devoir, responsable du préjudice causé par son fait à son cocontractant et tenue de réparer ce préjudice; elle ne peut alors se soustraire à l'application des règles du régime contractuel pour opter en faveur de règles qui lui seraient plus favorables.",
        "domaine": "Responsabilité contractuelle"
    },
    {
        "article": "Article 1375",
        "texte": "La bonne foi doit gouverner la conduite des parties, tant au moment de la naissance de l'obligation qu'à celui de son exécution ou de son extinction.",
        "domaine": "Obligations et Contrats"
    },
    {
        "article": "Article 6",
        "texte": "Toute personne est tenue d'exercer ses droits civils selon les exigences de la bonne foi.",
        "domaine": "Principes directeurs du droit"
    },
    {
        "article": "Article 7",
        "texte": "Aucun droit ne peut être exercé dans l'intention de nuire à autrui ou d'une manière excessive et déraisonnable, allant ainsi à l'encontre des exigences de la bonne foi.",
        "domaine": "Abus de droit"
    },
    {
        "article": "Article 1590",
        "texte": "L'obligation inexécutée confère au créancier le droit de forcer l'exécution en nature de l'obligation, d'en demander la résolution ou la résiliation, ou la réduction de sa propre obligation corrélative; il peut aussi réclamer des dommages-intérêts au débiteur.",
        "domaine": "Résolution de contrat"
    },
    {
        "article": "Article 1434",
        "texte": "Le contrat valablement formé lie ceux qui l'ont conclu; il s'étend aussi à leurs successeurs à titre universel, à moins qu'il ne résulte de la nature du contrat, de la loi ou d'une stipulation qu'il en soit autrement.",
        "domaine": "Force obligatoire du contrat"
    }
]

def parse_args():
    parser = argparse.ArgumentParser(description="Générateur de scénarios CoT basés sur le CCQ.")
    parser.add_argument(
        "--api_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="URL de l'API Teacher."
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
        help="Nom du modèle Teacher."
    )
    parser.add_argument(
        "--scenarios_per_article",
        type=int,
        default=5,
        help="Nombre de situations/scénarios fictifs différents à générer par article."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/generated_ccq_cot.jsonl",
        help="Fichier de sortie."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.api_key:
        print("Erreur : La clé API est requise.")
        sys.exit(1)
        
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"Lancement de la génération de scénarios pour {len(CCQ_ARTICLES_SAMPLES)} articles clés du CCQ...")
    
    success_count = 0
    
    with open(args.output_file, "a", encoding="utf-8") as f_out:
        for item in tqdm(CCQ_ARTICLES_SAMPLES, desc="Articles CCQ"):
            art_name = item["article"]
            art_text = item["texte"]
            art_domain = item["domaine"]
            
            system_prompt = (
                "Tu es un professeur de droit civil du Québec. Ta tâche est de créer des exemples de cas pratiques "
                "et de résolutions juridiques rigoureuses pour l'entraînement d'une IA juridique.\n\n"
                "Tu dois obligatoirement formater ta réponse en générant :\n"
                "1. Un bloc de réflexion <thinking>...</thinking> contenant le raisonnement juridique IRAC en français québécois :\n"
                "   - Issue (La question de droit soulevée par la situation)\n"
                "   - Rule (La règle applicable, en citant et expliquant l'article du Code civil du Québec)\n"
                "   - Application (L'application concrète de la règle aux faits de la situation)\n"
                "2. La réponse finale (Conclusion) claire et synthétique en français québécois.\n"
                "3. Une citation de bas de page pointant vers le texte officiel du CCQ sur CanLII au format JSON LexiorGPT :\n"
                "   [^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/fr/qc/legis/lois/rlrq-c-ccq-1991/...\",\"title\":\"Code civil du Québec\"}"
            )
            
            for j in range(args.scenarios_per_article):
                user_prompt = (
                    f"Rédige une mise en situation fictive, réaliste et complexe se déroulant au Québec, "
                    f"mettant en jeu l'application directe de l'article suivant :\n\n"
                    f"[{art_name} - Domaine : {art_domain}]\n"
                    f"Texte officiel : \"{art_text}\"\n\n"
                    f"Simule ensuite la résolution de ce cas par un avocat québécois. "
                    f"La réponse doit inclure le bloc <thinking> (avec Issue, Rule, Application) et la conclusion finale "
                    f"avec la citation de bas de page conforme."
                )
                
                try:
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7 # Plus de créativité pour varier les scénarios
                    )
                    
                    generated_output = response.choices[0].message.content
                    
                    if "<thinking>" in generated_output and "</thinking>" in generated_output:
                        parts = generated_output.split("</thinking>")
                        thinking_part = parts[0].replace("<thinking>", "").strip()
                        content_part = parts[1].strip()
                        
                        # Simuler la question de l'utilisateur
                        simulated_question = f"J'ai un litige concernant : {art_domain}. Voici les faits : "
                        # On essaie d'extraire la situation de la réflexion CoT
                        facts_match = re.search(r"(faits|situation|contexte)\s*:\s*(.*)", thinking_part, re.IGNORECASE)
                        if facts_match:
                            simulated_question = "Question juridique québécoise : " + facts_match.group(2).split("\n")[0].strip()
                        else:
                            simulated_question = f"Quelle est la responsabilité juridique selon l'{art_name} du CCQ dans une situation de {art_domain.lower()} ?"
                            
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
                    print(f"Erreur d'appel API pour {art_name} (scénario {j}) : {e}")
                    continue
                    
    print(f"Génération de cas CCQ terminée ! {success_count} scénarios CoT ajoutés dans '{args.output_file}'.")

if __name__ == "__main__":
    main()
