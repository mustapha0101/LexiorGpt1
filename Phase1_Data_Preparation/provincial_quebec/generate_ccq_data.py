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
        "article": "Article 1870",
        "texte": "Le locataire peut sous-louer tout ou partie du bien loué ou céder le bail. Il est alors tenu d’aviser le locateur de son intention, de lui indiquer le nom et l’adresse de la personne à qui il entend sous-louer le bien ou céder le bail et d’obtenir le consentement du locateur à la sous-location ou à la cession.",
        "domaine": "Cession de bail et sous-location"
    },
    {
        "article": "Article 1915",
        "texte": "Le locataire peut abandonner son logement s’il devient impropre à l’habitation. Il est alors tenu d’aviser le locateur de l’état du logement, avant l’abandon ou dans les 10 jours qui suivent.",
        "domaine": "Logement impropre à l'habitation"
    },
    {
        "article": "Article 1971",
        "texte": "Le locateur peut obtenir la résiliation du bail si le locataire est en retard de plus de trois semaines pour le paiement du loyer ou, encore, s’il en subit un préjudice sérieux, lorsque le locataire en retarde fréquemment le paiement.",
        "domaine": "Résiliation du bail pour retard de loyer"
    },
    {
        "article": "Article 2313",
        "texte": "Le prêt à usage ou commodat est un contrat par lequel le prêteur met un bien à la disposition de l'emprunteur, à charge pour lui de le lui restituer après un certain temps.",
        "domaine": "Prêt à usage (commodat)"
    },
    {
        "article": "Article 2327",
        "texte": "L'emprunteur est tenu de veiller, en bon père de famille, à la garde et à la conservation du bien prêté.",
        "domaine": "Garde et conservation du bien prêté"
    },
    {
        "article": "Article 2280",
        "texte": "Le dépôt est le contrat par lequel une personne, le déposant, remet un bien à une autre personne, le dépositaire, qui s'oblige à le garder et à le restituer.",
        "domaine": "Contrat de dépôt"
    },
    {
        "article": "Article 2283",
        "texte": "Le dépositaire doit apporter, dans la garde du bien, le même soin qu'il apporte à la garde de ses propres biens.",
        "domaine": "Soin dans la garde du dépôt"
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
    
    # Prise en charge automatique de la clé Gemini ou de la clé d'environnement
    api_key = args.api_key or os.environ.get("COPILOT_GEMINI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    api_url = args.api_url
    model_name = args.model
    
    if not api_key:
        print("Erreur : La clé API est requise (définissez COPILOT_GEMINI_API_KEY ou OPENAI_API_KEY).")
        sys.exit(1)
        
    if api_key.startswith("AQ."):
        api_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        if model_name == "gpt-4o-mini" or "Qwen" in model_name:
            model_name = "gemini-2.5-flash"
        print(f"Utilisation de l'API Gemini avec le modèle {model_name}...")
    
    client = OpenAI(base_url=api_url, api_key=api_key)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Écraser pour repartir de propre pour avoir des données neuves et saines
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        print(f"Lancement de la génération de scénarios pour {len(CCQ_ARTICLES_SAMPLES)} articles clés du CCQ...")
        success_count = 0
        
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
                    f"avec la citation de bas de page conforme. N'utilise pas d'autres langues ou de caractères étranges."
                )
                
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7
                    )
                    
                    generated_output = response.choices[0].message.content
                    
                    # Extraction robuste avec fallback
                    thinking_part = ""
                    content_part = ""
                    
                    if "<thinking>" in generated_output and "</thinking>" in generated_output:
                        parts = generated_output.split("</thinking>")
                        thinking_part = parts[0].replace("<thinking>", "").strip()
                        content_part = parts[1].strip()
                    else:
                        conclusion_match = re.search(r"(?:Conclusion|Réponse finale|### Conclusion)(.*)", generated_output, re.IGNORECASE | re.DOTALL)
                        if conclusion_match:
                            content_part = conclusion_match.group(0).strip()
                            thinking_part = generated_output.replace(content_part, "").strip()
                        else:
                            content_part = generated_output
                            thinking_part = f"Analyse et raisonnement juridique basés sur l'article {art_name} du CCQ."
                            
                    # Nettoyer les dialogues factices pour la question de l'utilisateur
                    simulated_question = f"Situation de litige concernant le domaine : {art_domain}. Quels sont les droits et recours applicables ?"
                    
                    # Extraction du scénario factuel
                    facts_match = re.search(r"(faits|situation|contexte)\s*:\s*(.*)", thinking_part, re.IGNORECASE)
                    if facts_match:
                        facts_sentence = facts_match.group(2).split("\n")[0].strip()
                        if facts_sentence:
                            simulated_question = f"En droit civil québécois : {facts_sentence}"
                    
                    # Nettoyer d'éventuels tags restants
                    thinking_part = thinking_part.replace("<thinking>", "").replace("</thinking>", "").strip()
                    
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
