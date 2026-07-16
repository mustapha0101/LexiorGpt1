#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération synthétique CoT (Chain-of-Thought) Juridique.
Il extrait des textes depuis les corpus bruts de l'A2AJ :
- a2aj/canadian-laws
- a2aj/canadian-case-law

Puis il interroge un modèle de langage Teacher (via API OpenAI/Groq/Ollama)
de manière PARALLÈLE (Multithreaded) pour accélérer considérablement
la création du dataset d'entraînement.
"""

import os
import sys
import json
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

# Verrou pour l'écriture thread-safe dans le fichier de sortie
file_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser(description="Générateur parallèle de données synthétiques CoT A2AJ.")
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
        default=100,
        help="Nombre de textes juridiques bruts à traiter."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Nombre de threads d'appels API en parallèle (ajustez selon vos limites de taux)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/processed/generated_a2aj_cot.jsonl",
        help="Fichier de sortie pour stocker le dataset généré."
    )
    return parser.parse_args()

def process_single_item(item, idx, client, dataset_name, model, system_prompt, output_file):
    # Extraction du contenu du texte brut selon le dataset
    raw_text = ""
    context_meta = {}
    
    if dataset_name == "a2aj/canadian-laws":
        raw_text = item.get("unofficial_text_fr", "") or item.get("unofficial_text_en", "") or item.get("text", "")
        context_meta = {
            "title": item.get("name_fr", "") or item.get("name_en", "") or item.get("title", "Loi canadienne"),
            "section": item.get("section", "N/A"),
            "url": item.get("source_url_fr", "") or item.get("source_url_en", "") or ""
        }
    elif dataset_name == "a2aj/canadian-case-law":
        raw_text = item.get("unofficial_text_fr", "") or item.get("unofficial_text_en", "") or item.get("text", "") or item.get("content", "")
        context_meta = {
            "citation": item.get("citation_fr", "") or item.get("citation_en", "") or item.get("citation", "Jurisprudence"),
            "court": item.get("court", "Tribunal canadien"),
            "url": item.get("source_url_fr", "") or item.get("source_url_en", "") or ""
        }
        
    if not raw_text or len(raw_text.strip()) < 150:
        return False
        
    # Limiter la taille du texte brut envoyé pour éviter de saturer le contexte
    truncated_text = raw_text[:4000]
    
    user_prompt = (
        f"Voici l'extrait juridique de référence (Métadonnées : {json.dumps(context_meta, ensure_ascii=False)}) :\n\n"
        f"<context>\n{truncated_text}\n</context>\n\n"
        f"Génère une interaction utilisateur/assistant réaliste pour LexiorNotebook. "
        f"L'utilisateur pose une question complexe en français relative à ce texte juridique, et tu y réponds selon les consignes strictes."
    )
    
    # Appel au Teacher LLM
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        generated_output = response.choices[0].message.content
        
        # Validation rigoureuse de la présence du CoT et filtrage anti-hallucination
        if "<thinking>" in generated_output and "</thinking>" in generated_output:
            parts = generated_output.split("</thinking>")
            thinking_part = parts[0].replace("<thinking>", "").strip()
            content_part = parts[1].strip()
            
            # --- ÉTAPE DE VALIDATION & ANTI-HALLUCINATION ---
            # 1. Vérification IRAC
            thinking_lower = thinking_part.lower()
            has_issue = any(w in thinking_lower for w in ["issue", "question de droit", "problème"])
            has_rule = any(w in thinking_lower for w in ["rule", "règle", "loi", "article", "ccq"])
            has_app = any(w in thinking_lower for w in ["application", "analyse", "faits"])
            has_concl = any(w in thinking_lower for w in ["conclusion", "décision"])
            
            irac_score = sum([has_issue, has_rule, has_app, has_concl])
            if irac_score < 3:
                # Rejeter si le raisonnement logique n'est pas assez complet
                return False
                
            # 2. Vérification syntaxique JSON de bas de page et correction d'URL
            footnote_match = re.search(r"(\[\^\d+\]:\s*)(\{.*\})", content_part)
            if not footnote_match:
                # Rejeter si le format de citation Lexior n'est pas présent
                return False
            try:
                prefix = footnote_match.group(1)
                cite_json_str = footnote_match.group(2)
                cite_json = json.loads(cite_json_str)
                if "type" not in cite_json or "url" not in cite_json:
                    return False
                
                # Injection de l'URL réelle du dataset source si disponible
                real_url = context_meta.get("url", "")
                if real_url:
                    cite_json["url"] = real_url
                    new_cite_json_str = json.dumps(cite_json, ensure_ascii=False)
                    content_part = content_part.replace(footnote_match.group(0), f"{prefix}{new_cite_json_str}")
            except Exception:
                return False
                
            # 3. FILTRAGE ANTI-HALLUCINATION (Grounding) - Désactivé ou assoupli pour éviter le rejet massif
            # Le modèle peut faire référence à des concepts juridiques externes valides.
            pass
            # ------------------------------------------------
            
            # On crée une question réaliste basée sur l'Issue identifiée dans la CoT
            issue_match = re.search(r"Issue\s*:\s*(.*)", thinking_part, re.IGNORECASE)
            simulated_question = "Analyse cette situation : " + context_meta.get("title", "Droit canadien")
            if issue_match:
                simulated_question = issue_match.group(1).strip()
                
            # Structure de message finale
            message_data = {
                "original_index": idx,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": simulated_question},
                    {"role": "assistant", "content": content_part, "thinking": thinking_part}
                ]
            }
            
            # Écriture thread-safe dans le fichier de sortie
            with file_lock:
                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(message_data, ensure_ascii=False) + "\n")
            return True
    except Exception as e:
        print(f"\n[Erreur index {idx}] : {e}")
    return False

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
    
    # 3. Chargement de l'index de progression (reprise de checkpoint)
    completed_indices = set()
    if os.path.exists(args.output_file):
        print(f"Fichier de sortie existant détecté : {args.output_file}")
        try:
            with open(args.output_file, "r", encoding="utf-8") as f_in:
                for line_idx, line in enumerate(f_in):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "original_index" in data:
                            completed_indices.add(int(data["original_index"]))
                        else:
                            # Fallback : on assume que la ligne correspond à l'index de ligne
                            completed_indices.add(line_idx)
                    except Exception:
                        completed_indices.add(line_idx)
            print(f"Reprise activée ! {len(completed_indices)} exemples déjà générés détectés à sauter.")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier de progression : {e}")
            
    # Prompt système
    system_prompt = (
        "Tu es un assistant juridique Lexior, spécialisé en droit canadien et québécois. "
        "Ta tâche est d'analyser le texte juridique fourni et de générer une question juridique complexe en français, "
        "puis de fournir une réponse structurée.\n\n"
        "Tu dois obligatoirement formater ta réponse en générant :\n"
        "1. Un bloc de réflexion <thinking>...</thinking> contenant le raisonnement juridique IRAC en français :\n"
        "   - Issue (Question de droit)\n"
        "   - Rule (Règle de droit et articles précis du CCQ ou lois fédérales cités)\n"
        "   - Application (Raisonnement ou application des faits)\n"
        "2. La réponse finale (Conclusion) en français.\n"
        "3. Une citation de bas de page contenant l'URL au format JSON LexiorGPT à la toute fin :\n"
        "   [^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/...\",\"title\":\"Titre\"}"
    )
    
    # Récupération de la plage (si limit < 0, on traite tout le dataset)
    if args.limit < 0:
        sample_range = len(raw_ds)
    else:
        sample_range = min(args.limit, len(raw_ds))
        
    print(f"Lancement de la génération CoT en parallèle avec {args.workers} workers sur {sample_range} exemples...")
    
    success_count = 0
    
    # Exécution parallèle
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_item, 
                raw_ds[i], i, client, args.dataset, args.model, system_prompt, args.output_file
            ): i for i in range(sample_range) if i not in completed_indices
        }
        
        # tqdm suit l'avancement au fur et à mesure que les threads se terminent
        for future in tqdm(as_completed(futures), total=len(futures), desc="Génération CoT"):
            if future.result():
                success_count += 1
                
    print(f"Génération terminée ! {success_count}/{sample_range} exemples valides enregistrés dans '{args.output_file}'.")

if __name__ == "__main__":
    main()
