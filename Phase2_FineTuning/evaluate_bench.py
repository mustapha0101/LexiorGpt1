#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'évaluation scientifique de LexiorGPT.
Il compare un modèle (de base ou entraîné) sur trois critères clés :
1. Conformité syntaxique du format XML/JSON.
2. Adhérence à la structure de raisonnement logique IRAC (Issue, Rule, Application, Conclusion).
3. Exactitude et pertinence des citations juridiques (grounding).
"""

import os
import sys
import json
import argparse
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Script d'évaluation scientifique pour LexiorGPT.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Chemin local du modèle ou identifiant HF à évaluer."
    )
    parser.add_argument(
        "--eval_file", 
        type=str, 
        default="../Phase1_Data_Preparation/data/processed/test_dataset.jsonl",
        help="Chemin vers le dataset de test formaté."
    )
    parser.add_argument(
        "--output_report", 
        type=str, 
        default="outputs/evaluation_report.json",
        help="Chemin de sauvegarde du rapport d'évaluation."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=50, 
        help="Nombre maximal d'exemples à évaluer pour la métrique."
    )
    return parser.parse_args()

def check_xml_json_compliance(response_text):
    """Vérifie si le modèle sépare bien la réflexion et produit une citation JSON valide."""
    has_thinking = "<thinking>" in response_text and "</thinking>" in response_text
    
    # Extraction de la citation de bas de page JSON
    footnote_match = re.search(r"\[\^\d+\]:\s*(\{.*\})", response_text)
    has_valid_json_citation = False
    
    if footnote_match:
        try:
            json_data = json.loads(footnote_match.group(1))
            if "type" in json_data and ("url" in json_data or "title" in json_data):
                has_valid_json_citation = True
        except json.JSONDecodeError:
            pass
            
    return {
        "has_thinking": has_thinking,
        "has_valid_json_citation": has_valid_json_citation,
        "syntax_score": float(has_thinking) * 0.5 + float(has_valid_json_citation) * 0.5
    }

def check_irac_adherence(thinking_text):
    """Vérifie la présence des étapes de la méthode IRAC dans la réflexion."""
    if not thinking_text:
        return {"score": 0.0, "details": []}
        
    thinking_lower = thinking_text.lower()
    
    # Mots-clés indicatifs de chaque section (bilingue)
    checks = {
        "issue": ["issue", "question de droit", "problème"],
        "rule": ["rule", "règle", "loi", "article", "ccq", "jurisprudence"],
        "application": ["application", "analyse", "faits", "application aux faits"],
        "conclusion": ["conclusion", "décision", "résolution"]
    }
    
    score_components = []
    details = {}
    
    for key, keywords in checks.items():
        found = any(kw in thinking_lower for kw in keywords)
        score_components.append(float(found))
        details[key] = found
        
    final_score = sum(score_components) / len(score_components)
    return {
        "score": final_score,
        "details": details
    }

def main():
    args = parse_args()
    
    if not os.path.exists(args.eval_file):
        print(f"Erreur : Le fichier de test '{args.eval_file}' est introuvable. Veuillez exécuter la Phase 1 d'abord.")
        sys.exit(1)
        
    print(f"Chargement du tokenizer et du modèle de test depuis : {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Erreur lors du chargement du modèle pour évaluation : {e}")
        sys.exit(1)
        
    # Lecture des exemples de test
    test_samples = []
    with open(args.eval_file, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))
            if len(test_samples) >= args.limit:
                break
                
    print(f"Début de l'évaluation scientifique sur {len(test_samples)} échantillons...")
    
    results = []
    total_syntax = 0.0
    total_irac = 0.0
    total_citations = 0.0
    
    for idx, sample in enumerate(tqdm(test_samples, desc="Évaluation")):
        # Extraction du prompt d'entrée (l'historique des messages sans la réponse de l'assistant)
        # Pour extraire le prompt, on applique le template de chat sans inclure le dernier message assistant
        full_text = sample["text"]
        
        # Séparation de l'entrée et de la cible attendue
        split_marker = "<|im_start|>assistant" if "im_start" in full_text else "assistant\n"
        if split_marker in full_text:
            prompt_input = full_text.split(split_marker)[0] + split_marker
        else:
            prompt_input = full_text  # Fallback
            
        inputs = tokenizer(prompt_input, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 1. Évaluation syntaxique (Balises XML & Citation JSON)
        syntax_eval = check_xml_json_compliance(generated_response)
        
        # 2. Évaluation du raisonnement (IRAC)
        thinking_text = ""
        if "<thinking>" in generated_response and "</thinking>" in generated_response:
            thinking_text = generated_response.split("</thinking>")[0].replace("<thinking>", "").strip()
        irac_eval = check_irac_adherence(thinking_text)
        
        # 3. Évaluation de l'alignement canadien (Grounding / Citation)
        # On vérifie si la réponse contient des articles de loi ou de la jurisprudence canadienne
        has_canadian_cite = bool(re.search(r"(ccq|canlii|csc|scc|art\.|l.r.c|l.q|décision)", generated_response, re.IGNORECASE))
        citation_score = 1.0 if has_canadian_cite else 0.0
        
        results.append({
            "id": idx,
            "response": generated_response,
            "syntax": syntax_eval,
            "irac": irac_eval,
            "citation_grounding": has_canadian_cite
        })
        
        total_syntax += syntax_eval["syntax_score"]
        total_irac += irac_eval["score"]
        total_citations += citation_score
        
    num_samples = len(test_samples)
    final_report = {
        "model_evaluated": args.model_path,
        "dataset_used": args.eval_file,
        "sample_count": num_samples,
        "metrics": {
            "xml_json_syntax_compliance_rate": total_syntax / num_samples,
            "irac_logical_adherence_rate": total_irac / num_samples,
            "canadian_citation_grounding_rate": total_citations / num_samples
        },
        "detailed_results": results
    }
    
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f_rep:
        json.dump(final_report, f_rep, indent=2, ensure_ascii=False)
        
    print(f"\nRapport d'évaluation enregistré dans '{args.output_report}'.")
    print("=== MÉTRIQUES DE PERFORMANCE ===")
    print(f"- Taux de conformité syntaxique (XML/JSON) : {final_report['metrics']['xml_json_syntax_compliance_rate']*100:.2f}%")
    print(f"- Taux d'adhérence logique IRAC : {final_report['metrics']['irac_logical_adherence_rate']*100:.2f}%")
    print(f"- Taux de citations canadiennes valides : {final_report['metrics']['canadian_citation_grounding_rate']*100:.2f}%")

if __name__ == "__main__":
    main()
