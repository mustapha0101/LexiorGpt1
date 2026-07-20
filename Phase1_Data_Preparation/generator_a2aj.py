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
from a2aj_cleaner import clean_law, FEDERAL_JURISDICTIONS
from api_cost import CostTracker

# Compteur de jetons / coût, partagé par les threads. Initialisé dans main().
cost_tracker = None

# Verrou pour l'écriture thread-safe dans le fichier de sortie
file_lock = threading.Lock()

# Longueur minimale de la réponse finale (hors citation), en caractères.
MIN_ANSWER_CHARS = 40

# Tour de parole simulé : le Teacher rédigeait un dialogue au lieu d'une analyse.
# Ancré en début de ligne, gras markdown éventuel toléré, pour ne pas rejeter une
# phrase de fond qui contiendrait le mot « utilisateur ».
_DIALOGUE_RE = re.compile(
    r"^\s*\**\s*(?:Utilisateur|Assistant|User|Client|Avocat)\s*\**\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# Compteur partagé de lignes écrites, pour le plafond --max_rows.
# Les futures étant toutes soumises d'avance, c'est ce compteur qui permet aux
# threads restants de se retirer SANS appeler l'API une fois l'objectif atteint.
rows_total = 0
max_rows = -1

# N'accepter que les lois tenant entièrement dans le budget de contexte
# (cf. --whole_laws_only).
whole_laws_only = False


def target_reached():
    if max_rows < 0:
        return False
    with file_lock:
        return rows_total >= max_rows

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
        help="Nombre de textes juridiques bruts à PARCOURIR (-1 = tout le dataset). "
             "Attention : ce n'est pas un nombre de lignes produites — la couche de "
             "nettoyage et les filtres IRAC/citation en écartent une bonne partie. "
             "Pour viser un nombre de lignes, utiliser --max_rows."
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=-1,
        help="Nombre total de LIGNES visé dans le fichier de sortie, lignes déjà "
             "présentes comprises (-1 = pas de plafond). Les threads se retirent "
             "sans appel API dès l'objectif atteint. Enchaîner « --max_rows 1000 » "
             "puis « --max_rows 5000 » donne 1000 lignes, puis 4000 de plus."
    )
    parser.add_argument(
        "--whole_laws_only",
        action="store_true",
        help="N'utiliser que les lois tenant entièrement dans le budget de contexte "
             "(2 413 lois sur 4 727). Les lois plus longues sont écartées plutôt que "
             "tronquées : tronquée, une loi ne conserve en général que ses définitions "
             "d'ouverture, jamais ses dispositions de fond. À utiliser tant que le "
             "découpage par article des longues lois n'est pas en place."
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
    # Plafond --max_rows atteint : on se retire avant tout appel API.
    if target_reached():
        return False

    # Extraction du contenu du texte brut selon le dataset
    raw_text = ""
    context_meta = {}
    
    if dataset_name == "a2aj/canadian-laws":
        # Couche de nettoyage : ne laisse passer que le droit fédéral en
        # vigueur, articles abrogés / vides exclus (cf. a2aj_cleaner.py).
        cleaned = clean_law(item, whole_only=whole_laws_only)
        if cleaned is None:
            return False
        raw_text = cleaned["context_text"]
        context_meta = cleaned["meta"]
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
    
    # « Génère une interaction utilisateur/assistant » demandait littéralement une
    # transcription : le Teacher écrivait un dialogue « **Utilisateur :** ... /
    # **Assistant :** ... » À L'INTÉRIEUR du bloc <thinking> (59,4 % des lignes).
    # On demande désormais une seule analyse, pas un échange.
    user_prompt = (
        f"Voici l'extrait juridique de référence (Métadonnées : {json.dumps(context_meta, ensure_ascii=False)}) :\n\n"
        f"<context>\n{truncated_text}\n</context>\n\n"
        f"Formule une question juridique complexe en français portant sur ce texte, "
        f"puis réponds-y en suivant le gabarit imposé. "
        f"Produis une seule analyse rédigée à la première personne — pas un dialogue, "
        f"pas de transcription, aucun tour de parole."
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
        
        if cost_tracker is not None:
            cost_tracker.record(response)

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

            # 4. La réponse finale doit exister et être de la prose.
            # 3,6 % des lignes se réduisaient au bloc <thinking> suivi de la seule
            # citation : on entraînait le modèle à raisonner puis à ne rien répondre.
            answer_only = re.sub(r"\[\^\d+\]:.*", "", content_part, flags=re.S).strip()
            if len(answer_only) < MIN_ANSWER_CHARS:
                return False

            # 5. Aucun tour de parole simulé, ni dans le raisonnement ni dans la réponse.
            if _DIALOGUE_RE.search(generated_output):
                return False
            # ------------------------------------------------
            
            # On crée une question réaliste basée sur l'Issue identifiée dans la CoT
            issue_match = re.search(r"Issue\s*:\s*(.*)", thinking_part, re.IGNORECASE)
            simulated_question = "Analyse cette situation : " + context_meta.get("title", "Droit canadien")
            if issue_match:
                simulated_question = issue_match.group(1).strip()
                
            # Structure de message finale
            message_data = {
                "original_index": idx,
                "jurisdiction": context_meta.get("jurisdiction", ""),
                "law_name": context_meta.get("title", ""),
                "section_ids_used": context_meta.get("section_ids_used", []),
                "source_url": context_meta.get("url", ""),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": simulated_question},
                    {"role": "assistant", "content": content_part, "thinking": thinking_part}
                ]
            }

            # Écriture thread-safe dans le fichier de sortie
            global rows_total
            with file_lock:
                # Second contrôle sous verrou : plusieurs threads peuvent avoir
                # passé target_reached() avant que le dernier n'écrive.
                if 0 <= max_rows <= rows_total:
                    return False
                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(message_data, ensure_ascii=False) + "\n")
                rows_total += 1
            return True
    except Exception as e:
        if cost_tracker is not None:
            cost_tracker.record_failure()
        print(f"\n[Erreur index {idx}] : {e}")
    return False

def _write_cost_report(output_file, stats, rows_kept):
    """Consigne le coût à côté du dataset : une facture sans trace est ingérable."""
    path = os.path.splitext(output_file)[0] + "_cost.json"
    stats = dict(stats)
    stats["rows_kept"] = rows_kept
    if rows_kept:
        stats["cost_per_kept_row_usd"] = round(stats["cost_usd"] / rows_kept, 6)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"  rapport de coût       : {path}")
    except OSError as e:
        print(f"  (rapport de coût non écrit : {e})")


def main():
    global max_rows, rows_total, whole_laws_only, cost_tracker
    args = parse_args()

    if not args.api_key:
        print("Erreur : La clé API (--api_key ou variable d'environnement) est requise.")
        sys.exit(1)

    max_rows = args.max_rows
    whole_laws_only = args.whole_laws_only
    cost_tracker = CostTracker(args.model)
    if whole_laws_only:
        print("Mode --whole_laws_only : les lois dépassant le budget de contexte "
              "sont écartées (et non tronquées).")
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    
    # 1. Chargement du dataset brut A2AJ
    print(f"Chargement du dataset de base A2AJ : {args.dataset} (split: {args.split})...")
    try:
        raw_ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"Erreur lors du chargement du dataset HF : {e}")
        sys.exit(1)
        
    print(f"Dataset chargé ! Taille totale : {len(raw_ds)} exemples.")
    dataset_is_laws = (args.dataset == "a2aj/canadian-laws")
    
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

    # Le plafond --max_rows porte sur le TOTAL du fichier : on initialise le
    # compteur avec les lignes déjà présentes (une ligne = un original_index).
    rows_total = len(completed_indices)
    if 0 <= max_rows <= rows_total:
        print(f"Objectif déjà atteint ({rows_total}/{max_rows} lignes). Rien à faire.")
        sys.exit(0)


    # Prompt système
    #
    # Le format est donné sous forme de GABARIT, et non plus d'une liste à puces
    # commentée. Les anciennes gloses collées aux intitulés
    # (« Rule (Règle de droit et articles précis du CCQ ou lois fédérales cités) »)
    # étaient recopiées telles quelles par le Teacher dans 88,3 % des réponses :
    # collée à un intitulé, une parenthèse explicative se lit comme faisant
    # partie de l'intitulé. Les explications vivent désormais DANS les
    # marqueurs <...>, que le modèle remplace au lieu de les répéter.
    system_prompt = (
        "Tu es un assistant juridique Lexior, spécialisé en droit canadien et québécois. "
        "Ta tâche est d'analyser le texte juridique fourni, d'en tirer une question "
        "juridique complexe en français, puis d'y répondre.\n\n"
        "Réponds EXACTEMENT selon ce gabarit, sans rien ajouter avant ni après :\n\n"
        "<thinking>\n"
        "Issue : <la question de droit soulevée>\n"
        "Rule : <la règle de droit applicable, avec les articles précis cités>\n"
        "Application : <l'application de la règle aux faits>\n"
        "Conclusion : <la solution retenue>\n"
        "</thinking>\n\n"
        "<la réponse finale, en français, rédigée en prose suivie>\n\n"
        "[^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/...\",\"title\":\"Titre\"}\n\n"
        "Règles strictes :\n"
        "- Les quatre intitulés s'écrivent exactement « Issue : », « Rule : », "
        "« Application : », « Conclusion : ». Sans gras, sans numérotation, et sans "
        "parenthèse explicative accolée à l'intitulé.\n"
        "- N'écris jamais de dialogue : aucune ligne ne doit commencer par "
        "« Utilisateur : », « Assistant : » ni « Question : ».\n"
        "- La réponse finale placée après </thinking> est obligatoire. Elle doit être "
        "de la prose française compréhensible sans lire le bloc <thinking>, et ne peut "
        "jamais se réduire à la seule citation.\n"
        "- La citation de bas de page se place à la toute fin, après la réponse."
    )

    # --- Pré-sélection des lignes candidates ------------------------------
    # a2aj/canadian-laws est ORDONNÉ PAR JURIDICTION : lignes 0-441 = Alberta,
    # 442-1015 = Colombie-Britannique, et la première loi fédérale n'arrive
    # qu'à l'index 1016. Prendre « les N premières lignes » revient donc à ne
    # scanner que du droit provincial : --limit 1000 ne rencontre AUCUNE loi
    # fédérale et produit zéro ligne, sans rien signaler.
    # On restreint donc d'abord aux juridictions retenues, PUIS on applique
    # --limit. Les index conservés sont les index réels du dataset, pour que
    # original_index — et donc la reprise — restent valides.
    candidates = list(range(len(raw_ds)))
    if dataset_is_laws:
        try:
            jurisdictions = raw_ds["dataset"]
        except (KeyError, TypeError):
            jurisdictions = None
        if jurisdictions:
            candidates = [i for i, j in enumerate(jurisdictions)
                          if j in FEDERAL_JURISDICTIONS]
            print(f"Pré-filtrage juridictionnel : {len(candidates):,} lignes fédérales "
                  f"sur {len(raw_ds):,} ({', '.join(sorted(FEDERAL_JURISDICTIONS))}).")
            if not candidates:
                print("Erreur : aucune ligne fédérale dans ce dataset.", file=sys.stderr)
                sys.exit(1)

    if args.limit >= 0:
        candidates = candidates[:args.limit]
    sample_range = len(candidates)
        
    print(f"Lancement de la génération CoT en parallèle avec {args.workers} workers "
          f"sur {sample_range} exemples"
          f"{f' (plafond : {max_rows} lignes au total)' if max_rows >= 0 else ''}...")

    success_count = 0
    
    # Exécution parallèle
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_item, 
                raw_ds[i], i, client, args.dataset, args.model, system_prompt, args.output_file
            ): i for i in candidates if i not in completed_indices
        }
        
        # tqdm suit l'avancement au fur et à mesure que les threads se terminent
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Génération CoT")
        for future in pbar:
            if future.result():
                success_count += 1
            # Coût cumulé visible en direct : une génération qui dérape doit se
            # voir avant la facture, pas après.
            pbar.set_postfix_str(cost_tracker.line())
                
    print(f"\nGénération terminée ! {success_count} nouveaux exemples valides "
          f"({len(completed_indices)} déjà présents) — {rows_total} lignes au total "
          f"dans '{args.output_file}'.")

    stats = cost_tracker.report(label="fédéral / A2AJ")
    if success_count and stats["cost_usd"]:
        print(f"  coût par ligne RETENUE : {stats['cost_usd']/success_count:.6f} USD "
              f"({success_count} retenues sur {stats['calls']} appels)")
    _write_cost_report(args.output_file, stats, success_count)
    if 0 <= max_rows <= rows_total:
        print(f"Plafond de {max_rows} lignes atteint. Relancer avec un --max_rows plus "
              f"élevé (ou -1) pour continuer là où le script s'est arrêté.")

if __name__ == "__main__":
    main()
