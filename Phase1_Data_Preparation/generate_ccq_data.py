#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de génération de scénarios juridiques basés sur le droit québécois.
Il lit les articles du Code civil du Québec (CCQ) et du Code de procédure
civile (CPC) depuis le corpus intelliwork/canadian-quebec-law-corpus, nettoyé
par ccq_cleaner.py, et génère des paires d'entraînement CoT (IRAC) en français
québécois, permettant au modèle d'apprendre la substance de ces codes.
"""

import os
import sys
import json
import re
import argparse
from collections import Counter
from tqdm import tqdm
from openai import OpenAI
from ccq_cleaner import load_quebec_articles
from api_cost import CostTracker

# Longueur minimale de la réponse finale (hors citation), en caractères.
MIN_ANSWER_CHARS = 40

# Longueur minimale de la mise en situation, qui devient le tour utilisateur.
MIN_QUESTION_CHARS = 60

# Tour de parole simulé : le Teacher rédigeait un dialogue au lieu d'une analyse.
# Ancré en début de ligne pour ne pas rejeter une phrase de fond contenant le mot.
_DIALOGUE_RE = re.compile(
    r"^\s*\**\s*(?:Utilisateur|Assistant|User|Client|Avocat)\s*\**\s*:",
    re.IGNORECASE | re.MULTILINE,
)

# Les articles ne sont plus codés en dur : ils proviennent directement du
# corpus intelliwork/canadian-quebec-law-corpus, nettoyé par ccq_cleaner.py.
# (L'ancienne liste CCQ_ARTICLES_SAMPLES contenait 7 articles, dont 5 dont le
#  libellé s'écartait du texte officiel — voir ccq_cleaner.py.)


def _write_cost_report(output_file, stats, rows_kept):
    """Consigne le coût à côté du dataset."""
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


def load_done_keys(output_file):
    """Relit le fichier de sortie et renvoie les scénarios déjà générés.

    Retourne un ensemble de couples (source_id, scenario_index).

    Le fichier étant ouvert en mode ajout, cet index est ce qui empêche une
    seconde exécution de tout regénérer et de dupliquer les lignes. Il permet
    aussi de reprendre après une interruption sans repartir de zéro.

    Les lignes illisibles ou dépourvues de clé sont ignorées silencieusement :
    mieux vaut regénérer un scénario que d'en perdre la trace.
    """
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sid = data.get("source_id")
                idx = data.get("scenario_index")
                if sid is not None and idx is not None:
                    done.add((sid, int(idx)))
            except (ValueError, TypeError):
                continue
    return done

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
        "--dataset",
        type=str,
        default="intelliwork/canadian-quebec-law-corpus",
        help="Corpus de droit québécois à charger depuis le Hub (dépôt privé)."
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        choices=["Code civil du Québec", "Code de procédure civile du Québec"],
        help="Restreindre à un seul code. Par défaut : les deux."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Nombre maximal d'ARTICLES à traiter (-1 = tous). "
             "Le corpus complet compte 4 278 articles exploitables."
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=-1,
        help="Nombre total de LIGNES visé dans le fichier de sortie, lignes déjà "
             "présentes comprises (-1 = pas de plafond). Le script s'arrête dès que "
             "le fichier atteint ce total. Enchaîner « --max_rows 1000 » puis "
             "« --max_rows 5000 » donne 1000 lignes, puis 4000 de plus."
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
    cost_tracker = CostTracker(args.model)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Chargement + nettoyage du corpus québécois (cf. ccq_cleaner.py) :
    # articles abrogés « (Abrogé). », omis « (Omis). » et vides sont écartés.
    print(f"Chargement du corpus de droit québécois : {args.dataset}...")
    articles, stats = load_quebec_articles(
        dataset_name=args.dataset,
        codes=[args.code] if args.code else None,
        limit=args.limit,
    )
    print(f"Nettoyage : {stats}")
    if not articles:
        print("Erreur : aucun article exploitable après nettoyage.")
        sys.exit(1)

    # --- Index de reprise -------------------------------------------------
    # Le fichier est ouvert en mode "a" (ajout). Sans index, une seconde
    # exécution réécrirait tout depuis le début et dupliquerait chaque ligne.
    # La clé est le COUPLE (source_id, scenario_index) et non le seul article :
    # un article dont 3 scénarios sur 10 sont faits doit reprendre au 4e.
    done = load_done_keys(args.output_file)
    rows_existing = len(done)
    if rows_existing:
        print(f"Reprise activée : {rows_existing} scénarios déjà présents dans "
              f"'{args.output_file}' seront ignorés.")

    # --- Plafond de lignes ------------------------------------------------
    # --max_rows compte le TOTAL de lignes visé dans le fichier, lignes déjà
    # présentes comprises. Ainsi « --max_rows 1000 » puis « --max_rows 5000 »
    # produit 1000 lignes, puis 4000 de plus — et non 5000 de plus.
    if args.max_rows >= 0 and rows_existing >= args.max_rows:
        print(f"Objectif déjà atteint ({rows_existing}/{args.max_rows} lignes). Rien à faire.")
        sys.exit(0)

    planned = len(articles) * args.scenarios_per_article - rows_existing
    if args.max_rows >= 0:
        planned = min(planned, args.max_rows - rows_existing)
    print(f"Lancement de la génération sur {len(articles)} articles "
          f"({args.scenarios_per_article} scénarios chacun) — "
          f"environ {max(planned, 0)} appels API à effectuer"
          f"{f' (plafond : {args.max_rows} lignes au total)' if args.max_rows >= 0 else ''}...")

    success_count = 0
    rejects = Counter()   # pourquoi les lignes sont écartées
    rows_total = rows_existing
    stop = False

    with open(args.output_file, "a", encoding="utf-8") as f_out:
        for item in tqdm(articles, desc="Articles québécois"):
            if stop:
                break
            art_name = item["article"]
            art_text = item["texte"]
            art_domain = item["domaine"]
            art_id = item["id"]
            art_url = item.get("source_url", "")
            
            # Gabarit, et non liste à puces glosée : les parenthèses explicatives
            # accolées aux intitulés étaient recopiées telles quelles par le Teacher
            # (« Rule (La règle applicable, en citant...) : »). Les explications sont
            # désormais dans les marqueurs <...>, que le modèle remplace.
            system_prompt = (
                "Tu es un professeur de droit civil du Québec. Ta tâche est de créer des exemples "
                "de cas pratiques et de résolutions juridiques rigoureuses pour l'entraînement "
                "d'une IA juridique.\n\n"
                # Les emplacements à remplir sont notés {{...}} et NON <...> :
                # avec des chevrons, le modèle prend « </thinking> » pour un
                # emplacement à substituer et le supprime — la balise fermante
                # manquait dans 10 sorties sur 12, ce qui faisait rejeter la ligne.
                "Réponds EXACTEMENT selon ce gabarit. Les éléments notés {{...}} sont à "
                "remplacer par ton texte ; TOUT LE RESTE est littéral et doit être recopié "
                "tel quel — en particulier les balises <thinking> et </thinking>.\n\n"
                "Situation : {{la mise en situation, rédigée du point de vue de la personne "
                "qui consulte : les faits, puis sa question. 3 à 6 phrases, à la première "
                "personne.}}\n\n"
                "<thinking>\n"
                "Issue : {{la question de droit soulevée par la situation}}\n"
                "Rule : {{la règle applicable, avec l'article précis du code cité et expliqué}}\n"
                "Application : {{l'application concrète de la règle aux faits}}\n"
                "Conclusion : {{la solution retenue}}\n"
                "</thinking>\n\n"
                "{{la réponse finale, claire et synthétique, en français québécois, rédigée en prose suivie}}\n\n"
                "[^1]:{\"type\":\"url\",\"url\":\"{{URL fournie dans la demande}}\",\"title\":\"{{nom du code}}\"}\n\n"
                "Règles strictes :\n"
                "- Les balises <thinking> et </thinking> sont LITTÉRALES. La balise fermante "
                "</thinking> est obligatoire : elle sépare le raisonnement de la réponse.\n"
                "- La ligne « Situation : » est OBLIGATOIRE et vient en PREMIER, avant le bloc "
                "<thinking>. C'est elle qui énonce les faits. Le bloc <thinking> ne doit contenir "
                "que le raisonnement IRAC : il ne réénonce pas le récit.\n"
                "- Les quatre intitulés s'écrivent exactement « Issue : », « Rule : », "
                "« Application : », « Conclusion : ». Sans gras, sans numérotation, et sans "
                "parenthèse explicative accolée à l'intitulé.\n"
                "- N'écris jamais de dialogue : aucune ligne ne doit commencer par "
                "« Utilisateur : », « Assistant : », « Avocat : » ni « Client : ».\n"
                "- La réponse finale placée après </thinking> est obligatoire. Elle doit être "
                "de la prose compréhensible sans lire le bloc <thinking>, et ne peut jamais "
                "se réduire à la seule citation.\n"
                "- La citation de bas de page [^1]:{...} est OBLIGATOIRE et se place à la "
                "toute fin, après la réponse.\n"
                "- Reprends TELLE QUELLE l'URL fournie dans la demande. N'invente aucune URL "
                "et ne laisse jamais de « ... » dans la citation."
            )
            
            for j in range(args.scenarios_per_article):
                # Reprise : ce scénario précis a-t-il déjà été généré ?
                if (art_id, j) in done:
                    continue
                # Plafond atteint : on s'arrête proprement, sans appel API.
                if args.max_rows >= 0 and rows_total >= args.max_rows:
                    stop = True
                    break

                user_prompt = (
                    f"Rédige une mise en situation fictive, réaliste et complexe se déroulant au Québec, "
                    f"mettant en jeu l'application directe de l'article suivant :\n\n"
                    f"[{art_name} - Domaine : {art_domain}]\n"
                    f"Texte officiel : \"{art_text}\"\n"
                    f"URL à citer : {art_url}\n\n"
                    f"Résous ensuite ce cas en suivant le gabarit imposé. "
                    f"Produis une seule analyse rédigée d'un seul tenant — pas un dialogue, "
                    f"pas de transcription, aucun tour de parole."
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
                    
                    cost_tracker.record(response)
                    generated_output = response.choices[0].message.content
                    
                    # Ce contrôle était le SEUL chemin non instrumenté : une sortie
                    # sans balises disparaissait sans laisser de trace, et le
                    # rapport affichait « aucun rejet » alors que 86 % des appels
                    # étaient perdus. Tout abandon est désormais compté.
                    if "<thinking>" not in generated_output:
                        rejects["balise <thinking> absente"] += 1
                        continue
                    if "</thinking>" not in generated_output:
                        rejects["balise </thinking> absente (sortie non fermée)"] += 1
                        continue

                    if True:
                        # --- La SITUATION est le tour utilisateur -------------
                        # Auparavant, le Teacher rédigeait le scénario en prose et
                        # tout ce qui précédait </thinking> devenait le raisonnement
                        # de l'assistant : les faits inventés (« Jean, 35 ans, sous
                        # tutelle... ») atterrissaient donc DANS la tête du modèle,
                        # tandis que la question utilisateur était un repli générique.
                        # On entraînait ainsi le modèle à fabriquer un client que
                        # l'utilisateur n'a jamais décrit (73 % des lignes).
                        # Le scénario est désormais émis sous « Situation : » et
                        # devient le tour UTILISATEUR — sa place légitime.
                        m_sit = re.search(r"Situation\s*:\s*(.*?)(?=<thinking>)",
                                          generated_output, re.S | re.IGNORECASE)
                        if not m_sit:
                            rejects["pas de ligne « Situation : »"] += 1
                            continue
                        simulated_question = re.sub(r"\s+", " ", m_sit.group(1)).strip()
                        if len(simulated_question) < MIN_QUESTION_CHARS:
                            rejects["situation trop courte"] += 1
                            continue

                        after_sit = generated_output[m_sit.end():]
                        parts = after_sit.split("</thinking>")
                        thinking_part = parts[0].replace("<thinking>", "").strip()
                        content_part = parts[1].strip() if len(parts) > 1 else ""

                        # La réponse finale doit exister et être de la prose : 3,6 % des
                        # lignes se réduisaient au raisonnement suivi de la seule citation.
                        answer_only = re.sub(r"\[\^\d+\]:.*", "", content_part, flags=re.S).strip()
                        if len(answer_only) < MIN_ANSWER_CHARS:
                            rejects["réponse finale vide"] += 1
                            continue

                        # Aucun tour de parole simulé.
                        if _DIALOGUE_RE.search(generated_output):
                            rejects["dialogue simulé"] += 1
                            continue

                        # Le raisonnement ne doit pas re-raconter le scénario : il
                        # commence par « Issue : ».
                        if not re.match(r"\s*Issue\s*:", thinking_part, re.IGNORECASE):
                            rejects["thinking ne commence pas par Issue :"] += 1
                            continue

                        # --- Réécriture de l'URL -----------------------------
                        # Équivalent québécois de generator_a2aj.py:160. Sans cela,
                        # le Teacher recopiait le gabarit du prompt : 58 citations
                        # sur 60 valaient « .../rlrq-c-ccq-1991/... », « ... » compris.
                        if art_url:
                            m_cit = re.search(r"(\[\^\d+\]:\s*)(\{.*?\})", content_part, re.S)
                            if not m_cit:
                                rejects["citation absente"] += 1
                                continue
                            try:
                                cite = json.loads(m_cit.group(2))
                            except ValueError:
                                rejects["citation JSON illisible"] += 1
                                continue
                            cite["url"] = art_url
                            cite.setdefault("type", "url")
                            cite.setdefault("title", item["code"])
                            content_part = content_part.replace(
                                m_cit.group(0),
                                f"{m_cit.group(1)}{json.dumps(cite, ensure_ascii=False)}")
                            
                        message_data = {
                            # Clé de reprise : identifie le scénario de façon unique.
                            "source_id": art_id,
                            "scenario_index": j,
                            "code": item["code"],
                            "jurisdiction": "Québec (Provincial)",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": simulated_question},
                                {"role": "assistant", "content": content_part, "thinking": thinking_part}
                            ]
                        }

                        f_out.write(json.dumps(message_data, ensure_ascii=False) + "\n")
                        f_out.flush()  # la reprise doit survivre à une coupure brutale
                        success_count += 1
                        rows_total += 1
                except Exception as e:
                    cost_tracker.record_failure()
                    print(f"Erreur d'appel API pour {art_name} (scénario {j}) : {e}")
                    continue
                    
    print(f"\nGénération québécoise terminée ! {success_count} scénarios ajoutés "
          f"({rows_existing} déjà présents) — {rows_total} lignes au total dans "
          f"'{args.output_file}'.")
    if args.max_rows >= 0 and rows_total >= args.max_rows:
        print(f"Plafond de {args.max_rows} lignes atteint. Relancer avec un "
              f"--max_rows plus élevé (ou -1) pour continuer là où le script s'est arrêté.")

    # Ventilation des rejets. Sans elle, un garde-fou trop strict peut écarter
    # 87 % des appels sans que rien ne dise lequel — c'est exactement ce qui
    # s'est produit lorsque la règle « citation obligatoire » a disparu du
    # prompt alors que le filtre correspondant, lui, était resté.
    total_attempts = success_count + sum(rejects.values())
    if rejects:
        print(f"\n  --- REJETS ({sum(rejects.values())}/{total_attempts}) ---")
        for why, n in rejects.most_common():
            print(f"    {why:42s} {n:5,}  ({100*n/total_attempts:4.1f}%)")
    elif total_attempts:
        print(f"\n  aucun rejet sur {total_attempts} appels.")

    stats = cost_tracker.report(label="québécois / CCQ+CPC")
    if success_count and stats["cost_usd"]:
        print(f"  coût par ligne RETENUE : {stats['cost_usd']/success_count:.6f} USD "
              f"({success_count} retenues sur {stats['calls']} appels)")
    _write_cost_report(args.output_file, stats, success_count)

if __name__ == "__main__":
    main()
