#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de formatage du dataset pour la distillation Chain-of-Thought (CoT).
Il télécharge le dataset SuperMust/irac-thinking ou lit un fichier local généré par generator_a2aj.py,
fusionne le champ 'thinking' et le champ 'content' de l'assistant dans des balises <thinking>,
et applique le chat template.
"""

import hashlib
import os
import argparse
import json
import random
import re
from datasets import load_dataset
from transformers import AutoTokenizer

# Prompt système juridique canonique. Appliqué aux seuls exemples juridiques,
# et seulement à ceux que le dropout épargne.
LEGAL_SYSTEM_PROMPT = (
    "Tu es un assistant juridique Lexior, spécialisé en droit canadien et québécois. "
    "Raisonne en français selon le format IRAC. Tu dois obligatoirement baser tes analyses "
    "sur la législation et la jurisprudence canadienne/québécoise (ex: Code civil du Québec, CanLII). "
    "Lorsque tu as fini de raisonner dans tes balises <thinking>, formate tes citations de bas de page "
    "strictement sous la forme [^1]:{\"type\":\"url\",\"url\":\"https://www.canlii.org/...\",\"title\":\"Titre\"}."
)

# Colonnes de métadonnées qui doivent survivre au formatage. L'ancienne version
# faisait remove_columns(toutes) : impossible ensuite de prouver quelles lignes
# d'identité avaient été entraînées.
METADATA_COLUMNS = ["dataset_type", "identity_category", "template_group",
                    "source_id", "language", "scenario_id", "scenario_family_id",
                    "request_type", "legal_domain", "expected_jurisdiction",
                    "resolved_jurisdiction"]

# --- Gabarit ChatML strict -------------------------------------------------
# Le gabarit d'origine de Qwen2.5 contient :
#
#   {%- if messages[0]['role'] == 'system' %} ... {%- else %}
#       {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.
#            You are a helpful assistant.<|im_end|>\n' }}
#
# Autrement dit : retirer le message système n'aboutit PAS à une absence de
# prompt système — Qwen en injecte un qui affirme que le modèle EST Qwen.
# Chaque exemple d'identité aurait donc été entraîné sous un prompt disant
# « You are Qwen, created by Alibaba Cloud », ce qui contredit frontalement
# l'objectif. Idem pour les exemples juridiques touchés par le dropout.
#
# Ce gabarit rend EXACTEMENT les messages fournis, sans rien injecter.
#
# IMPORTANT : le service d'inférence doit utiliser le même gabarit, sinon
# l'entraînement et la production ne voient pas le même contexte. Cf.
# deploy_vllm.py --chat-template.
STRICT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _dropout_hit(example, rate, seed):
    """Le prompt système de cet exemple juridique doit-il être retiré ?

    Décision DÉTERMINISTE, dérivée de la graine et d'un identifiant stable :
    deux exécutions avec la même graine retirent exactement les mêmes prompts.
    Un random.random() par ligne serait sensible à l'ordre de parcours.
    """
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    ident = str(example.get("source_id")
                or example.get("original_index")
                or _user_of(example))
    digest = hashlib.sha256(f"{seed}:{ident}".encode("utf-8")).hexdigest()
    # 8 chiffres hexadécimaux -> [0, 1)
    return (int(digest[:8], 16) / 0xFFFFFFFF) < rate


def _user_of(example):
    for m in example.get("messages", []) or []:
        if m.get("role") in ("user", "human"):
            return m.get("content", "")
    return ""


def group_key(example):
    """Clé de regroupement pour un découpage train/test disjoint.

    - identité : par template_group. Toutes les paraphrases d'une même famille
      restent du même côté, sinon le jeu de test contient une reformulation
      quasi identique à une ligne d'entraînement.
    - juridique : par source_id, qui est l'identifiant stable de la SOURCE.
      C'est plus fort que la question : les 10 scénarios tirés de l'article 1457
      portent tous sur l'article 1457. Les répartir des deux côtés ferait mesurer
      au jeu de test un article déjà vu 9 fois à l'entraînement.
    - à défaut : par question normalisée.
    """
    if example.get("scenario_family_id"):
        citations = sorted({
            citation
            for grounding in (example.get("grounding") or [])
            for citation in (grounding.get("citations") or [])
        })
        return "agentic:" + str(example["scenario_family_id"]) + "|" + "|".join(citations[:3])
    if example.get("template_group"):
        return "tg:" + str(example["template_group"])
    if example.get("source_id"):
        return "sid:" + str(example["source_id"])
    return question_key(example)


def _stratum(example, per_category):
    """Strate de découpage.

    per_category=False (défaut) : on stratifie par dataset_type. Le jeu de test
    reste proportionnel, et contient de l'identité sans en absorber une part
    excessive.

    per_category=True : on stratifie aussi par catégorie d'identité, ce qui
    force chaque catégorie dans le test — au prix fort. Une catégorie ne
    comptant que 2 familles y verse 1 famille sur 2, soit 50 % de ses lignes.
    Sur l'ensemble, cela retire ~32 % du jeu d'identité de l'entraînement,
    pour un bénéfice limité : la couverture par catégorie est précisément ce
    que mesure identity_bench.jsonl (§11), qui est un jeu distinct.
    """
    dtype = example.get("dataset_type") or "legal_federal"
    if per_category and dtype in ("identity", "identity_control"):
        return f"{dtype}/{example.get('identity_category') or 'inconnue'}"
    return dtype


def _grouped_split(group_keys, strata, test_size, seed):
    """Répartit les GROUPES (jamais les lignes), stratifié par strate.

    Garantit :
      - aucun groupe des deux côtés ;
      - chaque strate ayant au moins 2 groupes est présente dans le test.
    """
    rng = random.Random(seed)

    groups_by_stratum = {}
    for key, stratum in zip(group_keys, strata):
        groups_by_stratum.setdefault(stratum, set()).add(key)

    test_groups = set()
    for stratum in sorted(groups_by_stratum):
        groups = sorted(groups_by_stratum[stratum])
        rng.shuffle(groups)
        n = int(round(len(groups) * test_size))
        if len(groups) >= 2:
            n = max(1, min(n, len(groups) - 1))   # jamais 0, jamais tout
        else:
            n = 0                                  # 1 seul groupe -> train
        test_groups.update(groups[:n])

    train_idx = [i for i, k in enumerate(group_keys) if k not in test_groups]
    test_idx = [i for i, k in enumerate(group_keys) if k in test_groups]
    return train_idx, test_idx


def _split_audit(dataset, group_keys, strata, train_idx, test_idx, output_dir):
    """Rapport d'audit du découpage (§4). Écrit aussi un JSON."""
    from collections import Counter

    def dtype_of(i):
        return dataset[i].get("dataset_type") or "legal_federal"

    def cat_of(i):
        return dataset[i].get("identity_category")

    tr_types = Counter(dtype_of(i) for i in train_idx)
    te_types = Counter(dtype_of(i) for i in test_idx)
    tr_cats = Counter(c for i in train_idx if (c := cat_of(i)))
    te_cats = Counter(c for i in test_idx if (c := cat_of(i)))

    tr_groups = {group_keys[i] for i in train_idx}
    te_groups = {group_keys[i] for i in test_idx}
    overlap_groups = tr_groups & te_groups

    tr_src = {dataset[i].get("source_id") for i in train_idx}
    te_src = {dataset[i].get("source_id") for i in test_idx}
    overlap_src = {s for s in (tr_src & te_src) if s}

    tr_q = Counter(normalize_question(dataset[i]) for i in train_idx)
    dupe_q = sum(c - 1 for c in tr_q.values() if c > 1)

    print("\n  --- AUDIT DU DÉCOUPAGE ---")
    print(f"  {'type':22s} {'train':>8s} {'test':>7s}")
    for t in sorted(set(tr_types) | set(te_types)):
        print(f"  {t:22s} {tr_types.get(t,0):8,} {te_types.get(t,0):7,}")
    print(f"\n  catégories d'identité — train : {len(tr_cats)} / test : {len(te_cats)}")
    missing = set(tr_cats) - set(te_cats)
    if missing:
        print(f"  absentes du test : {sorted(missing)}")
    print(f"  template_group des deux côtés : {len(overlap_groups)}   <- doit être 0")
    print(f"  source_id des deux côtés      : {len(overlap_src)}   <- doit être 0")
    print(f"  questions dupliquées (train)  : {dupe_q}")

    report = {
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "by_dataset_type": {"train": dict(tr_types), "test": dict(te_types)},
        "identity_categories": {"train": dict(tr_cats), "test": dict(te_cats),
                                "missing_from_test": sorted(missing)},
        "overlapping_template_groups": sorted(overlap_groups),
        "overlapping_source_ids": sorted(overlap_src)[:50],
        "duplicate_normalized_questions_train": dupe_q,
    }
    path = os.path.join(output_dir, "split_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  rapport : {path}")

    if overlap_groups:
        raise SystemExit("Erreur : fuite de template_group entre train et test.")


def normalize_question(example):
    return re.sub(r"\s+", " ", _user_of(example)).strip().lower()


def question_key(example):
    """Clé de regroupement d'une ligne, pour un découpage train/test disjoint.

    Deux lignes partageant la même question doivent atterrir du même côté du
    découpage, sinon le jeu de test ne mesure que de la mémorisation.

    On regroupe sur la question de l'utilisateur, normalisée (casse et espaces).
    À défaut de question exploitable, on se rabat sur source_id/original_index,
    ce qui garde ensemble les scénarios issus d'un même article ou d'une même loi.
    """
    messages = example.get("messages") or []
    for msg in messages:
        role = msg.get("role", msg.get("from", ""))
        if role in ("user", "human"):
            content = (msg.get("content") or msg.get("value") or "").strip()
            if content:
                return "q:" + re.sub(r"\s+", " ", content).lower()

    for field in ("source_id", "original_index"):
        if example.get(field) is not None:
            return f"{field}:{example[field]}"

    return "unique:" + str(id(example))

def parse_args():
    parser = argparse.ArgumentParser(description="Formatage du dataset pour la distillation CoT.")
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="SuperMust/irac-thinking",
        help="Nom du dataset sur Hugging Face Hub (ou chemin local vers un fichier JSON/JSONL)."
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Nom du modèle pour extraire le tokenizer et le chat template."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed",
        help="Dossier de sauvegarde des fichiers formatés."
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.05, 
        help="Proportion de données pour le jeu de test (0.0 pour désactiver le split)."
    )
    parser.add_argument(
        "--local_file",
        type=str,
        default=None,
        help="Chemin local vers un fichier JSONL brut à formater."
    )
    parser.add_argument(
        "--legal_system_prompt_dropout",
        type=float,
        default=0.15,
        help="Fraction des exemples JURIDIQUES dont le prompt système est retiré "
             "(déterministe). Les exemples d'identité n'en ont jamais. Objectif : "
             "la spécialisation juridique ne doit pas disparaître dès qu'aucun "
             "prompt système n'est fourni à l'inférence."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Graine du dropout et du découpage train/test."
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default="strict",
        choices=["strict", "model_default"],
        help="strict (défaut) : rend exactement les messages fournis. Indispensable "
             "ici, car le gabarit d'origine de Qwen injecte « You are Qwen, created "
             "by Alibaba Cloud » dès qu'aucun message système n'est fourni — soit "
             "l'inverse de ce que les exemples d'identité doivent enseigner. "
             "model_default : gabarit d'origine du modèle (avec son injection). "
             "Le service d'inférence doit utiliser le MÊME gabarit que celui choisi ici."
    )
    parser.add_argument(
        "--identity_test_all_categories",
        action="store_true",
        help="Forcer CHAQUE catégorie d'identité dans le jeu de test. Coûteux : "
             "une catégorie à 2 familles y verse 1 famille sur 2, ce qui retire "
             "~32 %% du jeu d'identité de l'entraînement. Par défaut, le découpage "
             "reste proportionnel et la couverture par catégorie est mesurée par "
             "identity_bench.jsonl (Phase3), qui est un jeu distinct."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Création des répertoires de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Chargement du tokenizer pour appliquer le template
    print(f"Chargement du tokenizer pour '{args.model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Erreur lors du chargement du tokenizer : {e}")
        print("Tentative de chargement d'un tokenizer générique Qwen...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    # Neutralisation de l'injection du prompt système par défaut de Qwen.
    # Sans cela, toute ligne sans message système est rendue sous
    # « You are Qwen, created by Alibaba Cloud » — soit exactement l'inverse de
    # ce que les exemples d'identité doivent enseigner. Vérification explicite
    # plutôt que confiance : on rend un cas témoin et on regarde.
    if args.chat_template == "strict":
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            tokenize=False, add_generation_prompt=False)
        if "system" in probe:
            print("  Le gabarit du tokenizer injecte un prompt système par défaut "
                  "en l'absence de message système :")
            print(f"    {probe.splitlines()[1][:70]!r}")
            print("  -> remplacement par le gabarit ChatML strict "
                  "(aucune injection). Le service d'inférence doit utiliser le même.")
        tokenizer.chat_template = STRICT_CHATML_TEMPLATE
        check = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            tokenize=False, add_generation_prompt=False)
        if "system" in check:
            raise SystemExit("Erreur : le gabarit strict injecte encore un prompt système.")

    # 3. Chargement du dataset
    if args.local_file:
        print(f"Chargement du dataset local depuis {args.local_file}...")
        dataset = load_dataset("json", data_files=args.local_file, split="train")
    else:
        print(f"Chargement du dataset depuis Hugging Face Hub : '{args.dataset_name}'...")
        try:
            dataset = load_dataset(args.dataset_name, split="train")
        except Exception as e:
            print(f"Erreur lors du chargement de '{args.dataset_name}' : {e}")
            print("Tentative de chargement en tant que jeu brut sans spécifier de split...")
            dataset = load_dataset(args.dataset_name)
            if hasattr(dataset, "keys"):
                first_key = list(dataset.keys())[0]
                print(f"Utilisation du split : {first_key}")
                dataset = dataset[first_key]

    print(f"Nombre d'exemples initiaux : {len(dataset)}")

    # 4. Fonction de formatage CoT
    def format_cot_dataset(example):
        messages_bruts = example.get("messages", [])
        
        if not messages_bruts:
            for key in ["conversations", "dialogue", "chat"]:
                if key in example:
                    messages_bruts = example[key]
                    break
        
        if not messages_bruts:
            instruction = example.get("instruction", example.get("prompt", ""))
            input_context = example.get("input", "")
            output = example.get("output", example.get("response", ""))
            thinking = example.get("thinking", "")
            
            if input_context:
                user_content = f"{instruction}\n\nContexte :\n{input_context}"
            else:
                user_content = instruction
                
            messages_bruts = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output, "thinking": thinking}
            ]

        # --- traitement du prompt système, CONDITIONNEL au dataset_type ----
        # L'ancienne version réécrivait le message système de TOUTE ligne avec
        # le prompt juridique IRAC. Sur une ligne d'identité, cela produisait
        # une contradiction : « raisonne en IRAC et cite tes sources » suivi de
        # « Je suis LexiorGPT ». Pire, l'identité n'était alors apprise que
        # lorsque le prompt la soufflait déjà.
        dtype = example.get("dataset_type") or "legal_federal"
        is_identity = dtype in ("identity", "identity_control")
        is_agentic = dtype == "agentic_legal"

        if is_identity:
            keep_system = False       # jamais de prompt système sur l'identité
        elif is_agentic:
            # Conserver le contrat d'outils et le protocole Lexior fourni par
            # la trajectoire; l'ancien prompt IRAC le rendrait incohérent.
            keep_system = True
        else:
            # Dropout déterministe : une fraction des exemples juridiques perd
            # son prompt système, pour que la spécialisation juridique ne
            # s'évapore pas dès qu'aucun prompt n'est fourni à l'inférence.
            keep_system = not _dropout_hit(example, args.legal_system_prompt_dropout,
                                           args.seed)

        messages_formates = []
        for msg in messages_bruts:
            role = msg.get("role", msg.get("from", ""))
            if role in ["developer", "system"]:
                role = "system"
            elif role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant"]:
                role = "assistant"

            content = msg.get("content", msg.get("value", ""))
            thinking = msg.get("thinking", "")

            if role == "system":
                if not keep_system:
                    continue          # message système retiré
                if not is_agentic:
                    content = LEGAL_SYSTEM_PROMPT

            # Injection de la réflexion pour l'assistant
            if role == "assistant":
                if thinking and not is_identity:
                    content_final = f"<thinking>\n{thinking}\n</thinking>\n\n{content}"
                else:
                    # Une question d'identité n'appelle aucun raisonnement IRAC :
                    # tout champ thinking résiduel est ignoré.
                    content_final = content
                messages_formates.append({"role": role, "content": content_final})
            else:
                messages_formates.append({"role": role, "content": content})

        # Le prompt système juridique n'est ajouté que s'il doit l'être et que
        # la ligne n'en portait pas déjà un.
        if keep_system and not any(m["role"] == "system" for m in messages_formates):
            messages_formates.insert(0, {"role": "system", "content": LEGAL_SYSTEM_PROMPT})
                
        text = tokenizer.apply_chat_template(
            messages_formates,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Les messages sont conservés afin que le trainer construise un masque
        # assistant-only sans reparsing ambigu du texte. `text` reste présent
        # pour compatibilité et audit humain.
        return {"text": text, "messages": messages_formates}

    # 5. Clés de groupe, AVANT le mapping (qui détruit la colonne 'messages').
    # dataset.map() préserve l'ordre des lignes : ces clés restent alignées.
    group_keys = [group_key(ex) for ex in dataset]
    strata = [_stratum(ex, args.identity_test_all_categories) for ex in dataset]

    # 6. Mapping du dataset — les métadonnées SURVIVENT.
    print("Application du formatage CoT...")
    drop_columns = [c for c in dataset.column_names if c not in METADATA_COLUMNS]
    mapped_dataset = dataset.map(
        format_cot_dataset,
        remove_columns=drop_columns,
        desc="Formatting dataset to Qwen ChatML"
    )
    kept = [c for c in mapped_dataset.column_names if c != "text"]
    print(f"  colonnes conservées : {['text'] + kept}")

    # 7. Split Train / Test — par GROUPE, jamais par ligne.
    #   juridique : groupe = question normalisée
    #   identité  : groupe = template_group (toute une famille du même côté)
    # Un découpage aléatoire par ligne disperse les paraphrases des deux côtés :
    # le jeu de test ne mesurerait plus que de la mémorisation.
    if args.test_size > 0.0:
        print(f"\nDivision par groupe (test_size = {args.test_size}, graine = {args.seed})...")
        train_idx, test_idx = _grouped_split(group_keys, strata, args.test_size, args.seed)
        train_dataset = mapped_dataset.select(train_idx)
        test_dataset = mapped_dataset.select(test_idx) if test_idx else None
        _split_audit(dataset, group_keys, strata, train_idx, test_idx, args.output_dir)
    else:
        train_dataset = mapped_dataset
        test_dataset = None

    # 7. Sauvegarde locale
    train_path = os.path.join(args.output_dir, "train_dataset.jsonl")
    print(f"Sauvegarde du jeu d'entraînement dans {train_path} ({len(train_dataset)} exemples)...")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    if test_dataset:
        test_path = os.path.join(args.output_dir, "test_dataset.jsonl")
        print(f"Sauvegarde du jeu de test dans {test_path} ({len(test_dataset)} exemples)...")
        with open(test_path, "w", encoding="utf-8") as f:
            for item in test_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Formatage terminé avec succès !")

if __name__ == "__main__":
    main()
