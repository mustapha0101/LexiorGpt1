#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Générateur du jeu d'identité de LexiorGPT.

Aucun appel API : les conversations proviennent des familles de gabarits de
identity_templates.py, et sont validées contre identity_policy.py.

Différences avec la version précédente (31 exemples) :
  - AUCUN message système. L'ancienne version en plaçait un dans chaque exemple,
    qui affirmait déjà l'identité : le modèle apprenait alors seulement à se
    nommer quand on la lui avait soufflée. Sans prompt système, l'identité doit
    venir des poids.
  - AUCUN champ « thinking ». Une question d'identité n'appelle pas de
    raisonnement IRAC.
  - Métadonnées explicites (dataset_type, identity_category, template_group,
    source_id, language) qui survivent au formatage.
  - Validation stricte de chaque cible assistant avant écriture.
  - Déduplication sur forme normalisée.

Usage :
    python generate_identity_data.py --count 1000 --seed 3407 \
        --output_file data/processed/generated_identity_cot.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

from identity_policy import (PRODUCT_NAME, DEVELOPER_NAME, normalize,
                             validate_assistant_text, echoes_user_model_name,
                             CATEGORIES_REQUIRING_PRODUCT_NAME,
                             CATEGORIES_REQUIRING_DEVELOPER_NAME)
from identity_templates import FAMILIES, MULTI_TURN, LEGAL_CONTROLS


def parse_args():
    parser = argparse.ArgumentParser(
        description=f"Génération du jeu d'identité {PRODUCT_NAME} (sans appel API)."
    )
    parser.add_argument("--output_file", type=str,
                        default="data/processed/generated_identity_cot.jsonl",
                        help="Fichier JSONL de sortie.")
    parser.add_argument("--count", type=int, default=1000,
                        help="Taille du vivier d'identité à produire. Le mélange final "
                             "(mix_datasets.py) en prélève un sous-ensemble.")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Graine, pour une génération déterministe.")
    parser.add_argument("--no_strict", action="store_true",
                        help="Ne pas échouer si une cible assistant viole la politique "
                             "(déconseillé : la validation est la raison d'être du script).")
    return parser.parse_args()


def _record(category, group, language, turns, idx):
    """Construit un enregistrement. AUCUN message système, AUCUN thinking."""
    messages = []
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return {
        "dataset_type": "identity",
        "identity_category": category,
        "template_group": group,
        "source_id": f"identity_{group}_{idx}",
        "language": language,
        "messages": messages,
    }


def build_pool():
    """Produit le vivier complet, avant échantillonnage. Déterministe."""
    pool = []

    # Familles à un tour : produit cartésien utilisateur x assistant.
    for fam in FAMILIES:
        n = 0
        for user in fam["users"]:
            for assistant in fam["assistants"]:
                pool.append(_record(fam["category"], fam["template_group"],
                                    fam["language"], [(user, assistant)], n))
                n += 1

    # Conversations multi-tours.
    for fam in MULTI_TURN:
        pool.append(_record(fam["category"], fam["template_group"],
                            fam["language"], fam["turns"], 0))

    # Contrôles juridiques : l'assistant répond au droit sans se présenter.
    # dataset_type distinct — ce ne sont pas des exemples d'identité, ce sont
    # les contre-exemples qui empêchent le sur-branding.
    for fam in LEGAL_CONTROLS:
        n = 0
        for user in fam["users"]:
            for assistant in fam["assistants"]:
                rec = _record(fam["category"], fam["template_group"],
                              fam["language"], [(user, assistant)], n)
                rec["dataset_type"] = "identity_control"
                pool.append(rec)
                n += 1

    return pool


def validate_record(rec):
    """Valide un enregistrement. Retourne la liste des violations."""
    problems = []

    roles = [m["role"] for m in rec["messages"]]
    if "system" in roles:
        problems.append("message système présent")
    if not roles or roles[0] != "user" or roles[-1] != "assistant":
        problems.append(f"séquence de rôles invalide : {roles}")

    cat = rec["identity_category"]
    is_control = rec["dataset_type"] == "identity_control"

    # --- INTERDICTIONS : par tour ----------------------------------------
    for i, msg in enumerate(rec["messages"]):
        if msg["role"] != "assistant":
            continue
        if "thinking" in msg:
            problems.append("champ thinking présent")

        # Les contrôles juridiques répondent au droit : ils ne doivent PAS
        # se présenter. La validation d'identité ne s'y applique pas.
        if is_control:
            if PRODUCT_NAME.lower() in msg["content"].lower():
                problems.append("un contrôle juridique s'annonce inutilement")
            continue

        problems.extend(validate_assistant_text(msg["content"]))

        # L'assistant ne doit jamais reprendre un nom de modèle venu de la
        # question. C'est licite côté utilisateur, interdit côté assistant.
        user_before = rec["messages"][i - 1]["content"] if i > 0 else ""
        for echoed in echoes_user_model_name(user_before, msg["content"]):
            problems.append(f"reprend le nom « {echoed} » utilisé par l'utilisateur")

    # --- EXIGENCES POSITIVES : par CONVERSATION --------------------------
    # Une réponse comme « IntelliWork. » ou « Oui. » est parfaitement naturelle
    # au 2e tour ; exiger le nom du produit à CHAQUE tour produirait la langue
    # de brochure que la consigne interdit. Ce qui compte est que l'identité
    # soit établie quelque part dans l'échange.
    if not is_control:
        answers = " ".join(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        if cat in CATEGORIES_REQUIRING_PRODUCT_NAME and PRODUCT_NAME.lower() not in answers.lower():
            problems.append(f"la conversation ne nomme jamais {PRODUCT_NAME}")
        if cat in CATEGORIES_REQUIRING_DEVELOPER_NAME and DEVELOPER_NAME.lower() not in answers.lower():
            problems.append(f"la conversation ne nomme jamais {DEVELOPER_NAME}")

    return problems


def conversation_key(rec):
    """Clé de déduplication : forme normalisée de la conversation entière."""
    return "|".join(f"{m['role']}:{normalize(m['content'])}" for m in rec["messages"])


def main():
    args = parse_args()
    strict = not args.no_strict
    rng = random.Random(args.seed)

    print(f"Construction du vivier d'identité {PRODUCT_NAME} / {DEVELOPER_NAME}...")
    pool = build_pool()
    print(f"  combinaisons brutes : {len(pool):,}")

    # --- validation ------------------------------------------------------
    failures = []
    valid = []
    for rec in pool:
        problems = validate_record(rec)
        if problems:
            failures.append((rec["source_id"], problems))
        else:
            valid.append(rec)
    if failures:
        print(f"\n{len(failures)} enregistrement(s) en violation de la politique :", file=sys.stderr)
        for sid, problems in failures[:10]:
            print(f"  {sid} : {problems}", file=sys.stderr)
        if strict:
            print("\nÉchec : la politique d'identité est violée. Corriger "
                  "identity_templates.py.", file=sys.stderr)
            sys.exit(1)
    print(f"  valides             : {len(valid):,}")

    # --- déduplication ---------------------------------------------------
    seen = set()
    unique = []
    for rec in valid:
        key = conversation_key(rec)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    print(f"  uniques             : {len(unique):,}  ({len(valid) - len(unique)} doublons écartés)")

    if not unique:
        print("Erreur : vivier vide.", file=sys.stderr)
        sys.exit(1)

    # --- échantillonnage à --count ---------------------------------------
    # Mélange déterministe, puis complétion cyclique si --count dépasse le
    # nombre d'uniques (en le signalant : la répétition nuit à la diversité).
    rng.shuffle(unique)
    if args.count <= len(unique):
        out = [dict(r) for r in unique[:args.count]]
    else:
        out = [dict(r) for r in unique]
        while len(out) < args.count:
            out.append(dict(unique[len(out) % len(unique)]))
        print(f"\nAvertissement : --count ({args.count}) dépasse le nombre de "
              f"conversations uniques ({len(unique)}). "
              f"{args.count - len(unique)} répétitions ajoutées.", file=sys.stderr)

    # --- source_id unique après échantillonnage --------------------------
    for i, rec in enumerate(out):
        rec["source_id"] = f"{rec['source_id']}_{i:05d}"

    # --- écriture --------------------------------------------------------
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:  # "w" : jamais d'ajout
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- rapport ---------------------------------------------------------
    cats = Counter(r["identity_category"] for r in out)
    langs = Counter(r["language"] for r in out)
    types = Counter(r["dataset_type"] for r in out)
    groups = Counter(r["template_group"] for r in out)

    print(f"\n{len(out):,} conversations écrites dans '{args.output_file}'.")
    print(f"\n  par dataset_type : {dict(types)}")
    print(f"  familles (template_group) : {len(groups)}")
    print(f"  langues : {dict(langs)}")
    print(f"\n  par catégorie ({len(cats)}) :")
    for c, n in cats.most_common():
        print(f"    {c:26s} {n:5,}")


if __name__ == "__main__":
    main()
