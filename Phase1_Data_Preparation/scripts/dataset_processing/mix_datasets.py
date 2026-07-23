#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mélangeur des trois sources du jeu d'entraînement LexiorGPT.

Remplace le « cat » du pipeline. Un cat rendait la proportion d'identité
accidentelle : elle valait ce que les fichiers contenaient, sans contrôle ni
trace. Ici la proportion est demandée, vérifiée, et consignée dans un manifeste.

Deux modes, exclusifs :

  --identity_count N   nombre ABSOLU de lignes d'identité (défaut : 500).
                       Le vivier d'identité est fixe ; la proportion baisse
                       mécaniquement à mesure que le corpus juridique grandit.
                       500 identités sur 2 000 lignes juridiques = 20 %, mais
                       les mêmes 500 sur 20 000 = 2,4 %.

  --identity_ratio R   proportion CIBLE (0.02 / 0.05 / 0.08), pour les
                       expériences comparatives décrites au cahier des charges.
                       identité = juridique x R / (1 - R)

Usage :
    python mix_datasets.py \
        --federal_file  data/processed/generated_a2aj_cot.jsonl \
        --quebec_file   data/processed/generated_ccq_cot.jsonl \
        --identity_file data/processed/generated_identity_cot.jsonl \
        --output_file   data/processed/combined_raw_cot.jsonl \
        --identity_count 500
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

from identity_policy import normalize

# Au-delà de ce seuil, répéter le vivier d'identité nuit plus qu'il n'aide :
# le modèle mémorise les formulations au lieu d'apprendre l'identité.
MAX_REPEAT_FACTOR = 2.0


def parse_args():
    p = argparse.ArgumentParser(description="Mélange fédéral + québécois + identité.")
    p.add_argument("--federal_file", type=str,
                   default="data/processed/generated_a2aj_cot.jsonl")
    p.add_argument("--quebec_file", type=str,
                   default="data/processed/generated_ccq_cot.jsonl")
    p.add_argument("--identity_file", type=str,
                   default="data/processed/generated_identity_cot.jsonl")
    p.add_argument("--agentic_file", type=str,
                   default="data/agentic/accepted/accepted.jsonl")
    p.add_argument("--include_agentic", action=argparse.BooleanOptionalAction,
                   default=os.environ.get("INCLUDE_AGENTIC_LEGAL", "true").lower() == "true")
    p.add_argument("--include_legacy_legal", action=argparse.BooleanOptionalAction,
                   default=os.environ.get("INCLUDE_LEGACY_LEGAL", "false").lower() == "true")
    p.add_argument("--include_identity_data", action=argparse.BooleanOptionalAction,
                   default=os.environ.get("INCLUDE_IDENTITY_DATA", "true").lower() == "true")
    p.add_argument("--output_file", type=str,
                   default="data/processed/combined_raw_cot.jsonl")
    p.add_argument("--manifest_file", type=str,
                   default="data/processed/mix_manifest.json")
    p.add_argument("--identity_count", type=int, default=500,
                   help="Nombre ABSOLU de lignes d'identité à inclure (défaut : 500). "
                        "Ignoré si --identity_ratio est fourni.")
    p.add_argument("--identity_ratio", type=float, default=None,
                   help="Proportion CIBLE d'identité (ex. 0.02, 0.05, 0.08). "
                        "Prend le pas sur --identity_count.")
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def load_jsonl(path, required=True):
    if not os.path.exists(path):
        if required:
            print(f"Erreur : fichier introuvable : {path}", file=sys.stderr)
            sys.exit(1)
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                print(f"  avertissement : ligne {n} illisible dans {path}, ignorée",
                      file=sys.stderr)
    return rows


def label(rows, dataset_type):
    """Étiquette les lignes juridiques ; les lignes d'identité gardent les leurs."""
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        r.setdefault("dataset_type", dataset_type)
        if r["dataset_type"] in ("legacy_legal_federal", "legacy_legal_quebec",
                                 "agentic_legal"):
            r.setdefault("identity_category", None)
            r.setdefault("template_group", None)
            r.setdefault("language", "fr")
            if not r.get("source_id"):
                # Identifiant stable, nécessaire au regroupement train/test
                # et au tirage déterministe du dropout de prompt système.
                base = r.get("original_index", r.get("source_id", i))
                r["source_id"] = f"{dataset_type}_{base}"
        out.append(r)
    return out


def user_text(rec):
    for m in rec.get("messages", []):
        if m.get("role") in ("user", "human"):
            return m.get("content", "")
    return ""


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    agentic = label(load_jsonl(args.agentic_file, required=args.include_agentic),
                    "agentic_legal") if args.include_agentic else []
    federal = (label(load_jsonl(args.federal_file), "legacy_legal_federal")
               if args.include_legacy_legal else [])
    quebec = (label(load_jsonl(args.quebec_file), "legacy_legal_quebec")
              if args.include_legacy_legal else [])
    identity_pool = (load_jsonl(args.identity_file)
                     if args.include_identity_data else [])

    n_legal = len(agentic) + len(federal) + len(quebec)
    if n_legal == 0:
        print("Erreur : aucune ligne juridique. Le mélange serait vide de droit.",
              file=sys.stderr)
        sys.exit(1)
    if args.include_identity_data and not identity_pool:
        print("Erreur : vivier d'identité vide.", file=sys.stderr)
        sys.exit(1)

    # --- combien de lignes d'identité ? ----------------------------------
    if not args.include_identity_data:
        want = 0
        mode = "identité exclue explicitement"
    elif args.identity_ratio is not None:
        if not 0 < args.identity_ratio < 1:
            print("Erreur : --identity_ratio doit être strictement entre 0 et 1.",
                  file=sys.stderr)
            sys.exit(1)
        want = int(round(n_legal * args.identity_ratio / (1 - args.identity_ratio)))
        mode = f"ratio cible {args.identity_ratio:.0%}"
    else:
        want = args.identity_count
        mode = f"compte absolu {args.identity_count}"
    want = max(want, 0)

    # --- tirage déterministe, avec sur-échantillonnage si nécessaire ------
    pool = list(identity_pool)
    rng.shuffle(pool)
    repeats = 0
    if want == 0:
        identity = []
    elif want <= len(pool):
        identity = [dict(r) for r in pool[:want]]
    else:
        identity = [dict(r) for r in pool]
        while len(identity) < want:
            src = dict(pool[len(identity) % len(pool)])
            src["source_id"] = f"{src['source_id']}_rep{len(identity)//len(pool)}"
            identity.append(src)
        repeats = want - len(pool)
        factor = want / len(pool)
        print(f"\nAvertissement : {want} lignes d'identité demandées pour un vivier de "
              f"{len(pool)} uniques — {repeats} répétitions (facteur x{factor:.2f}).",
              file=sys.stderr)
        if factor > MAX_REPEAT_FACTOR:
            print(f"Erreur : facteur de répétition {factor:.2f} > {MAX_REPEAT_FACTOR}. "
                  f"Agrandir le vivier (generate_identity_data.py --count) plutôt que "
                  f"de répéter : au-delà, le modèle mémorise les formulations.",
                  file=sys.stderr)
            sys.exit(1)

    # --- assemblage et mélange déterministe ------------------------------
    combined = agentic + federal + quebec + identity
    rng.shuffle(combined)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- manifeste -------------------------------------------------------
    n_total = len(combined)
    n_ident = len(identity)
    actual_ratio = n_ident / n_total if n_total else 0.0
    id_cats = Counter(r.get("identity_category") for r in identity
                      if r.get("identity_category"))
    dupe_users = len(combined) - len({normalize(user_text(r)) for r in combined})

    manifest = {
        "seed": args.seed,
        "mode": mode,
        "inputs": {
            "federal_file": args.federal_file,
            "quebec_file": args.quebec_file,
            "agentic_file": args.agentic_file,
            "identity_file": args.identity_file,
        },
        "output_file": args.output_file,
        "source_counts": {
            "agentic_legal": len(agentic),
            "legacy_legal_federal": len(federal),
            "legacy_legal_quebec": len(quebec),
            "identity_pool_available": len(identity_pool),
        },
        "final_counts": {
            "agentic_legal": len(agentic),
            "legacy_legal_federal": len(federal),
            "legacy_legal_quebec": len(quebec),
            "identity": sum(1 for r in identity if r.get("dataset_type") == "identity"),
            "identity_control": sum(1 for r in identity
                                    if r.get("dataset_type") == "identity_control"),
            "total": n_total,
        },
        "percentages": {
            "agentic_legal": round(100 * len(agentic) / n_total, 2),
            "legacy_legal_federal": round(100 * len(federal) / n_total, 2),
            "legacy_legal_quebec": round(100 * len(quebec) / n_total, 2),
            "identity_all": round(100 * n_ident / n_total, 2),
        },
        "identity_requested": want,
        "identity_actual": n_ident,
        "identity_ratio_actual": round(actual_ratio, 4),
        "identity_repeats": repeats,
        "duplicate_user_questions": dupe_users,
        "identity_category_distribution": dict(id_cats.most_common()),
    }
    os.makedirs(os.path.dirname(args.manifest_file) or ".", exist_ok=True)
    with open(args.manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # --- rapport ---------------------------------------------------------
    print(f"\nMélange écrit dans '{args.output_file}' ({mode}, graine {args.seed})")
    print(f"  {'source':22s} {'lignes':>8s} {'%':>7s}")
    print(f"  {'-'*40}")
    for k, v in (("agentic_legal", len(agentic)),
                 ("legacy_legal_federal", len(federal)),
                 ("legacy_legal_quebec", len(quebec)),
                 ("identité (toutes)", n_ident)):
        print(f"  {k:22s} {v:8,} {100*v/n_total:6.2f}%")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':22s} {n_total:8,}")
    print(f"\n  identité : {n_ident} demandée(s) {want} — proportion réelle "
          f"{actual_ratio:.2%}, {repeats} répétition(s)")
    print(f"  questions utilisateur dupliquées : {dupe_users}")
    print(f"  manifeste : {args.manifest_file}")

    # --- garde-fous ------------------------------------------------------
    if args.include_identity_data and n_ident == 0:
        print("\nErreur : aucune ligne d'identité dans le mélange.", file=sys.stderr)
        sys.exit(1)
    if (args.include_identity_data and args.identity_ratio is not None
            and abs(actual_ratio - args.identity_ratio) > 0.02):
        print(f"\nErreur : proportion réelle {actual_ratio:.2%} trop éloignée de la "
              f"cible {args.identity_ratio:.2%}.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
