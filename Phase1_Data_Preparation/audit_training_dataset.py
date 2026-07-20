#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audit d'un jeu d'entraînement LexiorGPT, brut ou formaté.

Fonctionne sur les deux formes :
  - BRUT    : lignes avec une liste `messages` (sortie de mix_datasets.py)
  - FORMATÉ : lignes avec un champ `text` en ChatML (sortie de dataset_formatter.py)

Sort en code non nul si une violation CRITIQUE de la politique d'identité est
détectée — le pipeline doit s'arrêter plutôt que téléverser un jeu fautif.

    python audit_training_dataset.py --files data/processed/train_dataset.jsonl \
                                             data/processed/test_dataset.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

from identity_policy import (PRODUCT_NAME, normalize, find_forbidden_terms,
                             find_forbidden_claims)

# Marqueurs ChatML de Qwen — utilisés pour relire une ligne déjà formatée.
_CHATML_TURN = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)
_THINKING = re.compile(r"<thinking>.*?</thinking>", re.S)


def parse_args():
    p = argparse.ArgumentParser(description="Audit d'un jeu d'entraînement LexiorGPT.")
    p.add_argument("--files", nargs="+", required=True,
                   help="Fichiers JSONL à auditer (bruts ou formatés).")
    p.add_argument("--expect_identity_ratio", type=float, default=None,
                   help="Proportion d'identité attendue. Échoue si l'écart dépasse "
                        "--ratio_tolerance.")
    p.add_argument("--expect_identity_count", type=int, default=None,
                   help="Nombre absolu de lignes d'identité attendu (toutes splits "
                        "confondues). Échoue en cas d'écart.")
    p.add_argument("--ratio_tolerance", type=float, default=0.02)
    p.add_argument("--require_identity_in_test", action="store_true",
                   help="Échouer si aucune ligne d'identité n'est présente dans le "
                        "fichier dont le nom contient « test ».")
    p.add_argument("--report_file", type=str, default=None)
    return p.parse_args()


def parse_row(row):
    """Normalise une ligne (brute ou formatée) en (turns, meta).

    turns : [(role, content), ...]
    """
    meta = {k: row.get(k) for k in
            ("dataset_type", "identity_category", "template_group", "source_id", "language")}

    if row.get("messages"):
        turns = [(m.get("role", ""), m.get("content", "") or "")
                 for m in row["messages"]]
        # Le champ thinking du format brut est fusionné au formatage : on le
        # reconstitue pour que l'audit voie ce que verra l'entraînement.
        for i, m in enumerate(row["messages"]):
            if m.get("thinking"):
                turns[i] = (turns[i][0], f"<thinking>\n{m['thinking']}\n</thinking>\n\n{turns[i][1]}")
        return turns, meta

    text = row.get("text") or ""
    if text:
        return [(r, c.strip()) for r, c in _CHATML_TURN.findall(text)], meta

    return [], meta


def approx_tokens(text):
    """Approximation sans tokenizer : ~3,8 caractères par token en français.

    Suffisant pour comparer des proportions ; ce n'est pas un compte exact.
    """
    return max(1, int(len(text) / 3.8))


def main():
    args = parse_args()

    rows = []
    for path in args.files:
        if not os.path.exists(path):
            print(f"Erreur : fichier introuvable : {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append((os.path.basename(path), json.loads(line)))

    if not rows:
        print("Erreur : aucun enregistrement.", file=sys.stderr)
        sys.exit(1)

    n = len(rows)
    by_type = Counter()
    by_cat = Counter()
    tokens_by_type = Counter()
    with_system = Counter()
    with_thinking = Counter()
    user_norm = Counter()
    asst_norm = Counter()
    malformed = []
    missing_meta = []
    violations = []          # critiques
    identity_by_file = Counter()

    for fname, row in rows:
        turns, meta = parse_row(row)
        dtype = meta.get("dataset_type") or "inconnu"
        by_type[dtype] += 1
        if meta.get("identity_category"):
            by_cat[meta["identity_category"]] += 1
        if dtype in ("identity", "identity_control"):
            identity_by_file[fname] += 1

        if not turns:
            malformed.append((meta.get("source_id"), "conversation vide"))
            continue
        roles = [r for r, _ in turns]
        if "assistant" not in roles:
            malformed.append((meta.get("source_id"), f"aucun tour assistant : {roles}"))

        if dtype == "inconnu" or not meta.get("source_id"):
            missing_meta.append(meta.get("source_id") or "(sans source_id)")

        full = "\n".join(c for _, c in turns)
        tokens_by_type[dtype] += approx_tokens(full)

        if "system" in roles:
            with_system[dtype] += 1
            # CRITIQUE : un exemple d'identité ne doit jamais porter de prompt système.
            if dtype in ("identity", "identity_control"):
                violations.append((meta.get("source_id"),
                                   "message système sur un exemple d'identité"))

        for role, content in turns:
            if role != "assistant":
                continue
            if _THINKING.search(content):
                with_thinking[dtype] += 1
                if dtype in ("identity", "identity_control"):
                    violations.append((meta.get("source_id"),
                                       "bloc <thinking> sur un exemple d'identité"))
            if dtype == "identity":
                for t in find_forbidden_terms(content):
                    violations.append((meta.get("source_id"),
                                       f"terme technique interdit : « {t} »"))
                for c in find_forbidden_claims(content):
                    violations.append((meta.get("source_id"), f"affirmation interdite : {c}"))
            asst_norm[normalize(content)] += 1

        for role, content in turns:
            if role in ("user", "human"):
                user_norm[normalize(content)] += 1
                break

    n_ident = sum(v for k, v in by_type.items() if k in ("identity", "identity_control"))
    ratio = n_ident / n
    tok_total = sum(tokens_by_type.values())

    # --- rapport ---------------------------------------------------------
    print("=" * 66)
    print(f"AUDIT — {n:,} enregistrements sur {len(args.files)} fichier(s)")
    print("=" * 66)
    print(f"\n{'type':22s} {'lignes':>8s} {'% lignes':>9s} {'~tokens':>10s} {'% tokens':>9s}")
    print("-" * 62)
    for t in sorted(by_type):
        print(f"{t:22s} {by_type[t]:8,} {100*by_type[t]/n:8.2f}% "
              f"{tokens_by_type[t]:10,} {100*tokens_by_type[t]/tok_total:8.2f}%")
    print("-" * 62)
    print(f"{'TOTAL':22s} {n:8,} {'100.00%':>9s} {tok_total:10,} {'100.00%':>9s}")

    print("\n  Le % de LIGNES et le % de TOKENS diffèrent nécessairement : un exemple")
    print("  juridique (raisonnement IRAC + citation) est bien plus long qu'une")
    print("  réponse d'identité. Une part de 20 % des lignes peut ne peser que")
    print("  quelques pour cent des tokens — c'est le token qui porte le gradient.")

    print(f"\nIDENTITÉ : {n_ident:,} lignes ({ratio:.2%} des lignes, "
          f"{100*sum(tokens_by_type[t] for t in ('identity','identity_control'))/tok_total:.2f}% des tokens)")
    if by_cat:
        print(f"\n  catégories ({len(by_cat)}) :")
        for c, v in by_cat.most_common():
            print(f"    {c:26s} {v:5,}")

    print(f"\nSTRUCTURE")
    print(f"  lignes avec message système : {dict(with_system)}")
    print(f"  lignes avec <thinking>      : {dict(with_thinking)}")
    print(f"  questions dupliquées        : {sum(c-1 for c in user_norm.values() if c > 1):,}")
    print(f"  réponses dupliquées         : {sum(c-1 for c in asst_norm.values() if c > 1):,}")
    print(f"  conversations malformées    : {len(malformed)}")
    print(f"  métadonnées manquantes      : {len(missing_meta)}")

    # --- verdict ---------------------------------------------------------
    critical = []
    if violations:
        critical.append(f"{len(violations)} violation(s) de la politique d'identité")
    if n_ident == 0:
        critical.append("aucune ligne d'identité")
    if args.expect_identity_count is not None and n_ident != args.expect_identity_count:
        critical.append(f"identité : {n_ident} lignes, {args.expect_identity_count} attendues")
    if args.expect_identity_ratio is not None and \
            abs(ratio - args.expect_identity_ratio) > args.ratio_tolerance:
        critical.append(f"proportion d'identité {ratio:.2%} hors tolérance "
                        f"(cible {args.expect_identity_ratio:.2%} ± {args.ratio_tolerance:.2%})")
    if args.require_identity_in_test:
        test_files = [f for f in identity_by_file if "test" in f.lower()]
        if not test_files or all(identity_by_file[f] == 0 for f in test_files):
            critical.append("aucune ligne d'identité dans le jeu de test")

    if violations:
        print(f"\nVIOLATIONS CRITIQUES ({len(violations)}) — 10 premières :")
        for sid, why in violations[:10]:
            print(f"  {sid} : {why}")
    if malformed:
        print(f"\nMalformées ({len(malformed)}) — 5 premières : {malformed[:5]}")

    report = {
        "total": n, "by_dataset_type": dict(by_type),
        "identity_categories": dict(by_cat),
        "approx_tokens_by_type": dict(tokens_by_type),
        "identity_rows": n_ident, "identity_row_ratio": round(ratio, 4),
        "rows_with_system": dict(with_system), "rows_with_thinking": dict(with_thinking),
        "duplicate_user_questions": sum(c-1 for c in user_norm.values() if c > 1),
        "duplicate_assistant_answers": sum(c-1 for c in asst_norm.values() if c > 1),
        "malformed": len(malformed), "missing_metadata": len(missing_meta),
        "policy_violations": [{"source_id": s, "problem": w} for s, w in violations],
        "critical": critical,
    }
    if args.report_file:
        os.makedirs(os.path.dirname(args.report_file) or ".", exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nRapport : {args.report_file}")

    print("\n" + "=" * 66)
    if critical:
        print("ÉCHEC — violations critiques :")
        for c in critical:
            print(f"  - {c}")
        print("=" * 66)
        sys.exit(1)
    print("AUDIT RÉUSSI — aucune violation critique.")
    print("=" * 66)


if __name__ == "__main__":
    main()
