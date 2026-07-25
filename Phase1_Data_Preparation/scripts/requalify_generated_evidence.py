#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Sort du dataset les trajectoires fondées sur du texte généré.

Certains serveurs MCP renvoient une synthèse rédigée par un modèle au lieu
du texte source (voir ``contains_generated_summary``). Le contrôle de
grounding validait jusqu'ici les citations de la réponse finale contre ce
texte-là. Les trajectoires déjà acceptées dans ce cas ne sont pas valides.

Une trajectoire est requalifiée quand un numéro d'article cité dans la
réponse finale n'est attesté QUE par une synthèse générée.

Les URLs et les citations de décisions échappent à ce contrôle : ce sont
des faits structurels, valides même quand le serveur les enrobe de prose
générée. Les exclure rendrait inutilisable l'outil de jurisprudence, dont
c'est justement la sortie. Le critère suit exactement celui de
``validators.validate_trajectory``.

Par défaut le script ne modifie rien : il compte et il liste.

    python scripts/requalify_generated_evidence.py            # constat
    python scripts/requalify_generated_evidence.py --apply    # déplacement

``--apply`` copie chaque fichier modifié en ``.bak`` avant réécriture :
``data/`` n'est pas versionné, il n'y a pas d'autre filet.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic.response_verifier import (  # noqa: E402
    contains_generated_summary,
)
from lexior.agentic.validators import (  # noqa: E402
    ARTICLE_CITATION_RE,
    URL_RE,
)

DATA = PHASE1 / "data" / "agentic"
NON_CITABLE_TOOLS = {"semantic_search_ccq", "semantic_search_cpc"}
REASON = ("preuve fondée sur une synthèse générée par un modèle "
          "(requalification rétroactive)")


def _final_answer(record: dict) -> str:
    for message in reversed(record.get("messages") or []):
        if message.get("role") == "assistant":
            return message.get("content") or ""
    return ""


def contamination(record: dict) -> list[str]:
    """Éléments cités introuvables ailleurs que dans du texte généré."""
    trace = record.get("tool_trace") or []
    generated, source = [], []
    for observation in trace:
        if observation.get("tool_name") in NON_CITABLE_TOOLS:
            continue
        text = observation.get("normalized_response") or ""
        (generated if contains_generated_summary(text) else source).append(
            observation)
    if not generated:
        return []

    final = _final_answer(record)
    problems: list[str] = []

    source_text = "\n".join(
        o.get("normalized_response") or "" for o in source).casefold()
    generated_text = "\n".join(
        o.get("normalized_response") or "" for o in generated).casefold()
    for article in set(ARTICLE_CITATION_RE.findall(final)):
        needle = f"article {article}".casefold()
        if needle in generated_text and needle not in source_text:
            problems.append(f"article {article}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="déplacer réellement les trajectoires")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    accepted_dir = DATA / "accepted"
    rejected_dir = DATA / "rejected"
    if not accepted_dir.is_dir():
        print(f"[requalif] répertoire absent : {accepted_dir}", file=sys.stderr)
        return 1

    total = 0
    moved = 0
    touched_files = 0
    for path in sorted(accepted_dir.glob("*.jsonl")):
        keep: list[str] = []
        drop: list[tuple[dict, list[str]]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                keep.append(line)
                continue
            problems = contamination(record)
            if problems:
                drop.append((record, problems))
            else:
                keep.append(line)
        if not drop:
            continue
        touched_files += 1
        moved += len(drop)
        print(f"\n{path.name} : {len(drop)} trajectoire(s) sur "
              f"{len(drop) + len(keep)}")
        for record, problems in drop:
            print(f"  - {record.get('scenario_id', '?')} "
                  f"({record.get('request_type', '?')}) : "
                  f"{', '.join(problems)}")
        if not args.apply:
            continue

        shutil.copy2(path, path.with_suffix(".jsonl.bak"))
        path.write_text("\n".join(keep) + ("\n" if keep else ""),
                        encoding="utf-8")
        rejected_dir.mkdir(parents=True, exist_ok=True)
        target = rejected_dir / path.name
        with target.open("a", encoding="utf-8") as handle:
            for record, problems in drop:
                handle.write(json.dumps({
                    "scenario_id": record.get("scenario_id"),
                    "request_type": record.get("request_type"),
                    "stage": "requalification",
                    "reasons": [f"{REASON} : {', '.join(problems)}"],
                    "timestamp": record.get("generation_metadata", {}).get(
                        "generated_at", ""),
                    "trajectory": record,
                }, ensure_ascii=False) + "\n")

    verb = "déplacées" if args.apply else "à déplacer"
    print(f"\n[requalif] {total} trajectoires acceptées examinées | "
          f"{moved} {verb} | {touched_files} fichier(s) concerné(s)")
    if moved and not args.apply:
        print("[requalif] relancer avec --apply pour appliquer "
              "(sauvegarde .bak de chaque fichier modifié)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
