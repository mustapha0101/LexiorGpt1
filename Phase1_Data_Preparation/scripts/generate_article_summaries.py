#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Génère deux résumés en français facile par article du corpus.

Ces textes servent UNIQUEMENT à retrouver l'article. Ils ne sont jamais
montrés ni cités : l'extrait vu par le reranker et par l'usager vient
toujours de ``document.text`` (verrouillé par deux tests).

La consigne interdit d'inventer un exemple et impose à la version courte
d'énoncer la règle de l'article plutôt que de définir le concept — les
deux causes des 40 % de résumés défectueux relevés sur la première série.

    python scripts/generate_article_summaries.py --allow-remote-calls

Environ 0,45 $ US pour 4 278 articles. Reprise gratuite : tout passe par
le cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic.config import load_config  # noqa: E402
from lexior.agentic.storage import JsonCache  # noqa: E402
from lexior.agentic.teacher_client import TeacherClient  # noqa: E402

INDEX = PHASE1 / "data" / "agentic" / "rag_index"
OUTPUT = PHASE1 / "data" / "agentic" / "article_summaries.json"

SYSTEM = (
    "Tu expliques un article de loi québécois à une personne sans formation "
    "juridique, en français facile (niveau B1).\n"
    "RÈGLES STRICTES :\n"
    "1. N'INVENTE AUCUN EXEMPLE. N'illustre pas. Reformule uniquement ce que "
    "l'article dit. Un exemple inventé est la première cause d'erreur.\n"
    "2. « courte » énonce LA RÈGLE de cet article précis, pas la définition "
    "générale du concept. Écrire « la prescription est un délai » quand "
    "l'article dit qu'elle joue même contre l'État, c'est manquer l'article.\n"
    "3. Ne dis rien que l'article ne dise pas. Si tu hésites, reste plus près "
    "du texte.\n"
    "4. Phrases courtes, vocabulaire courant, aucun terme juridique laissé "
    "sans explication.\n"
    "5. Des EXPLICATIONS, jamais des questions.\n"
    'Réponds uniquement par {"courte":"une ligne qui énonce la règle",'
    '"b1":"trois ou quatre phrases"}.'
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-remote-calls", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    documents = [json.loads(line) for line
                 in (INDEX / "documents.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        documents = documents[:args.limit]

    config = load_config()
    client = TeacherClient(
        config.teacher, allow_remote_calls=args.allow_remote_calls,
        cache=JsonCache(PHASE1 / "data" / "agentic" / "cache" / "summaries"),
        cache_extra_key="summaries-v2")

    resultats: dict[str, dict] = {}
    verrou = threading.Lock()
    started = time.monotonic()
    fautes = [0]

    def travailler(document: dict) -> None:
        cle = f'{document["code"]}:{document["article_number"]}'
        try:
            reponse = client.complete_json(
                "retrieval_reranker",
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": document["text"]}],
                temperature=0.0)
            valeur = {"courte": str(reponse.get("courte") or "").strip(),
                      "b1": str(reponse.get("b1") or "").strip()}
        except Exception:
            with verrou:
                fautes[0] += 1
            return
        with verrou:
            resultats[cle] = valeur
            fait = len(resultats)
            if fait % 250 == 0:
                ecoule = time.monotonic() - started
                reste = (len(documents) - fait) * ecoule / max(fait, 1)
                print(f"[résumés] {fait}/{len(documents)} | "
                      f"{ecoule / 60:.1f} min | reste ~{reste / 60:.1f} min",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(travailler, documents))

    vides = sum(1 for v in resultats.values()
                if not v["courte"] or not v["b1"])
    OUTPUT.write_text(json.dumps(resultats, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    usage = client.cost_report().get("total", {})
    print(f"\n[résumés] {len(resultats)}/{len(documents)} écrits dans {OUTPUT}")
    print(f"[résumés] échecs d'appel {fautes[0]} | résumés vides {vides}")
    print(f"[résumés] {(time.monotonic() - started) / 60:.1f} min | "
          f"${float(usage.get('cost_usd', 0)):.4f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
