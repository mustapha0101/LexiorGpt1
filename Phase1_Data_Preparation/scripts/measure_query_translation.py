#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Traduire la question en vocabulaire du Code aide-t-il à la retrouver ?

Diagnostic établi : sur les 20 questions dont la bonne réponse ne sort pas
en top-3, le recouvrement de vocabulaire entre la question et l'article
attendu est de 10 % contre 27 % pour les réussites, et huit questions n'ont
AUCUN mot en commun avec leur article. L'usager écrit dans le vocabulaire
de sa situation, le Code dans celui de la règle.

Mesure ciblée sur ces huit cas, plus un contrôle de non-régression sur les
32 autres — on ne répare pas huit questions en en cassant dix.

    python scripts/measure_query_translation.py --allow-remote-calls
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PHASE1 = Path(__file__).resolve().parents[1]
for candidate in (str(PHASE1 / "src"), str(PHASE1 / "tests"), str(PHASE1),
                  str(PHASE1 / "scripts")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from lexior.agentic.config import RAGConfig, load_config  # noqa: E402
from lexior.agentic.legal_rag import (  # noqa: E402
    LegalRAG, OpenAIEmbedder, _normalize_rows,
)
from lexior.agentic.storage import JsonCache  # noqa: E402
from lexior.agentic.teacher_client import TeacherClient  # noqa: E402
from sweep_retrieval import embedder_and_gold  # noqa: E402
from test_retrieval_gold import INDEX_DIR  # noqa: E402

# Les huit questions sans aucun mot commun avec leur article attendu.
ZERO_OVERLAP = [
    "ccq-resp-generale", "ccq-resp-mineur", "ccq-mise-en-demeure",
    "ccq-gain-manque", "ccq-bail-jouissance", "ccq-travail-preavis",
    "ccq-voisinage", "ccq-prescription",
]

SYSTEM = (
    "Tu traduis la situation décrite par un usager en vocabulaire du Code "
    "civil du Québec et du Code de procédure civile. Donne les termes que "
    "le LÉGISLATEUR emploierait pour régir cette situation : noms des "
    "régimes, des institutions et des obligations en jeu. Emploie le mot "
    "technique là où l'usager emploie le mot courant — « mise en demeure » "
    "pour « lettre d'avertissement », « autorité parentale » pour « mon "
    "gars », « inconvénients de voisinage » pour « les odeurs du "
    "restaurant d'à côté ». N'invente AUCUN numéro d'article et ne cite "
    "aucune loi. Réponds uniquement par l'objet JSON "
    '{"termes":"cinq à quinze mots-clés séparés par des espaces"}.'
)


def translate(teacher: TeacherClient, question: str) -> str:
    answer = teacher.complete_json(
        "retrieval_reranker",  # rôle de coût existant : la traduction sert le retrieval
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": question}],
        temperature=0.0)
    return str(answer.get("termes") or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-remote-calls", action="store_true")
    parser.add_argument("--output", default="translation.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cached, entries = embedder_and_gold()
    config = load_config()
    teacher = TeacherClient(
        config.teacher, allow_remote_calls=args.allow_remote_calls,
        cache=JsonCache(PHASE1 / "data" / "agentic" / "cache" / "translation"),
        cache_extra_key="query-translation-v1")
    live = OpenAIEmbedder(config.rag, allow_remote_calls=args.allow_remote_calls)
    rag = LegalRAG.load(RAGConfig(index_dir=str(INDEX_DIR)), cached)

    def dense_rank(text: str, code: str, article: str) -> int | None:
        indices = np.asarray(
            [i for i, d in enumerate(rag.documents) if d.code == code],
            dtype=np.int64)
        query = _normalize_rows(live.embed([text]))[0]
        dense = rag.embeddings[indices] @ query
        numbers = [rag.documents[int(indices[p])].article_number
                   for p in np.argsort(-dense)]
        return numbers.index(article) + 1 if article in numbers else None

    answerable = [e for e in entries if e["answerable"]]
    rows = []
    for entry in answerable:
        article = entry["expected_articles"][0]
        termes = translate(teacher, entry["question"])
        rows.append({
            "id": entry["id"], "cible": entry["id"] in ZERO_OVERLAP,
            "article": article, "termes": termes,
            "brute": dense_rank(entry["question"], entry["code"], article),
            "traduite": dense_rank(termes, entry["code"], article),
            "combinee": dense_rank(f"{entry['question']}\n{termes}",
                                   entry["code"], article),
        })

    def bloc(titre, subset):
        print(f"\n{titre}\n")
        print(f"{'question':26} {'art':>6} {'brute':>7} {'traduite':>9} "
              f"{'combinée':>9}")
        print("-" * 62)
        for r in subset:
            print(f"{r['id']:26} {r['article']:>6} {r['brute']:>7} "
                  f"{r['traduite']:>9} {r['combinee']:>9}")
        for variante in ("traduite", "combinee"):
            gains = [r["brute"] - r[variante] for r in subset]
            mieux = sum(1 for g in gains if g > 0)
            top10 = sum(1 for r in subset if (r[variante] or 9999) <= 10)
            base10 = sum(1 for r in subset if (r["brute"] or 9999) <= 10)
            print(f"  {variante:9} : améliore {mieux}/{len(subset)} | "
                  f"médiane {sorted(gains)[len(gains)//2]:+.0f} rangs | "
                  f"dans le top-10 : {base10} -> {top10}")

    cibles = [r for r in rows if r["cible"]]
    controle = [r for r in rows if not r["cible"]]
    bloc("═══ LES 8 CAS À RECOUVREMENT NUL ═══", cibles)
    bloc("═══ CONTRÔLE DE NON-RÉGRESSION : LES 32 AUTRES ═══", controle)

    print("\ntraductions produites pour les 8 cibles :")
    for r in cibles:
        print(f"  {r['id']:26} -> « {r['termes'][:78]} »")

    usage = teacher.cost_report().get("total", {})
    print(f"\ncoût : ${float(usage.get('cost_usd', 0)):.4f} teacher + "
          f"${float(live.cost_report()['total']['cost_usd']):.6f} embeddings")

    Path(args.output).write_text(
        json.dumps({"rows": rows, "usage": usage}, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    print(f"sortie brute : {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
