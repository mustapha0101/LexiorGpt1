# -*- coding: utf-8 -*-
"""Pré-interrogation MCP pour constituer des banques d'ancres fédérales."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class CaseAnchor:
    citation_fr: str
    name_fr: str
    dataset: str
    topic_hint: str


@dataclass(frozen=True)
class LawAnchor:
    citation_fr: str
    name_fr: str


FEDERAL_COURTS = frozenset({"SCC", "FC", "FCA"})

_CASE_QUERIES = (
    "faillite insolvabilité",
    "immigration réfugié",
    "brevet propriété intellectuelle",
    "marque de commerce",
    "banque institution financière",
    "environnement pollution fédéral",
    "droit maritime transport",
    "impôt revenu fiscal",
    "travail emploi fédéral",
    "télécommunications radiodiffusion",
    "concurrence antitrust",
    "douane importation exportation",
)

_LAW_QUERIES = (
    "Loi sur les banques",
    "Loi sur la faillite",
    "Loi sur les brevets",
    "Loi sur les marques de commerce",
    "Loi sur l'immigration",
    "Loi sur le divorce",
    "Loi sur les pêches",
    "Loi sur la concurrence",
    "Loi sur les transports",
    "Loi sur les douanes",
    "Loi de l'impôt sur le revenu",
    "Loi sur les langues officielles",
)

_TOPIC_FROM_QUERY = {
    "faillite insolvabilité": "faillite",
    "immigration réfugié": "immigration",
    "brevet propriété intellectuelle": "brevet",
    "marque de commerce": "marques de commerce",
    "banque institution financière": "banque",
    "environnement pollution fédéral": "environnement",
    "droit maritime transport": "droit maritime",
    "impôt revenu fiscal": "fiscalité",
    "travail emploi fédéral": "droit du travail",
    "télécommunications radiodiffusion": "télécommunications",
    "concurrence antitrust": "concurrence",
    "douane importation exportation": "douanes",
}


def _extract_cases(results: list[dict[str, Any]], topic: str) -> list[CaseAnchor]:
    anchors = []
    for r in results:
        citation_fr = r.get("citation_fr")
        name_fr = r.get("name_fr")
        dataset = r.get("dataset", "")
        if not citation_fr or not name_fr:
            continue
        if dataset not in FEDERAL_COURTS:
            continue
        anchors.append(CaseAnchor(
            citation_fr=citation_fr,
            name_fr=name_fr,
            dataset=dataset,
            topic_hint=topic,
        ))
    return anchors


def _extract_laws(results: list[dict[str, Any]]) -> list[LawAnchor]:
    anchors = []
    for r in results:
        citation_fr = r.get("citation_fr") or r.get("citation_en", "")
        name_fr = r.get("name_fr") or r.get("name_en", "")
        if not citation_fr or not name_fr:
            continue
        anchors.append(LawAnchor(citation_fr=citation_fr, name_fr=name_fr))
    return anchors


@dataclass
class AnchorBank:
    cases: list[CaseAnchor] = field(default_factory=list)
    laws: list[LawAnchor] = field(default_factory=list)
    _used_cases: set[str] = field(default_factory=set)
    _used_laws: set[str] = field(default_factory=set)

    def sample_case(self, rng: random.Random) -> Optional[CaseAnchor]:
        available = [a for a in self.cases if a.citation_fr not in self._used_cases]
        if not available:
            self._used_cases.clear()
            available = self.cases
        if not available:
            return None
        pick = rng.choice(available)
        self._used_cases.add(pick.citation_fr)
        return pick

    def sample_law(self, rng: random.Random) -> Optional[LawAnchor]:
        available = [a for a in self.laws if a.citation_fr not in self._used_laws]
        if not available:
            self._used_laws.clear()
            available = self.laws
        if not available:
            return None
        pick = rng.choice(available)
        self._used_laws.add(pick.citation_fr)
        return pick


async def _query_mcp(transport, query: str, doc_type: str, size: int = 5) -> list[dict]:
    try:
        params: dict = {
            "query": query,
            "doc_type": doc_type,
            "search_language": "fr",
            "size": size,
        }
        if doc_type == "cases":
            params["dataset"] = "SCC,FC,FCA"
        else:
            params["search_type"] = "name"
        raw = await transport.call(
            "a2aj", "a2aj", "search_legal_documents", params,
        )
        from .mcp_executor import _text_content
        import json as _json
        text = _text_content(raw)
        parsed = _json.loads(text)
        if isinstance(parsed, dict):
            return parsed.get("results", [])
        if isinstance(parsed, list):
            return parsed
    except Exception as exc:
        import sys
        print(f"[anchor_bank] échec requête '{query}' ({doc_type}): {exc}",
              file=sys.stderr, flush=True)
    return []


async def _build_bank(transport) -> AnchorBank:
    bank = AnchorBank()
    case_tasks = [_query_mcp(transport, q, "cases", 5) for q in _CASE_QUERIES]
    law_tasks = [_query_mcp(transport, q, "laws", 3) for q in _LAW_QUERIES]
    case_results = await asyncio.gather(*case_tasks)
    law_results = await asyncio.gather(*law_tasks)
    seen_cases: set[str] = set()
    for query, results in zip(_CASE_QUERIES, case_results):
        topic = _TOPIC_FROM_QUERY.get(query, query)
        for anchor in _extract_cases(results, topic):
            if anchor.citation_fr not in seen_cases:
                seen_cases.add(anchor.citation_fr)
                bank.cases.append(anchor)
    seen_laws: set[str] = set()
    for results in law_results:
        for anchor in _extract_laws(results):
            if anchor.citation_fr not in seen_laws:
                seen_laws.add(anchor.citation_fr)
                bank.laws.append(anchor)
    return bank


def build_anchor_bank(transport) -> AnchorBank:
    return asyncio.run(_build_bank(transport))
