# -*- coding: utf-8 -*-

"""
Taxonomie v2.0 : 12 types de demandes juridiques.

Les types décrivent l'intention de l'utilisateur, pas le routage d'outils.
Les modes de panne et la clarification sont des métadonnées orthogonales,
pas des types à part entière.

Distributions configurables dans configs/agentic_generation.yaml.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .schemas import (
    ConditionalTool, ExpectedRoute, ExpectedRouteStep, RoutePolicy,
)


def _route(*steps, clarification: bool = False, no_tool: bool = False) -> ExpectedRoute:
    parsed = []
    for s in steps:
        if isinstance(s, str):
            parsed.append(ExpectedRouteStep(tool=s))
        elif isinstance(s, tuple):
            if len(s) == 2:
                tool, optional = s
                parsed.append(ExpectedRouteStep(tool=tool, optional=optional))
            elif len(s) == 3:
                tool, optional, condition = s
                parsed.append(ExpectedRouteStep(
                    tool=tool, optional=optional, condition=condition))
    return ExpectedRoute(steps=parsed, requires_clarification=clarification,
                         no_tool=no_tool)


def _policy(*caps, conditional=None, forbidden=None, preferred=None,
            clarification=False, no_tool=False, expected_route=None):
    return RoutePolicy(
        required_capabilities=list(caps),
        conditional_tools=[
            ConditionalTool(tool=t, condition=c)
            for t, c in (conditional or [])
        ],
        forbidden_tools=list(forbidden or []),
        preferred_initial_tools=list(preferred or []),
        requires_clarification=clarification,
        no_tool=no_tool,
        expected_route=expected_route or ExpectedRoute(),
    )


# ---------------------------------------------------------------------------
# Outils interdits par juridiction
# ---------------------------------------------------------------------------

QUEBEC_ONLY_TOOLS = frozenset({
    "semantic_search_ccq", "semantic_search_cpc",
    "get_ccq_articles", "get_cpc_articles",
    "search_ccq_keywords", "search_cpc_keywords",
    "search_quebec_jurisprudence",
    "search_quebec_regulations", "get_quebec_regulation",
    "get_quebec_legal_info",
})

FEDERAL_ONLY_TOOLS = frozenset({
    "search_legal_documents", "fetch_document",
})


@dataclass
class RequestTypeSpec:
    name: str
    description: str
    expected_route: ExpectedRoute
    route_policy: RoutePolicy = field(default_factory=RoutePolicy)
    default_jurisdiction_status: str = "supported_quebec"
    default_source_intents: list = field(default_factory=list)
    legal_domain: str = "droit civil québécois"
    default_weight: float = 1.0
    # Faits sans lesquels une réponse ne peut pas être fondée. Consommés par
    # le nœud analyze_facts, qui les inscrit dans
    # ``missing_facts_before_search``; validate_plan force alors une
    # clarification, indépendamment de ce que propose le planner.
    required_facts: list = field(default_factory=list)
    clarification_weights: dict = field(default_factory=lambda: {
        "none": 0.55, "before_search": 0.20, "after_initial_research": 0.25,
    })


# ---------------------------------------------------------------------------
# Les 12 types de demandes (spec section 5)
# ---------------------------------------------------------------------------

REQUEST_TYPES: dict[str, RequestTypeSpec] = {rt.name: rt for rt in [
    RequestTypeSpec(
        name="case_analysis",
        required_facts=["juridiction applicable"],
        description="Cas concret : appliquer la loi à des faits particuliers, "
                    "identifier la règle, chercher la jurisprudence pertinente.",
        expected_route=_route(
            ("semantic_search_ccq", True, "if Quebec civil law and article unknown"),
            ("get_ccq_articles", False, "retrieve official text of identified article"),
            ("search_quebec_jurisprudence", True,
             "only if user requests application examples or jurisprudence needed"),
        ),
        route_policy=_policy(
            "official_text_retrieval", "case_law_application",
            conditional=[
                ("search_quebec_jurisprudence",
                 "only if user requests application examples or jurisprudence needed"),
                ("semantic_search_ccq", "article number unknown"),
            ],
            preferred=["semantic_search_ccq", "get_ccq_articles"],
        ),
        default_source_intents=["legislation", "jurisprudence"],
        default_weight=0.25,
        clarification_weights={
            "none": 0.40, "before_search": 0.30, "after_initial_research": 0.30,
        },
    ),
    RequestTypeSpec(
        name="procedure_guidance",
        required_facts=["juridiction applicable"],
        description="Question de procédure : appel, rétractation, injonction, "
                    "mise en demeure, signification, délais.",
        expected_route=_route(
            ("semantic_search_cpc", True, "if procedural provision unknown"),
            ("get_cpc_articles", False, "retrieve procedural text"),
            ("search_quebec_jurisprudence", True,
             "only if procedural test needs jurisprudential clarification"),
        ),
        route_policy=_policy(
            "official_text_retrieval",
            conditional=[
                ("search_quebec_jurisprudence",
                 "only if procedural test needs jurisprudential clarification"),
            ],
            preferred=["semantic_search_cpc", "get_cpc_articles"],
        ),
        legal_domain="procédure civile québécoise",
        default_source_intents=["procedural_rule"],
        default_weight=0.18,
        clarification_weights={
            "none": 0.25, "before_search": 0.35, "after_initial_research": 0.40,
        },
    ),
    RequestTypeSpec(
        name="topic_research",
        description="Recherche thématique : sujet CCQ/CPC sans cas concret, "
                    "explication générale d'un domaine juridique.",
        expected_route=_route(
            "semantic_search_ccq",
            "get_ccq_articles",
            ("search_quebec_jurisprudence", True,
             "only if facts involve application to specific situation"),
        ),
        route_policy=_policy(
            "article_discovery", "official_text_retrieval",
            conditional=[
                ("search_quebec_jurisprudence",
                 "facts involve application to specific situation"),
            ],
            preferred=["semantic_search_ccq"],
        ),
        default_source_intents=["legislation"],
        default_weight=0.12,
        clarification_weights={
            "none": 0.80, "before_search": 0.10, "after_initial_research": 0.10,
        },
    ),
    RequestTypeSpec(
        name="exact_text_retrieval",
        description="Texte officiel d'un article dont le numéro est donné : "
                    "récupération exacte, mot pour mot, sans analyse.",
        expected_route=_route("get_ccq_articles"),
        route_policy=_policy(
            "official_text_retrieval",
            preferred=["get_ccq_articles", "get_cpc_articles"],
            forbidden=["search_legal_documents", "fetch_document",
                        "search_quebec_jurisprudence",
                        "search_ccq_keywords", "search_cpc_keywords"],
        ),
        default_source_intents=["legislation"],
        default_weight=0.07,
        clarification_weights={
            "none": 1.0, "before_search": 0.0, "after_initial_research": 0.0,
        },
    ),
    RequestTypeSpec(
        name="article_explanation",
        description="Explication d'un article précis : récupérer le texte officiel, "
                    "puis expliquer avec la jurisprudence si pertinent.",
        expected_route=_route(
            "get_ccq_articles",
            ("search_quebec_jurisprudence", True,
             "article contains open-ended notion or facts warrant it"),
        ),
        route_policy=_policy(
            "official_text_retrieval",
            conditional=[
                ("search_quebec_jurisprudence",
                 "article contains open-ended notion or facts warrant it"),
            ],
            preferred=["get_ccq_articles"],
            forbidden=["search_legal_documents", "fetch_document"],
        ),
        default_source_intents=["legislation", "jurisprudence"],
        default_weight=0.06,
        clarification_weights={
            "none": 0.90, "before_search": 0.05, "after_initial_research": 0.05,
        },
    ),
    RequestTypeSpec(
        name="case_law_research",
        description="L'utilisateur demande explicitement des décisions judiciaires.",
        expected_route=_route(
            ("get_ccq_articles", True, "article number known or discoverable"),
            "search_quebec_jurisprudence",
        ),
        route_policy=_policy(
            "case_law_application",
            conditional=[
                ("get_ccq_articles", "article number known or discoverable"),
            ],
            preferred=["search_quebec_jurisprudence"],
        ),
        default_source_intents=["jurisprudence"],
        default_weight=0.10,
        clarification_weights={
            "none": 0.50, "before_search": 0.20, "after_initial_research": 0.30,
        },
    ),
    RequestTypeSpec(
        name="law_or_regulation_identification",
        description="Identifier une loi, un règlement ou un texte officiel "
                    "dont l'utilisateur connaît le sujet mais pas le titre exact.",
        expected_route=_route(
            "search_quebec_regulations",
            "get_quebec_regulation",
        ),
        route_policy=_policy(
            "regulation_discovery", "regulation_retrieval",
            preferred=["search_quebec_regulations"],
        ),
        legal_domain="droit réglementaire québécois",
        default_source_intents=["regulation"],
        default_weight=0.07,
        clarification_weights={
            "none": 0.70, "before_search": 0.15, "after_initial_research": 0.15,
        },
    ),
    RequestTypeSpec(
        name="legislative_status_verification",
        description="Vérifier l'entrée en vigueur, les modifications ou l'abrogation "
                    "d'une disposition.",
        expected_route=_route("get_quebec_legal_info"),
        route_policy=_policy(
            "legislative_metadata",
            preferred=["get_quebec_legal_info"],
            forbidden=["search_legal_documents", "fetch_document",
                        "search_quebec_jurisprudence"],
        ),
        default_source_intents=["legislative_metadata"],
        default_weight=0.05,
        clarification_weights={
            "none": 0.80, "before_search": 0.10, "after_initial_research": 0.10,
        },
    ),
    RequestTypeSpec(
        name="document_analysis",
        required_facts=[
            "nature du document",
        ],
        description="Analyser un document fourni (jugement, bail, contrat, "
                    "mise en demeure, constat d'infraction).",
        expected_route=_route(
            ("semantic_search_ccq", True, "if relevant legislation unknown"),
            ("get_ccq_articles", True, "if article identified in document"),
        ),
        route_policy=_policy(
            conditional=[
                ("semantic_search_ccq", "relevant legislation unknown"),
                ("get_ccq_articles", "article identified in document"),
            ],
        ),
        default_source_intents=["provided_document", "legislation"],
        default_weight=0.04,
        clarification_weights={
            "none": 0.30, "before_search": 0.40, "after_initial_research": 0.30,
        },
    ),
    RequestTypeSpec(
        name="comparative_law",
        required_facts=[
            "juridictions à comparer",
        ],
        description="Comparaison entre le régime québécois et le régime fédéral "
                    "ou entre juridictions.",
        expected_route=_route(
            ("semantic_search_ccq", True),
            "get_ccq_articles",
            "search_legal_documents",
            ("fetch_document", True, "federal document identified"),
        ),
        route_policy=_policy(
            "official_text_retrieval", "federal_law_retrieval",
            conditional=[
                ("fetch_document", "federal document identified"),
            ],
            preferred=["semantic_search_ccq", "get_ccq_articles"],
        ),
        default_jurisdiction_status="undetermined",
        legal_domain="droit comparé",
        default_source_intents=["legislation"],
        default_weight=0.02,
        clarification_weights={
            "none": 0.60, "before_search": 0.20, "after_initial_research": 0.20,
        },
    ),
    RequestTypeSpec(
        name="dataset_coverage",
        description="Couverture du corpus A2AJ : coverage est justifié.",
        expected_route=_route(
            "coverage",
            ("search_legal_documents", True,
             "coverage reveals gap worth investigating"),
        ),
        route_policy=_policy(
            "dataset_coverage",
            conditional=[
                ("search_legal_documents",
                 "coverage reveals gap worth investigating"),
            ],
            preferred=["coverage"],
        ),
        default_jurisdiction_status="supported_federal",
        legal_domain="droit fédéral canadien",
        default_source_intents=["legislative_metadata"],
        default_weight=0.01,
        clarification_weights={
            "none": 1.0, "before_search": 0.0, "after_initial_research": 0.0,
        },
    ),
    RequestTypeSpec(
        name="non_legal",
        description="Salutation ou question hors droit : AUCUN appel MCP.",
        expected_route=_route(no_tool=True),
        route_policy=_policy(no_tool=True),
        default_jurisdiction_status="undetermined",
        legal_domain="hors droit",
        default_weight=0.03,
        clarification_weights={
            "none": 1.0, "before_search": 0.0, "after_initial_research": 0.0,
        },
    ),
]}


# ---------------------------------------------------------------------------
# Distributions par défaut (spec sections 15-16)
# ---------------------------------------------------------------------------

DEFAULT_JURISDICTION_WEIGHTS: dict[str, float] = {
    "supported_quebec": 0.58,
    "supported_federal": 0.22,
    "municipal_coverage_uncertain": 0.07,
    "supported_other_canadian": 0.08,
    "unsupported_foreign": 0.05,
}

DEFAULT_FAILURE_MODE_WEIGHTS: dict[str, float] = {
    "irrelevant_results": 0.25,
    "wrong_document_type": 0.20,
    "empty_result": 0.15,
    "truncated_source": 0.15,
    "tool_error": 0.10,
    "stale_source": 0.08,
    "malformed_result": 0.04,
    "coverage_gap": 0.03,
}

DEFAULT_FAILURE_INJECTION_RATE: float = 0.07


# ---------------------------------------------------------------------------
# Sets de politique (anciennement NO_JURISPRUDENCE / OFFICIAL_TEXT_REQUIRED)
# ---------------------------------------------------------------------------

NO_JURISPRUDENCE = {
    "exact_text_retrieval",
    "legislative_status_verification",
    "non_legal",
}

OFFICIAL_TEXT_REQUIRED = {
    "exact_text_retrieval",
    "article_explanation",
}


# ---------------------------------------------------------------------------
# Fonctions de taxonomie
# ---------------------------------------------------------------------------

def weights(overrides: Optional[dict[str, float]] = None) -> dict[str, float]:
    w = {name: rt.default_weight for name, rt in REQUEST_TYPES.items()}
    for name, val in (overrides or {}).items():
        if name not in w:
            raise KeyError(f"Type de demande inconnu dans la configuration : {name}")
        w[name] = float(val)
    return w


def sample_request_type(
    rng: random.Random,
    overrides: Optional[dict[str, float]] = None,
) -> RequestTypeSpec:
    w = weights(overrides)
    names = sorted(w)
    total = sum(w[n] for n in names)
    if total <= 0:
        raise ValueError("Somme des poids de taxonomie nulle.")
    pick = rng.uniform(0, total)
    acc = 0.0
    for name in names:
        acc += w[name]
        if pick <= acc:
            return REQUEST_TYPES[name]
    return REQUEST_TYPES[names[-1]]


# Backward compatibility
sample_category = sample_request_type


def sample_clarification_stage(
    rng: random.Random,
    request_type: RequestTypeSpec,
    overrides: Optional[dict[str, float]] = None,
) -> str:
    cw = dict(request_type.clarification_weights)
    if overrides:
        for key, val in overrides.items():
            if key in cw:
                cw[key] = float(val)
    stages = sorted(cw)
    total = sum(cw[s] for s in stages)
    if total <= 0:
        return "none"
    pick = rng.uniform(0, total)
    acc = 0.0
    for stage in stages:
        acc += cw[stage]
        if pick <= acc:
            return stage
    return "none"


def sample_jurisdiction(
    rng: random.Random,
    overrides: Optional[dict[str, float]] = None,
) -> str:
    jw = dict(DEFAULT_JURISDICTION_WEIGHTS)
    if overrides:
        for key, val in overrides.items():
            if key in jw:
                jw[key] = float(val)
    jurisdictions = sorted(jw)
    total = sum(jw[j] for j in jurisdictions)
    if total <= 0:
        return "supported_quebec"
    pick = rng.uniform(0, total)
    acc = 0.0
    for j in jurisdictions:
        acc += jw[j]
        if pick <= acc:
            return j
    return "supported_quebec"


def sample_failure_mode(
    rng: random.Random,
    injection_rate: float = DEFAULT_FAILURE_INJECTION_RATE,
    overrides: Optional[dict[str, float]] = None,
) -> Optional[str]:
    if rng.random() > injection_rate:
        return None
    fw = dict(DEFAULT_FAILURE_MODE_WEIGHTS)
    if overrides:
        for key, val in overrides.items():
            if key in fw:
                fw[key] = float(val)
    modes = sorted(fw)
    total = sum(fw[m] for m in modes)
    if total <= 0:
        return None
    pick = rng.uniform(0, total)
    acc = 0.0
    for mode in modes:
        acc += fw[mode]
        if pick <= acc:
            return mode
    return None


def target_request_type_counts(
    total: int,
    overrides: Optional[dict[str, float]] = None,
) -> dict[str, int]:
    if total < 0:
        raise ValueError("objectif total négatif")
    weighted = weights(overrides)
    weight_sum = sum(weighted.values())
    if weight_sum <= 0:
        raise ValueError("Somme des poids de taxonomie nulle.")
    exact = {name: total * value / weight_sum for name, value in weighted.items()}
    targets = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(targets.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - targets[name]), name))
    for name in order[:remaining]:
        targets[name] += 1
    return targets


# Backward compatibility
target_category_counts = target_request_type_counts


def get_request_type(name: str) -> RequestTypeSpec:
    if name not in REQUEST_TYPES:
        raise KeyError(f"Type de demande inconnu : {name}")
    return REQUEST_TYPES[name]


# Backward compatibility
def get_category(name: str) -> RequestTypeSpec:
    return get_request_type(name)


# Backward compatibility alias
CATEGORIES = REQUEST_TYPES
