# -*- coding: utf-8 -*-

"""
Taxonomie des demandes juridiques et politique de routage minimale.

Chaque catégorie définit :
  - la route d'outils attendue (ExpectedRoute) — la politique que le dataset
    doit enseigner ;
  - si une clarification doit précéder toute recherche ;
  - si aucun outil ne doit être appelé ;
  - un mode de panne simulé éventuel (panne_mcp, resultat_vide,
    source_trop_longue).

Les PROPORTIONS sont configurables dans configs/agentic_generation.yaml ;
ce module ne fixe que la structure.

Règle jurisprudence : elle n'est JAMAIS recherchée automatiquement. Elle est
justifiée quand l'utilisateur demande des décisions, quand la question porte
sur l'application d'une règle à des faits particuliers, quand la loi contient
une notion ouverte, ou quand des exceptions jurisprudentielles peuvent
changer le résultat. D'où : optional=True sur les étapes jurisprudence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .schemas import ExpectedRoute, ExpectedRouteStep


def _route(*steps, clarification: bool = False, no_tool: bool = False) -> ExpectedRoute:
    parsed = []
    for s in steps:
        if isinstance(s, tuple):
            tool, optional = s
            parsed.append(ExpectedRouteStep(tool=tool, optional=optional))
        else:
            parsed.append(ExpectedRouteStep(tool=s))
    return ExpectedRoute(steps=parsed, requires_clarification=clarification,
                         no_tool=no_tool)


@dataclass
class Category:
    name: str
    description: str
    expected_route: ExpectedRoute
    expected_jurisdiction: str = "Québec"
    legal_domain: str = "droit civil québécois"
    expected_source_types: list = field(default_factory=list)
    failure_mode: Optional[str] = None
    default_weight: float = 1.0


# ---------------------------------------------------------------------------
# Les 23 catégories imposées par le cahier des charges.
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, Category] = {c.name: c for c in [
    Category(
        name="article_ccq_precis",
        description="Demande du texte d'un article CCQ dont le numéro est donné.",
        expected_route=_route("get_ccq_articles"),
        expected_source_types=["legislation"],
        default_weight=2.0),
    Category(
        name="article_cpc_precis",
        description="Demande du texte d'un article CPC dont le numéro est donné.",
        expected_route=_route("get_cpc_articles"),
        legal_domain="procédure civile québécoise",
        expected_source_types=["legislation"],
        default_weight=1.5),
    Category(
        name="explication_article",
        description="Explication d'un article précis (le texte officiel doit être "
                    "récupéré avant d'expliquer).",
        expected_route=_route("get_ccq_articles"),
        expected_source_types=["legislation"],
        default_weight=1.5),
    Category(
        name="recherche_theme_ccq",
        description="Sujet CCQ sans numéro d'article : recherche sémantique "
                    "puis récupération officielle des articles retenus.",
        expected_route=_route("semantic_search_ccq", "get_ccq_articles"),
        expected_source_types=["legislation"],
        default_weight=2.0),
    Category(
        name="recherche_theme_cpc",
        description="Sujet CPC sans numéro d'article.",
        expected_route=_route("semantic_search_cpc", "get_cpc_articles"),
        legal_domain="procédure civile québécoise",
        expected_source_types=["legislation"],
        default_weight=1.5),
    Category(
        name="cas_civil_quebecois",
        description="Cas civil québécois concret : CCQ d'abord, jurisprudence "
                    "seulement si l'application factuelle le justifie.",
        expected_route=_route(("semantic_search_ccq", True), "get_ccq_articles",
                              ("search_quebec_jurisprudence", True)),
        expected_source_types=["legislation", "jurisprudence"],
        default_weight=2.0),
    Category(
        name="cas_procedure_quebecoise",
        description="Cas concret de procédure civile québécoise.",
        expected_route=_route(("semantic_search_cpc", True), "get_cpc_articles",
                              ("search_quebec_jurisprudence", True)),
        legal_domain="procédure civile québécoise",
        expected_source_types=["legislation"],
        default_weight=1.0),
    Category(
        name="reglement_quebecois_connu",
        description="Règlement québécois dont l'identité est connue : recherche "
                    "puis récupération par l'URL réellement retournée.",
        expected_route=_route("search_quebec_regulations", "get_quebec_regulation"),
        legal_domain="droit réglementaire québécois",
        expected_source_types=["reglement"],
        default_weight=1.0),
    Category(
        name="reglement_quebecois_inconnu",
        description="Règlement québécois à identifier : search_quebec_regulations "
                    "puis get_quebec_regulation avec l'URL retournée.",
        expected_route=_route("search_quebec_regulations", "get_quebec_regulation"),
        legal_domain="droit réglementaire québécois",
        expected_source_types=["reglement"],
        default_weight=1.0),
    Category(
        name="jurisprudence_quebecoise",
        description="L'utilisateur demande explicitement des décisions québécoises.",
        expected_route=_route(("get_ccq_articles", True), "search_quebec_jurisprudence"),
        expected_source_types=["jurisprudence"],
        default_weight=1.0),
    Category(
        name="loi_federale",
        description="Loi fédérale canadienne : recherche A2AJ puis fetch_document "
                    "avec la citation réellement retournée.",
        expected_route=_route("search_legal_documents", "fetch_document"),
        expected_jurisdiction="Canada (fédéral)",
        legal_domain="droit fédéral canadien",
        expected_source_types=["legislation"],
        default_weight=1.5),
    Category(
        name="jurisprudence_federale",
        description="Décisions des tribunaux fédéraux ou de la Cour suprême.",
        expected_route=_route("search_legal_documents", ("fetch_document", True)),
        expected_jurisdiction="Canada (fédéral)",
        legal_domain="droit fédéral canadien",
        expected_source_types=["jurisprudence"],
        default_weight=1.0),
    Category(
        name="cas_federal_concret",
        description="Cas concret relevant du droit fédéral (banque, faillite, "
                    "propriété intellectuelle, droit maritime...). Route fédérale, "
                    "pas CCQ par défaut.",
        expected_route=_route("search_legal_documents", "fetch_document",
                              ("search_legal_documents", True)),
        expected_jurisdiction="Canada (fédéral)",
        legal_domain="droit fédéral canadien",
        expected_source_types=["legislation", "jurisprudence"],
        default_weight=1.0),
    Category(
        name="comparaison_quebec_federal",
        description="Comparaison entre le régime québécois et le régime fédéral.",
        expected_route=_route("semantic_search_ccq", "get_ccq_articles",
                              "search_legal_documents",
                              ("fetch_document", True)),
        expected_jurisdiction="Québec et Canada (fédéral)",
        legal_domain="droit comparé Québec/fédéral",
        expected_source_types=["legislation"],
        default_weight=0.7),
    Category(
        name="juridiction_ambigue",
        description="La juridiction dépend d'un fait manquant : clarification "
                    "AVANT toute recherche.",
        expected_route=_route(clarification=True),
        expected_jurisdiction="indéterminée",
        default_weight=1.0),
    Category(
        name="question_incomplete",
        description="Information essentielle manquante (« Mon patron peut-il "
                    "faire ça ? ») : clarification avant recherche.",
        expected_route=_route(clarification=True),
        expected_jurisdiction="indéterminée",
        default_weight=1.0),
    Category(
        name="verification_entree_en_vigueur",
        description="Vérifier l'entrée en vigueur / les modifications d'une "
                    "disposition québécoise.",
        expected_route=_route("get_quebec_legal_info"),
        expected_source_types=["metadonnees_legislatives"],
        default_weight=0.7),
    Category(
        name="couverture_dataset",
        description="La couverture du corpus A2AJ est incertaine : coverage est "
                    "justifié (jamais appelé systématiquement ailleurs).",
        expected_route=_route("coverage", ("search_legal_documents", True)),
        expected_jurisdiction="Canada (fédéral)",
        legal_domain="droit fédéral canadien",
        expected_source_types=["metadonnees_dataset"],
        default_weight=0.5),
    Category(
        name="question_non_juridique",
        description="Salutation ou question hors droit : AUCUN appel MCP.",
        expected_route=_route(no_tool=True),
        expected_jurisdiction="sans objet",
        legal_domain="hors droit",
        default_weight=0.7),
    Category(
        name="panne_mcp",
        description="L'outil renvoie une erreur : le modèle doit le dire, sans "
                    "fabriquer de réponse.",
        expected_route=_route("get_ccq_articles"),
        failure_mode="panne_mcp",
        default_weight=0.5),
    Category(
        name="resultat_vide",
        description="La recherche ne renvoie rien : reformulation limitée puis "
                    "réponse prudente.",
        expected_route=_route("semantic_search_ccq", ("semantic_search_ccq", True),
                              ("coverage", True)),
        failure_mode="resultat_vide",
        default_weight=0.5),
    Category(
        name="source_trop_longue",
        description="fetch_document renvoie un document très long : récupération "
                    "par section ou par plage de caractères, troncature déclarée.",
        expected_route=_route("search_legal_documents", "fetch_document",
                              ("fetch_document", True)),
        expected_jurisdiction="Canada (fédéral)",
        legal_domain="droit fédéral canadien",
        failure_mode="source_trop_longue",
        default_weight=0.5),
    Category(
        name="clarification_puis_recherche",
        description="Clarification d'abord ; la réponse de l'utilisateur permet "
                    "ensuite la recherche.",
        expected_route=_route(("semantic_search_ccq", True), "get_ccq_articles",
                              clarification=True),
        default_weight=1.0),
]}


def weights(overrides: Optional[dict[str, float]] = None) -> dict[str, float]:
    """Poids d'échantillonnage : défauts de la taxonomie, surchargés par le YAML."""
    w = {name: cat.default_weight for name, cat in CATEGORIES.items()}
    for name, val in (overrides or {}).items():
        if name not in w:
            raise KeyError(f"Catégorie inconnue dans la configuration : {name}")
        w[name] = float(val)
    return w


def sample_category(rng: random.Random,
                    overrides: Optional[dict[str, float]] = None) -> Category:
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
            return CATEGORIES[name]
    return CATEGORIES[names[-1]]


def target_category_counts(total: int,
                           overrides: Optional[dict[str, float]] = None) -> dict[str, int]:
    """Répartit un objectif d'acceptation selon les poids de la taxonomie.

    La méthode du plus fort reste garantit que la somme des quotas est
    exactement ``total``. Le CLI pilote ensuite les *acceptés* vers ces quotas,
    plutôt que de laisser les catégories faciles monopoliser le dataset.
    """
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


def get_category(name: str) -> Category:
    if name not in CATEGORIES:
        raise KeyError(f"Catégorie inconnue : {name}")
    return CATEGORIES[name]


# Catégories où une recherche de jurisprudence est un signe de sur-recherche :
# une simple demande d'article ne la justifie jamais.
NO_JURISPRUDENCE = {
    "article_ccq_precis", "article_cpc_precis", "explication_article",
    "recherche_theme_ccq", "recherche_theme_cpc",
    "reglement_quebecois_connu", "reglement_quebecois_inconnu",
    "verification_entree_en_vigueur", "question_non_juridique",
}

# Catégories où l'utilisateur demande un TEXTE OFFICIEL : répondre de mémoire
# (aucun appel d'outil) est une faute de politique.
OFFICIAL_TEXT_REQUIRED = {
    "article_ccq_precis", "article_cpc_precis", "explication_article",
    "reglement_quebecois_connu", "reglement_quebecois_inconnu",
}
