# -*- coding: utf-8 -*-
"""Générateur de scénarios : LLM en production, fixtures déterministes offline."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Optional

from .prompts import SCENARIO_GENERATOR_SYSTEM, scenario_user_prompt
from .schemas import ScenarioSpec
from .taxonomy import Category, get_category, sample_category


ARTICLE_RE = re.compile(r"\b(?:article\s+)?(\d{1,4}(?:\.\d+)?)\b", re.I)


FIXTURE_QUERIES = {
    "article_ccq_precis": "Quel est le texte officiel de l’article 1457 du Code civil du Québec?",
    "article_cpc_precis": "Pouvez-vous me donner le texte officiel de l’article 1 du Code de procédure civile?",
    "explication_article": "Que signifie l’article 1457 du Code civil du Québec?",
    "recherche_theme_ccq": "Quels articles du Code civil du Québec portent sur les vices cachés?",
    "recherche_theme_cpc": "Quelles dispositions du CPC concernent la signification d’une demande?",
    "cas_civil_quebecois": "J’ai acheté une maison au Québec et découvert après la vente un vice caché important. Quels sont mes recours?",
    "cas_procedure_quebecoise": "Au Québec, une demande m’a été signifiée tardivement. Quelles conséquences sont possibles?",
    "reglement_quebecois_connu": "Où puis-je consulter le règlement québécois sur l’encadrement d’activités environnementales?",
    "reglement_quebecois_inconnu": "Quel règlement québécois encadre les activités selon leur impact environnemental?",
    "jurisprudence_quebecoise": "Pouvez-vous trouver des décisions québécoises récentes sur les vices cachés?",
    "loi_federale": "Trouvez le texte officiel de la Loi sur les banques du Canada.",
    "jurisprudence_federale": "Quelles décisions de la Cour fédérale traitent récemment de marques de commerce?",
    "cas_federal_concret": "Une banque située à Montréal refuse une opération. Quelle règle fédérale pourrait s’appliquer?",
    "comparaison_quebec_federal": "Comparez les sources québécoises et fédérales applicables à cette opération bancaire.",
    "juridiction_ambigue": "Mon contrat a été rompu; quels sont mes recours?",
    "question_incomplete": "Mon patron peut-il faire ça?",
    "verification_entree_en_vigueur": "Comment vérifier si une modification législative québécoise est entrée en vigueur?",
    "couverture_dataset": "Le corpus contient-il des décisions de la Cour suprême du Canada?",
    "question_non_juridique": "Bonjour! Comment allez-vous?",
    "panne_mcp": "Donnez-moi le texte officiel de l’article 1457 du Code civil du Québec.",
    "resultat_vide": "Quels articles du CCQ parlent exactement de cryptosuccession?",
    "source_trop_longue": "Retrouvez une loi fédérale sur les banques et la partie pertinente sur les activités autorisées.",
    "clarification_puis_recherche": "Quels recours ai-je contre le vendeur de ma maison?",
}

# Ancres volontairement conservatrices : dispositions courantes dont
# l'existence est stable. Elles empêchent le modèle de fabriquer un numéro
# hors du code (p. ex. CCQ 3408) pour les catégories à article précis.
VERIFIED_CCQ_ARTICLES = (
    1, 6, 7, 10, 11, 35, 1375, 1376, 1457, 1458, 1463, 1474,
    1607, 1611, 1708, 1713, 1726, 1739, 2085, 2098, 2130, 2803,
    2843, 2925, 2938, 3148,
)
VERIFIED_CPC_ARTICLES = (
    1, 2, 9, 18, 20, 49, 84, 100, 145, 168, 171, 202, 340, 341,
    346, 491, 500, 511, 529, 535, 570,
)

_FEDERAL_MARKERS = (
    "faill", "insolv", "banque", "bancaire", "brevet", "marque de commerce",
    "maritime", "fédéral", "canada",
)
_CIVIL_MARKERS = (
    "maison", "immeuble", "vendeur", "vente", "contrat", "logement",
    "locataire", "propriétaire", "voisin", "responsabil", "préjudice",
    "vice caché",
)
_PENAL_MARKERS = ("amende", "stationnement", "infraction", "criminel", "pénal")
_LEGAL_MARKERS = (
    "article", "code civil", "tribunal", "loi", "juridique", "recours",
    "contrat", "responsabilité", "règlement",
)


class ScenarioGenerator:
    def __init__(self, client=None, seed: int = 3407, offline: bool = False,
                 taxonomy_proportions: Optional[dict[str, float]] = None):
        self.client = client
        self.rng = random.Random(seed)
        self.seed = seed
        self.offline = offline
        self.proportions = taxonomy_proportions or {}
        self.index = 0

    def generate(self, category_name: Optional[str] = None) -> ScenarioSpec:
        category = get_category(category_name) if category_name else sample_category(self.rng, self.proportions)
        self.index += 1
        if self.offline:
            return self._fixture(category)
        if self.client is None:
            raise RuntimeError("client Teacher requis hors mode offline")
        article_anchor = self._article_anchor(category.name)
        prompt = scenario_user_prompt(category.name, category.description,
                                      category.expected_jurisdiction,
                                      str(self.seed + self.index),
                                      str(article_anchor) if article_anchor else None)
        raw = self.client.complete_json("scenario_generator", [
            {"role": "system", "content": SCENARIO_GENERATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        # Les métadonnées d'évaluation sont imposées par la taxonomie; le LLM
        # n'est jamais autorisé à les remplacer dans son JSON.
        raw["request_type"] = category.name
        raw["expected_route"] = category.expected_route.model_dump(mode="json")
        raw["expected_jurisdiction"] = category.expected_jurisdiction
        raw["legal_domain"] = category.legal_domain
        raw["expected_source_types"] = category.expected_source_types
        if article_anchor:
            raw["user_query"] = self._anchored_query(
                category.name, str(raw.get("user_query", "")), article_anchor)
            facts = dict(raw.get("facts_provided") or {})
            facts["article_number"] = article_anchor
            raw["facts_provided"] = facts
        self._enforce_category_contract(raw, category)
        query = str(raw.get("user_query", ""))
        raw["scenario_id"] = hashlib.sha256(
            f"{self.seed}:{category.name}:{self.index}:{query}".encode()).hexdigest()[:16]
        facts_shape = sorted((raw.get("facts_provided") or {}).keys())
        raw["scenario_family_id"] = hashlib.sha256(
            f"{category.name}:{facts_shape}".encode()).hexdigest()[:16]
        return ScenarioSpec.model_validate(raw)

    def _enforce_category_contract(self, raw: dict, category: Category) -> None:
        """Garde-fous déterministes pour les confusions de domaine coûteuses."""
        query = str(raw.get("user_query", ""))
        folded = query.casefold()
        if category.name == "clarification_puis_recherche":
            if not any(marker in folded for marker in _CIVIL_MARKERS):
                raw["user_query"] = FIXTURE_QUERIES[category.name]
                raw["clarification_answer"] = (
                    "La maison est au Québec et j'ai découvert un vice caché "
                    "important après la vente."
                )
                raw["facts_provided"] = {}
                raw["facts_missing"] = [
                    "lieu de l'immeuble", "nature précise du problème"
                ]
            else:
                raw["facts_missing"] = list(raw.get("facts_missing") or [
                    "fait civil essentiel à préciser"
                ])
                if not raw.get("clarification_answer"):
                    raw["clarification_answer"] = (
                        "La situation est au Québec et concerne un contrat civil."
                    )
        elif category.name == "recherche_theme_cpc":
            if any(marker in folded for marker in _PENAL_MARKERS):
                raw["user_query"] = FIXTURE_QUERIES[category.name]
                raw["facts_provided"] = {}
                raw["facts_missing"] = []
                raw["clarification_answer"] = None
        elif category.name == "cas_federal_concret":
            if not any(marker in folded for marker in _FEDERAL_MARKERS):
                raw["user_query"] = FIXTURE_QUERIES[category.name]
                raw["facts_provided"] = {}
                raw["facts_missing"] = []
                raw["clarification_answer"] = None
        elif category.name == "loi_federale":
            # Cette catégorie exige une loi identifiable afin que la recherche
            # puisse fournir une citation réutilisable par fetch_document.
            if not re.search(r"\bloi\s+(?:sur|de|concernant)\b", folded):
                raw["user_query"] = FIXTURE_QUERIES[category.name]
                raw["facts_provided"] = {}
                raw["facts_missing"] = []
                raw["clarification_answer"] = None
        elif category.name == "question_non_juridique":
            if any(marker in folded for marker in _LEGAL_MARKERS):
                raw["user_query"] = FIXTURE_QUERIES[category.name]
            raw["facts_provided"] = {}
            raw["facts_missing"] = []
            raw["clarification_answer"] = None

    def _article_anchor(self, category_name: str) -> Optional[int]:
        if category_name in {"article_ccq_precis", "explication_article"}:
            return self.rng.choice(VERIFIED_CCQ_ARTICLES)
        if category_name == "article_cpc_precis":
            return self.rng.choice(VERIFIED_CPC_ARTICLES)
        return None

    @staticmethod
    def _anchored_query(category_name: str, query: str, article: int) -> str:
        # Le Teacher reçoit déjà l'ancre. Ce garde-fou garantit néanmoins que
        # la donnée ne conserve jamais un autre numéro en cas de non-respect.
        numbers = ARTICLE_RE.findall(query)
        folded = query.casefold()
        expected_code_named = (
            (category_name == "article_cpc_precis" and
             ("code de procédure civile" in folded or "cpc" in folded))
            or (category_name in {"article_ccq_precis", "explication_article"} and
                ("code civil" in folded or "ccq" in folded))
        )
        if numbers == [str(article)] and expected_code_named:
            return query
        if category_name == "article_cpc_precis":
            return (f"Pouvez-vous me donner le texte officiel de l'article {article} "
                    "du Code de procédure civile du Québec?")
        if category_name == "explication_article":
            return (f"Que signifie l'article {article} du Code civil du Québec "
                    "dans son texte officiel?")
        return (f"Pouvez-vous me donner le texte officiel de l'article {article} "
                "du Code civil du Québec?")

    def _fixture(self, category: Category) -> ScenarioSpec:
        family = hashlib.sha256(f"{category.name}:{self.index // 3}".encode()).hexdigest()[:12]
        scenario_id = hashlib.sha256(f"{self.seed}:{category.name}:{self.index}".encode()).hexdigest()[:16]
        missing = []
        clarification_answer = None
        if category.name in {"juridiction_ambigue", "question_incomplete"}:
            missing = ["juridiction", "faits essentiels"]
        if category.name == "clarification_puis_recherche":
            missing = ["lieu de l’immeuble", "nature du problème"]
            clarification_answer = "La maison est au Québec et j’ai découvert un vice caché après la vente."
        return ScenarioSpec(
            scenario_id=scenario_id,
            scenario_family_id=f"{category.name}-{family}",
            request_type=category.name,
            language="fr",
            user_query=FIXTURE_QUERIES[category.name],
            legal_domain=category.legal_domain,
            expected_jurisdiction=category.expected_jurisdiction,
            facts_provided={"fixture": True},
            facts_missing=missing,
            expected_source_types=category.expected_source_types,
            expected_route=category.expected_route,
            clarification_answer=clarification_answer,
            failure_mode=category.failure_mode,
        )
