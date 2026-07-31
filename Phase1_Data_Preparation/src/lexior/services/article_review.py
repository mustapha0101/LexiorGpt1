# -*- coding: utf-8 -*-
"""Structures déterministes pour relire les articles récupérés.

Le reviewer peut qualifier un texte, mais son motif narratif n'est pas une
question utilisateur. Ce module transforme cette qualification en contrat
stable : rôle de récupération, opération juridique, rôles couverts et faits
manquants. Aucun numéro d'article n'est utilisé pour choisir un rôle.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Iterable


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _contains(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


_EXCLUSION_TERMS = (
    "exclure la responsabilite", "exclusion de responsabilite",
    "limiter la responsabilite", "clause limitative", "clause d'exclusion",
    "faute lourde", "ne peut exclure", "ne peut limiter",
)
_PREVENTIVE_TERMS = (
    "menace de tomber", "abattre l'arbre", "redresser l'arbre",
    "branches ou des racines", "nuire serieusement a son usage",
)
_GENERAL_LIABILITY_TERMS = (
    "devoir de respecter", "faute", "prejudice", "dommage", "reparer",
    "responsable du prejudice", "responsabilite civile",
)
_CUSTODIAL_TERMS = (
    "gardien d'un bien", "fait autonome", "bien qu'elle a sous sa garde",
)
_PROPERTY_TERMS = (
    "ruine", "immeuble", "defaut d'entretien", "vice de construction",
)
_REMEDY_TERMS = (
    "dommages-interets", "tenu de reparer", "reparation", "prejudice",
)
_DAMAGE_TERMS = (
    "dommage", "dommages", "garage", "materiel", "prejudice",
)
_EVENT_TERMS = (
    "est tombe", "s'est tombe", "tombe sur", "accident", "deja realise",
    "deja produit", "apres la chute",
)


@dataclass(frozen=True)
class LegislativeSufficiency:
    sufficient: bool
    required_rule_roles: tuple[str, ...]
    covered_rule_roles: tuple[str, ...]
    missing_rule_roles: tuple[str, ...]
    primary_articles: tuple[str, ...]
    conditional_articles: tuple[str, ...]
    contextual_articles: tuple[str, ...]
    should_fetch_next_batch: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "required_rule_roles": list(self.required_rule_roles),
            "covered_rule_roles": list(self.covered_rule_roles),
            "missing_rule_roles": list(self.missing_rule_roles),
            "primary_articles": list(self.primary_articles),
            "conditional_articles": list(self.conditional_articles),
            "contextual_articles": list(self.contextual_articles),
            "should_fetch_next_batch": self.should_fetch_next_batch,
            "reason": self.reason,
        }


def infer_article_profile(text: str, facts: str = "") -> dict[str, Any]:
    """Infère le rôle fonctionnel d'un texte sans connaître son numéro."""
    corpus = _fold(text)
    fact_text = _fold(facts)

    if _contains(corpus, _EXCLUSION_TERMS):
        return {
            "retrieval_group": "contextual",
            "legal_operation": "exclusion_or_limitation_of_liability",
            "rule_roles": ["exclusion_or_limitation"],
            "missing_fact_keys": [],
            "clarification_priority": 0,
        }

    preventive = _contains(corpus, _PREVENTIVE_TERMS)
    event_realized = _contains(fact_text, _EVENT_TERMS) or _contains(
        fact_text, _DAMAGE_TERMS)
    if preventive and event_realized and not _contains(corpus, _REMEDY_TERMS):
        return {
            "retrieval_group": "contextual",
            "legal_operation": "preventive_neighbourhood_remedy",
            "rule_roles": ["preventive_measure"],
            "missing_fact_keys": [],
            "clarification_priority": 0,
        }

    roles: list[str] = []
    operation = "other"
    if _contains(corpus, _GENERAL_LIABILITY_TERMS):
        roles.extend(["general_liability_basis", "fault", "causation"])
        operation = "civil_liability"
    if _contains(corpus, _CUSTODIAL_TERMS):
        roles.extend(["custody_of_property", "causation"])
        operation = "custodial_property_liability"
    if _contains(corpus, _PROPERTY_TERMS):
        roles.extend(["special_property_regime", "maintenance"])
        operation = "property_ruin"
    if _contains(corpus, _REMEDY_TERMS):
        roles.append("remedy_or_practical_consequence")
    roles = list(dict.fromkeys(roles))

    if not roles:
        return {
            "retrieval_group": "contextual",
            "legal_operation": operation,
            "rule_roles": [],
            "missing_fact_keys": [],
            "clarification_priority": 0,
        }

    missing: list[str] = []
    if "fault" in roles and not _contains(
            fact_text, ("savait", "connaissait", "informe", "averti", "au courant")):
        missing.append("prior_knowledge")
    if "fault" in roles and not _contains(
            fact_text, ("a rien fait", "n'a rien fait", "aucune mesure",
                         "sans mesure", "n'a pas pris", "mesures prises")):
        missing.append("failure_to_take_reasonable_action")
    if "causation" in roles and not _contains(
            fact_text, ("a cause", "causé", "causes", "sur mon garage", "dommage")):
        missing.append("causal_connection")
    if "remedy_or_practical_consequence" in roles and not _contains(
            fact_text, ("reparation", "réparation", "indemn", "rembourse")):
        missing.append("damage_assessment")

    return {
        "retrieval_group": "primary",
        "legal_operation": operation,
        "rule_roles": roles,
        "missing_fact_keys": list(dict.fromkeys(missing)),
        "clarification_priority": 90 if "general_liability_basis" in roles else 60,
    }


def enrich_article_review(
        *, article_number: str, status: str, reason: str, text: str,
        facts: str, rank: int | None = None,
        source: str = "") -> dict[str, Any]:
    profile = infer_article_profile(text, facts)
    review = {
        "article_number": str(article_number),
        "status": status,
        "reason": reason[:300],
        "source": source,
        "reviewed": bool(reason or status != "unreviewed"),
        "text_available": bool(text.strip()),
        "rerank_rank": rank if rank is not None else 10_000,
        **profile,
    }
    return review


def known_fact_keys(facts: dict[str, Any], clarification_history: list[dict[str, Any]],
                    user_statements: Iterable[str] = ()) -> set[str]:
    """Retourne les clés déjà établies ou explicitement traitées."""
    known: set[str] = set()
    for key, value in (facts or {}).items():
        if key == "user_statements":
            continue
        if isinstance(value, dict) and "value" in value:
            if value.get("value") is not None:
                known.add(str(key))
        elif value not in (None, "", False, []):
            known.add(str(key))
    corpus = _fold(" ".join([*map(str, user_statements), *(str(x) for x in facts.values())]))
    if _contains(corpus, ("savait", "connaissait", "au courant", "informe", "averti")):
        known.add("prior_knowledge")
    if _contains(corpus, ("n'a rien fait", "aucune mesure", "n'a pas pris")):
        known.add("failure_to_take_reasonable_action")
    if _contains(corpus, ("sur mon garage", "dommage", "prejudice")):
        known.add("causal_connection")
    for entry in clarification_history or []:
        if entry.get("answered"):
            known.update(str(key) for key in entry.get("fact_keys", entry.get("missing_facts", [])))
    return known


_FACT_QUESTIONS = {
    "prior_knowledge": "Votre voisin savait-il que l'arbre était en mauvais état ou risquait de tomber avant l'accident?",
    "failure_to_take_reasonable_action": "Avait-il eu la possibilité de prendre des mesures pour réduire ce risque avant l'accident?",
    "causal_connection": "Les dommages au garage sont-ils directement liés à la chute de l'arbre?",
    "damage_assessment": "Avez-vous déjà une estimation ou une évaluation des dommages au garage?",
}


def question_for_fact_keys(fact_keys: Iterable[str]) -> str:
    """Formule une question utilisateur depuis des clés, jamais un motif interne."""
    keys = list(dict.fromkeys(str(key) for key in fact_keys if str(key).strip()))
    known = ["prior_knowledge", "failure_to_take_reasonable_action"]
    if all(key in keys for key in known):
        return (_FACT_QUESTIONS[known[0]] + " " +
                _FACT_QUESTIONS[known[1]])
    if keys and keys[0] in _FACT_QUESTIONS:
        return _FACT_QUESTIONS[keys[0]]
    labels = {key.replace("_", " ") for key in keys[:2]}
    return ("Pouvez-vous préciser le fait suivant : " +
            " et ".join(sorted(labels)) + "?")


def build_clarification(
        reviews: dict[str, dict[str, Any]], facts: dict[str, Any],
        clarification_history: list[dict[str, Any]],
        user_statements: Iterable[str] = ()) -> dict[str, Any] | None:
    """Sélectionne la meilleure règle principale et produit une question factuelle."""
    known = known_fact_keys(facts, clarification_history, user_statements)
    candidates: list[dict[str, Any]] = []
    for number, review in (reviews or {}).items():
        if review.get("retrieval_group") != "primary":
            continue
        if review.get("status") != "conditionally_applicable":
            continue
        missing = [key for key in review.get("missing_fact_keys", []) if key not in known]
        if not missing:
            continue
        question = _FACT_QUESTIONS.get(missing[0])
        if not question:
            continue
        candidates.append({
            "article_number": str(number),
            "review": review,
            "missing": missing,
            "question": question,
        })
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        -int(item["review"].get("clarification_priority", 0)),
        int(item["review"].get("rerank_rank", 10_000)),
        str(item["article_number"]),
    ))
    selected = candidates[0]
    missing = selected["missing"]
    question = selected["question"]
    return {
        "clarification_id": "fact-" + "-".join(missing),
        "category": "fact",
        "fact_keys": missing,
        "question": question,
        "answer_type": "yes_no_or_explanation",
        "source_articles": [selected["article_number"]],
        "status": "pending",
        "clarification_priority": selected["review"].get("clarification_priority", 0),
    }


def required_rule_roles(case_description: str, facts: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Déduit les rôles nécessaires de la situation, sans articles codés."""
    corpus = _fold(" ".join([case_description, *(str(v) for v in (facts or {}).values())]))
    roles = ["general_liability_basis", "fault", "causation"]
    if _contains(corpus, _DAMAGE_TERMS) or _contains(corpus, ("garage", "bien")):
        roles.append("remedy_or_practical_consequence")
    if _contains(corpus, ("arbre", "bien", "proprietaire", "immeuble", "objet")):
        roles.append("special_property_regime")
    return tuple(dict.fromkeys(roles))


def assess_legislative_sufficiency(
        reviews: dict[str, dict[str, Any]], case_description: str,
        facts: dict[str, Any] | None = None,
        remaining_candidates: bool = False) -> LegislativeSufficiency:
    required = required_rule_roles(case_description, facts)
    covered: set[str] = set()
    primary: list[str] = []
    conditional: list[str] = []
    contextual: list[str] = []
    for number, review in (reviews or {}).items():
        group = review.get("retrieval_group")
        status = review.get("status")
        roles = set(review.get("rule_roles", []))
        if group == "primary" and status in {"applicable", "conditionally_applicable"}:
            covered.update(roles)
            (conditional if status == "conditionally_applicable" else primary).append(str(number))
        elif group == "contextual":
            contextual.append(str(number))
    missing = tuple(role for role in required if role not in covered)
    sufficient = not missing and bool(primary or conditional)
    fetch_next = bool(remaining_candidates and not sufficient)
    if sufficient:
        reason = "Les rôles essentiels sont couverts par des règles principales retenues."
    elif missing:
        reason = "Des rôles juridiques essentiels restent non couverts : " + ", ".join(missing) + "."
    else:
        reason = "Aucune règle principale applicable ou conditionnelle n'est encore retenue."
    return LegislativeSufficiency(
        sufficient=sufficient,
        required_rule_roles=required,
        covered_rule_roles=tuple(sorted(covered)),
        missing_rule_roles=missing,
        primary_articles=tuple(primary),
        conditional_articles=tuple(conditional),
        contextual_articles=tuple(contextual),
        should_fetch_next_batch=fetch_next,
        reason=reason,
    )
