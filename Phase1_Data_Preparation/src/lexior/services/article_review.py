# -*- coding: utf-8 -*-
"""Deterministic review structures for Phase 1 legal retrieval."""

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
_DAMAGE_TERMS = ("dommage", "dommages", "garage", "materiel", "prejudice")
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
    sufficient_for_conditional_answer: bool | None = None
    required_roles_covered: tuple[str, ...] = ()
    supporting_rule_roles: tuple[str, ...] = ()
    supporting_roles_covered: tuple[str, ...] = ()
    optional_rule_roles: tuple[str, ...] = ()
    optional_roles_missing: tuple[str, ...] = ()

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
            "sufficient_for_conditional_answer": (
                self.sufficient if self.sufficient_for_conditional_answer is None
                else self.sufficient_for_conditional_answer),
            "required_roles_covered": list(self.required_roles_covered),
            "supporting_rule_roles": list(self.supporting_rule_roles),
            "supporting_roles_covered": list(self.supporting_roles_covered),
            "optional_rule_roles": list(self.optional_rule_roles),
            "optional_roles_missing": list(self.optional_roles_missing),
        }


def infer_article_profile(text: str, facts: str = "") -> dict[str, Any]:
    """Infer a functional role from a text without using its article number."""
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
            fact_text, ("a cause", "cause", "sur mon garage", "dommage")):
        missing.append("causal_connection")
    if "remedy_or_practical_consequence" in roles and not _contains(
            fact_text, ("reparation", "indemn", "rembourse")):
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
        facts: str, rank: int | None = None, source: str = "") -> dict[str, Any]:
    profile = infer_article_profile(text, facts)
    return {
        "article_number": str(article_number),
        "status": status,
        "reason": reason[:300],
        "source": source,
        "reviewed": bool(reason or status != "unreviewed"),
        "text_available": bool(text.strip()),
        "rerank_rank": rank if rank is not None else 10_000,
        **profile,
    }


def known_fact_keys(facts: dict[str, Any], clarification_history: list[dict[str, Any]],
                    user_statements: Iterable[str] = ()) -> set[str]:
    """Return keys established or explicitly answered by the user."""
    known: set[str] = set()
    for key, value in (facts or {}).items():
        if key in {"user_statements", "entities", "case_entities"}:
            continue
        if isinstance(value, dict) and "value" in value:
            if value.get("value") is not None:
                known.add(str(key))
        elif value not in (None, "", False, []):
            known.add(str(key))
    corpus = _fold(" ".join([
        *map(str, user_statements), *(str(x) for x in facts.values())]))
    if _contains(corpus, ("savait", "connaissait", "au courant", "informe", "averti")):
        known.add("prior_knowledge")
    if _contains(corpus, ("n'a rien fait", "aucune mesure", "n'a pas pris")):
        known.add("failure_to_take_reasonable_action")
    if _contains(corpus, ("sur mon garage", "dommage", "prejudice")):
        known.add("causal_connection")
    for entry in clarification_history or []:
        if (entry.get("answered") and entry.get("answer_interpretation")
                not in {"uncertain", "unanswered"}):
            known.update(str(key) for key in entry.get(
                "fact_keys", entry.get("missing_facts", [])))
    return known


_FACT_QUESTIONS = {
    "prior_knowledge": "La personne concernee connaissait-elle le risque avant l'evenement?",
    "failure_to_take_reasonable_action": "La personne concernee pouvait-elle prendre des mesures raisonnables avant l'evenement?",
    "causal_connection": "Les dommages decrits sont-ils directement lies a l'evenement?",
    "damage_assessment": "Disposez-vous d'une estimation ou d'une evaluation des dommages?",
}


def _entity_value(entities: dict[str, Any] | None, key: str) -> str:
    value = (entities or {}).get(key, "")
    if isinstance(value, dict):
        value = value.get("value", "")
    return str(value or "").strip()


def question_for_fact_keys(
        fact_keys: Iterable[str], entities: dict[str, Any] | None = None) -> str:
    """Build a user-facing question from keys, without internal motifs."""
    keys = list(dict.fromkeys(str(key) for key in fact_keys if str(key).strip()))
    if keys and keys[0] in _FACT_QUESTIONS:
        question = _FACT_QUESTIONS[keys[0]]
        actor = _entity_value(entities, "actor") or "La personne concernee"
        object_name = _entity_value(entities, "object")
        event_name = _entity_value(entities, "event")
        damaged_object = _entity_value(entities, "damaged_object")
        event_stage = _entity_value(entities, "event_stage") or "avant l'evenement"
        if keys[0] == "prior_knowledge" and object_name:
            question = (f"{actor} connaissait-il ou connaissait-elle le risque lie a "
                        f"{object_name} {event_stage}?")
        elif keys[0] == "failure_to_take_reasonable_action":
            risk = _entity_value(entities, "risk_condition") or "ce risque"
            question = (f"{actor} pouvait-il ou pouvait-elle prendre des mesures "
                        f"raisonnables pour reduire {risk} {event_stage}?")
        elif keys[0] == "causal_connection":
            event = event_name or "l'evenement"
            target = damaged_object or "le dommage decrit"
            question = f"Le dommage concernant {target} est-il directement lie a {event}?"
        return question
    labels = {key.replace("_", " ") for key in keys[:2]}
    return ("Pouvez-vous preciser le fait suivant : " +
            " et ".join(sorted(labels)) + "?")


def build_clarification(
        reviews: dict[str, dict[str, Any]], facts: dict[str, Any],
        clarification_history: list[dict[str, Any]],
        user_statements: Iterable[str] = ()) -> dict[str, Any] | None:
    """Select one primary missing fact and bind only that fact to the question."""
    known = known_fact_keys(facts, clarification_history, user_statements)
    candidates: list[dict[str, Any]] = []
    entities = (facts or {}).get("entities") or (facts or {}).get("case_entities")
    for number, review in (reviews or {}).items():
        if review.get("retrieval_group") != "primary":
            continue
        if review.get("status") != "conditionally_applicable":
            continue
        missing = [key for key in review.get("missing_fact_keys", []) if key not in known]
        if not missing:
            continue
        selected_key = str(missing[0])
        question = question_for_fact_keys([selected_key], entities)
        candidates.append({
            "article_number": str(number), "review": review, "missing": missing,
            "question_fact_keys": [selected_key], "question": question,
        })
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        -int(item["review"].get("clarification_priority", 0)),
        int(item["review"].get("rerank_rank", 10_000)),
        str(item["article_number"])))
    selected = candidates[0]
    question_fact_keys = selected["question_fact_keys"]
    return {
        "clarification_id": "fact-" + "-".join(question_fact_keys),
        "category": "fact",
        "fact_keys": question_fact_keys,
        "question": selected["question"],
        "answer_type": "yes_no_or_explanation",
        "source_articles": [selected["article_number"]],
        "status": "pending",
        "clarification_priority": selected["review"].get("clarification_priority", 0),
    }


def required_rule_roles(case_description: str, facts: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return roles strictly required for a conditional answer."""
    return ("general_liability_basis",)


def assess_legislative_sufficiency(
        reviews: dict[str, dict[str, Any]], case_description: str,
        facts: dict[str, Any] | None = None,
        remaining_candidates: bool = False) -> LegislativeSufficiency:
    required = required_rule_roles(case_description, facts)
    supporting = ("fault", "causation", "custody_of_property", "maintenance")
    optional = ("special_property_regime", "remedy_or_practical_consequence",
                "preventive_measure", "exclusion_or_limitation")
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
    required_covered = tuple(role for role in required if role in covered)
    supporting_covered = tuple(role for role in supporting if role in covered)
    optional_missing = tuple(role for role in optional if role not in covered)
    sufficient = not missing and bool(primary or conditional)
    fetch_next = bool(remaining_candidates and not sufficient)
    if sufficient:
        reason = ("La base principale est couverte; les autres roles restent "
                  "des precisions conditionnelles ou de soutien.")
    elif missing:
        reason = ("Des roles juridiques essentiels restent non couverts : "
                  + ", ".join(missing) + ".")
    else:
        reason = "Aucune regle principale applicable ou conditionnelle n'est retenue."
    return LegislativeSufficiency(
        sufficient=sufficient, required_rule_roles=required,
        covered_rule_roles=tuple(sorted(covered)), missing_rule_roles=missing,
        primary_articles=tuple(primary), conditional_articles=tuple(conditional),
        contextual_articles=tuple(contextual), should_fetch_next_batch=fetch_next,
        reason=reason, sufficient_for_conditional_answer=sufficient,
        required_roles_covered=required_covered, supporting_rule_roles=supporting,
        supporting_roles_covered=supporting_covered,
        optional_rule_roles=optional, optional_roles_missing=optional_missing,
    )


_FACT_LABELS = {
    "prior_knowledge": "connaissance anterieure du risque",
    "failure_to_take_reasonable_action": "mesures raisonnables non prises",
    "causal_connection": "lien causal entre l'evenement et les dommages",
    "damage_assessment": "evaluation des dommages",
}


def fact_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def format_facts_for_query(facts: dict[str, Any] | None) -> list[str]:
    """Format only asserted facts; never expose internal keys or dict repr."""
    formatted: list[str] = []
    for key, raw in sorted((facts or {}).items()):
        if key in {"user_statements", "entities", "case_entities"}:
            continue
        value = fact_value(raw)
        if value is True:
            formatted.append(_FACT_LABELS.get(key, key.replace("_", " ")))
        elif isinstance(value, str) and value.strip():
            formatted.append(value.strip())
    return formatted


def _first_sentences(text: str, limit: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(part.strip() for part in parts[:limit] if part.strip())[:500]


def build_conditional_reasoning_contract(
        reviews: dict[str, dict[str, Any]], texts: dict[str, str],
        facts: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Create a source-bounded contract for conditional reasoning."""
    result: list[dict[str, Any]] = []
    asserted = format_facts_for_query(facts)
    known = set(known_fact_keys(facts or {}, [], []))
    for number, review in (reviews or {}).items():
        if review.get("retrieval_group") != "primary":
            continue
        if review.get("status") not in {"applicable", "conditionally_applicable"}:
            continue
        text = texts.get(str(number), "")
        missing = [str(key) for key in review.get("missing_fact_keys", []) if key not in known]
        result.append({
            "article": str(number),
            "status": review.get("status", "conditionally_applicable"),
            "source_propositions": [_first_sentences(text)] if text else [],
            "user_asserted_facts": [
                {"fact": fact, "status": "asserted_not_verified"}
                for fact in asserted],
            "unresolved_conditions": [
                _FACT_LABELS.get(key, key.replace("_", " ")) for key in missing],
            "permitted_claims": [
                "le texte peut être pertinent si ses conditions sont réunies",
                "le fait déclaré peut appuyer l'analyse sans établir à lui seul la responsabilité",
            ],
            "prohibited_claims": [
                "responsabilité automatique",
                "condition juridique présentée comme établie sans preuve",
            ],
        })
    return result
