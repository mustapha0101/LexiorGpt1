# -*- coding: utf-8 -*-
"""Classification générique des expressions de recours.

Le service ne connaît ni les scénarios ni les numéros d'articles. Il interprète
une expression et vérifie si les textes officiels déjà récupérés la couvrent.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from lexior.agentic.schemas import RemedyIntent, RuleContract


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(char for char in text if not unicodedata.combining(char)).casefold()


def _first_match(text: str, patterns: Iterable[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def _source_evidence(source_texts: dict[str, str], patterns: Iterable[str],
                     proposition: str) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    compiled = [re.compile(_fold(pattern), re.IGNORECASE) for pattern in patterns]
    for source_id, text in source_texts.items():
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", text or "")
            if part.strip()
        ]
        for sentence in sentences:
            if any(pattern.search(_fold(sentence)) for pattern in compiled):
                evidence.append({
                    "source_id": str(source_id),
                    "passage": sentence,
                    "proposition": proposition,
                })
                break
    return evidence


def _contract_text(rule_contract: RuleContract | dict[str, Any] | None) -> str:
    if not rule_contract:
        return ""
    contract = (
        rule_contract if isinstance(rule_contract, dict)
        else rule_contract.model_dump(mode="json")
    )
    parts: list[str] = []
    for key in (
        "rule_summary", "permitted_claims", "prohibited_claims",
        "conditional_branches", "application_limits",
    ):
        value = contract.get(key, "")
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    for key in ("elements", "consequences_or_remedies"):
        for element in contract.get(key, []) or []:
            if isinstance(element, dict):
                parts.append(str(element.get("description", "")))
                parts.extend(
                    str(item) for item in element.get("supporting_passages", [])
                )
    return "\n".join(parts)


def classify_remedy_intent(
    raw_text: str,
    *,
    source_texts: dict[str, str] | None = None,
    rule_contract: RuleContract | dict[str, Any] | None = None,
) -> RemedyIntent:
    """Classifie une expression et borne ses interprétations aux sources."""
    text = str(raw_text or "").strip()
    folded = _fold(text)
    sources = dict(source_texts or {})
    source_context = "\n".join(sources.values()) + "\n" + _contract_text(rule_contract)
    folded_sources = _fold(source_context)

    raw_expression = _first_match(text, (
        r"porter plainte", r"déposer une plainte", r"deposer une plainte",
        r"faire une plainte", r"plainte", r"poursuivre", r"signaler",
        r"signalement", r"faire une réclamation", r"faire une reclamation",
        r"réclamation", r"reclamation", r"mise en demeure", r"recours",
        r"indemnisation", r"être indemnisé", r"etre indemnise",
        r"dommages[- ]intérêts", r"dommages[- ]interets",
        r"réparer les dommages",
    ))
    if not raw_expression:
        return RemedyIntent(reason="Aucune expression de recours détectée.")

    possible: list[str] = []
    explicit_police = bool(re.search(
        r"\b(?:police|criminel(?:le)?|pénal(?:e)?|penal(?:e)?)\b", folded))
    explicit_municipal = bool(re.search(
        r"\b(?:ville|municipal(?:e)?|municipalité|municipalite|arrondissement|inspecteur)\b",
        folded))
    explicit_civil = bool(re.search(
        r"\b(?:indemn(?:isation|iser)|réparation|reparation|dommages?[- ]intérêts|"
        r"dommages?[- ]interets|préjudice|prejudice|compensation)\b", folded))
    if explicit_police:
        possible.append("police_or_criminal_complaint")
    if explicit_municipal:
        possible.append("municipal_or_administrative_report")
    if explicit_civil:
        possible.append("civil_compensation")
    if (re.search(r"\b(?:porter plainte|déposer|deposer|plainte)\b", folded)
            and not (explicit_police or explicit_municipal)):
        possible.extend([
            "civil_compensation", "police_or_criminal_complaint",
            "municipal_or_administrative_report",
        ])
    if re.search(r"\bpoursuivre\b|\brecours\b", folded):
        possible.extend(["civil_compensation", "civil_proceeding"])
    if re.search(r"\b(?:réclamation|reclamation)\b", folded):
        possible.append("civil_compensation")
    if re.search(r"mise en demeure", folded):
        possible.append("formal_notice")
    if re.search(
        r"\b(?:ordre professionnel|employeur|disciplin(?:aire|e)|organisme de réglementation)\b",
        folded,
    ):
        possible.append("internal_or_regulatory_complaint")
    possible = list(dict.fromkeys(possible))
    if not possible:
        possible = ["unknown_or_ambiguous"]

    support_patterns: dict[str, tuple[tuple[str, ...], str]] = {
        "civil_compensation": (
            (
                r"réparer le préjudice", r"reparer le prejudice",
                r"réparer(?: [a-z]+){0,3} préjudice",
                r"reparer(?: [a-z]+){0,3} prejudice",
                r"dommages[- ]intérêts", r"dommages[- ]interets",
                r"indemn(?:iser|isation)", r"réparation du préjudice",
                r"reparation du prejudice",
            ),
            "la réparation du préjudice",
        ),
        "civil_proceeding": (
            (
                r"tribunal", r"action judiciaire", r"demande judiciaire",
                r"introduire", r"instance civile",
            ),
            "une procédure civile",
        ),
        "police_or_criminal_complaint": (
            (
                r"police", r"criminel", r"pénal", r"penal",
                r"infraction", r"accusation",
            ),
            "une plainte policière ou criminelle",
        ),
        "municipal_or_administrative_report": (
            (
                r"municipal", r"municipalité", r"municipalite", r"ville",
                r"autorité", r"autorite", r"règlement", r"reglement",
                r"signalement",
            ),
            "un signalement municipal ou administratif",
        ),
        "formal_notice": (
            (r"mise en demeure", r"demande formelle"),
            "une mise en demeure",
        ),
        "internal_or_regulatory_complaint": (
            (
                r"ordre professionnel", r"organisme", r"disciplin",
                r"employeur",
            ),
            "une plainte interne ou réglementaire",
        ),
    }
    supported: list[str] = []
    evidence: dict[str, list[dict[str, str]]] = {}
    for remedy_type in possible:
        if remedy_type not in support_patterns:
            continue
        patterns, proposition = support_patterns[remedy_type]
        found = _source_evidence(sources, patterns, proposition)
        if not found and any(re.search(_fold(pattern), folded_sources)
                             for pattern in patterns):
            found = [{
                "source_id": "rule_contract",
                "passage": _contract_text(rule_contract),
                "proposition": proposition,
            }]
        if found:
            supported.append(remedy_type)
            evidence[remedy_type] = found
    unsupported = [
        item for item in possible
        if item not in supported and item != "unknown_or_ambiguous"
    ]
    ambiguity = len(possible) > 1 or "unknown_or_ambiguous" in possible
    primary = next(
        (
            item for item in (
                "civil_compensation", "civil_proceeding",
                "police_or_criminal_complaint",
                "municipal_or_administrative_report", "formal_notice",
                "internal_or_regulatory_complaint",
            )
            if item in supported
        ),
        possible[0] if possible else "unknown_or_ambiguous",
    )
    distinctions: list[str] = []
    if ambiguity and "civil_compensation" in possible:
        distinctions.append(
            "Une réclamation civile vise la réparation du préjudice; une plainte policière ou un signalement municipal constituent des démarches distinctes et ne sont pas établis par une source civile seule."
        )
    elif "police_or_criminal_complaint" in unsupported:
        distinctions.append(
            "Les sources récupérées ne permettent pas de confirmer une procédure policière ou criminelle."
        )
    elif "municipal_or_administrative_report" in unsupported:
        distinctions.append(
            "Les sources récupérées ne permettent pas de confirmer un signalement municipal ou administratif."
        )
    clarification_required = (
        not supported
        and any(item != "unknown_or_ambiguous" for item in possible)
    )
    answerable = bool(supported) and (ambiguity or bool(unsupported))
    requested_outcome = (
        "obtenir une réparation ou une indemnisation"
        if "civil_compensation" in possible
        else "engager une démarche auprès d'une autorité"
        if possible
        else ""
    )
    return RemedyIntent(
        raw_expression=raw_expression,
        requested_outcome=requested_outcome,
        possible_remedy_types=possible,
        primary_interpretation=primary,
        ambiguity_detected=ambiguity,
        clarification_required=clarification_required,
        answerable_with_distinctions=answerable,
        supported_remedy_types=supported,
        unsupported_remedy_types=unsupported,
        distinction_to_explain=distinctions,
        reason=(
            "L'expression est ambiguë; une interprétation couverte peut être "
            "répondue avec la distinction des démarches non couvertes."
            if answerable and ambiguity
            else "L'expression explicite est couverte par les sources récupérées."
            if supported
            else "Aucun type de recours demandé n'est couvert par les sources récupérées."
        ),
        supported_remedy_evidence=evidence,
    )
