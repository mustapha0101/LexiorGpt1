# -*- coding: utf-8 -*-
"""Evidence-first policies shared by the live and dataset graph.

This module deliberately contains no article numbers or scenario-specific
rules. It only projects already retrieved observations and user-stated facts
into typed, auditable contracts.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from lexior.agentic.schemas import (
    ClaimLedger,
    LegalClaim,
    PrimaryAuthoritySelection,
    RuleContract,
    RuleElement,
    SourceRejection,
    SourceSufficiencyDecision,
    ToolObservation,
    Decision,
    DecisionTrace,
    PlannerDecision,
)

from .article_review import infer_article_profile


_REGULATION_RE = re.compile(
    r"[^.\n]{0,100}\b(?:prescrit|prévu|prevu|selon|déterminé|determine|"
    r"formulaire obligatoire|annexe)\b[^.\n]{0,160}\b(?:règlement|reglement|annexe|formulaire)\b[^.\n]*",
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r"\b(?:article|art\.)\s+(\d{1,4}(?:\.\d+)?)", re.I)
_LEGAL_MARKER_RE = re.compile(
    r"\b(?:article|articles|code civil|loi|règlement|reglement|responsabil|"
    r"doit|peut|tenu de|droit à|droit au|prescription|délai|delai)\b", re.I,
)


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def source_id(tool_name: str, article_number: str) -> str:
    prefix = "ccq" if "ccq" in tool_name.casefold() else "cpc" if "cpc" in tool_name.casefold() else tool_name
    return f"{prefix}:{article_number}"


def retrieved_articles(observations: Iterable[ToolObservation]) -> dict[str, tuple[str, str, int]]:
    """Return source_id -> (text, article number, observation index)."""
    result: dict[str, tuple[str, str, int]] = {}
    for index, observation in enumerate(observations):
        if observation.tool_name not in {"get_ccq_articles", "get_cpc_articles"} or not observation.ok:
            continue
        text = observation.normalized_response or ""
        numbers = list(dict.fromkeys(_ARTICLE_RE.findall(text)))
        if not numbers:
            for raw in observation.arguments.get("articles", []) if isinstance(observation.arguments, dict) else []:
                numbers.append(str(raw))
        blocks = re.split(r"(?=\bArticle\s+\d)", text, flags=re.I)
        for number in numbers:
            match = next((block for block in blocks if re.search(rf"\bArticle\s+{re.escape(number)}\b", block, re.I)), text)
            result[source_id(observation.tool_name, number)] = (match.strip(), number, index)
    return result


def select_primary_authorities(
    observations: Iterable[ToolObservation],
    reviews: dict[str, dict[str, Any]],
    *, task_id: str = "",
    maximum_primary: int = 3,
    maximum_secondary: int = 3,
) -> PrimaryAuthoritySelection:
    articles = retrieved_articles(observations)
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for sid, (text, number, _index) in articles.items():
        review = reviews.get(number, {})
        status = str(review.get("status", ""))
        profile = infer_article_profile(text, "")
        if status not in {"applicable", "conditionally_applicable", ""}:
            continue
        score = 0.0
        score += 3.0 if status == "applicable" else 2.0
        score += 1.0 if review.get("retrieval_group") == "primary" else 0.0
        score += min(0.5, len(profile.get("rule_roles", [])) * 0.1)
        candidates.append((score, sid, {"text": text, "number": number, "review": review, "profile": profile}))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    primary = [sid for _score, sid, _meta in candidates[:max(1, min(maximum_primary, 3))]]
    secondary = [sid for _score, sid, _meta in candidates[max(1, min(maximum_primary, 3)):max(1, min(maximum_primary, 3)) + maximum_secondary]]
    rejected: list[SourceRejection] = []
    selected = set(primary) | set(secondary)
    for sid, (_text, number, _index) in articles.items():
        if sid in selected:
            continue
        review = reviews.get(number, {})
        reason = str(review.get("reason") or "non retenue par la comparaison source-faits")
        if str(review.get("status")) == "incompatible":
            reason = reason or "source incompatible avec les faits déclarés"
        rejected.append(SourceRejection(source_id=sid, reason=reason))
    confidence = None
    if candidates:
        confidence = min(1.0, 0.5 + candidates[0][0] / 10.0)
    return PrimaryAuthoritySelection(
        task_id=task_id,
        primary_sources=primary,
        secondary_sources=secondary,
        rejected_sources=rejected,
        selection_reason=("Sources officielles récupérées comparées aux faits déclarés; "
                          "seules les sources allowlistées sont transmises au rédacteur."),
        confidence=confidence,
    )


def _status_for(review: dict[str, Any], known_facts: set[str]) -> str:
    missing = set(str(x) for x in review.get("missing_fact_keys", []))
    if missing - known_facts:
        return "missing" if not known_facts else "uncertain"
    status = str(review.get("status", ""))
    return "present" if status == "applicable" else "uncertain"


def build_rule_contract(
    selection: PrimaryAuthoritySelection,
    observations: Iterable[ToolObservation],
    reviews: dict[str, dict[str, Any]],
    facts: dict[str, Any],
    *, task_id: str = "",
) -> RuleContract:
    articles = retrieved_articles(observations)
    known = {str(key) for key, value in facts.items() if value not in (None, "", [], {})}
    elements: list[RuleElement] = []
    roles: list[str] = []
    decisive: list[str] = []
    for sid in selection.primary_sources:
        if sid not in articles:
            continue
        text, number, _index = articles[sid]
        review = reviews.get(number, {})
        profile = review or infer_article_profile(text, "")
        for role in profile.get("rule_roles", []):
            if role not in roles:
                roles.append(str(role))
                elements.append(RuleElement(
                    id=str(role),
                    description=f"Élément opérant identifié dans la source {sid}: {role}.",
                    support_source_ids=[sid],
                    status=_status_for(profile, known),
                ))
        for key in profile.get("missing_fact_keys", []):
            key = str(key)
            if key not in decisive and key not in known:
                decisive.append(key)
        if not profile.get("rule_roles"):
            elements.append(RuleElement(
                id=f"operative_text_{number}",
                description=text[:500],
                support_source_ids=[sid],
                status="uncertain",
            ))
    summary = " ".join(
        articles[sid][0][:300] for sid in selection.primary_sources if sid in articles
    ).strip()
    return RuleContract(
        task_id=task_id,
        rule_type=", ".join(roles) or "source_bounded_rule",
        rule_summary=summary,
        primary_source_ids=list(selection.primary_sources),
        elements=elements,
        decisive_facts_needed=decisive,
        facts_not_required=[],
        application_limits=["L’application reste limitée au texte effectivement récupéré et aux faits déclarés."],
    )


def validate_rule_contract(contract: RuleContract,
                           retrieved_source_ids: set[str],
                           rejected_source_ids: set[str] | None = None) -> list[str]:
    """Validate provenance without importing any legal knowledge."""
    rejected = rejected_source_ids or set()
    errors: list[str] = []
    for source_id in contract.primary_source_ids:
        if source_id not in retrieved_source_ids:
            errors.append(f"source principale absente: {source_id}")
    for element in [*contract.elements, *contract.supported_exceptions]:
        if not element.support_source_ids:
            errors.append(f"élément sans source: {element.id}")
        for source_id in element.support_source_ids:
            if source_id not in retrieved_source_ids:
                errors.append(f"source absente pour {element.id}: {source_id}")
            if source_id in rejected:
                errors.append(f"source rejetée utilisée par {element.id}: {source_id}")
    return list(dict.fromkeys(errors))


def authorize_planner_action(state: dict[str, Any], proposed_action: Any) -> PlannerDecision:
    """Deterministic policy boundary after the LLM planner proposal."""
    decision = (proposed_action if isinstance(proposed_action, PlannerDecision)
                else PlannerDecision.model_validate(proposed_action))
    sufficiency = state.get("source_sufficiency_decision") or {}
    if hasattr(sufficiency, "model_dump"):
        sufficiency = sufficiency.model_dump(mode="json")
    if (decision.decision == Decision.call_tool
            and decision.next_tool == "search_quebec_jurisprudence"
            and state.get("request_type") != "case_law_research"
            and sufficiency.get("legislation_status") == "sufficient"
            and sufficiency.get("jurisprudence_status") == "not_needed"):
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            decision=Decision.final_answer,
            thinking_text="Les sources officielles suffisent pour une première réponse; la jurisprudence n'est pas autorisée sans lacune explicite.",
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="source suffisante",
                next_action="final_answer"),
        )
    if (decision.decision == Decision.call_tool
            and decision.next_tool
            and state.get("tool_history")
            and not state.get("information_gap")
            and decision.next_tool in {
                "search_quebec_jurisprudence", "search_quebec_regulations",
                "semantic_search_ccq", "semantic_search_cpc",
            }):
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            decision=Decision.final_answer,
            thinking_text="Aucune lacune d'information explicite n'autorise une recherche supplémentaire.",
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="gap explicite absent",
                next_action="final_answer"),
        )
    return decision


def normative_references(observations: Iterable[ToolObservation]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for sid, (text, _number, _index) in retrieved_articles(observations).items():
        for match in _REGULATION_RE.finditer(text):
            refs.append({
                "reference_type": "regulation",
                "reference_text": match.group(0).strip(),
                "target_title": "",
                "source_id": sid,
                "status": "unresolved",
            })
    return refs


def decide_source_sufficiency(
    selection: PrimaryAuthoritySelection,
    rule_contract: RuleContract,
    observations: Iterable[ToolObservation],
    *, task_id: str = "",
    jurisprudence_requested: bool = False,
) -> SourceSufficiencyDecision:
    refs = normative_references(observations)
    if not selection.primary_sources:
        legislation = "missing"
    elif rule_contract.elements:
        legislation = "sufficient"
    else:
        legislation = "required"
    regulation = "required" if refs else "not_needed"
    if refs:
        missing_questions = ["Quel règlement ou quelle annexe est visé par le renvoi explicite de la source récupérée?"]
    else:
        missing_questions = list(rule_contract.decisive_facts_needed)
    if jurisprudence_requested:
        jurisprudence = "required"
    else:
        jurisprudence = "not_needed" if legislation == "sufficient" else "conditionally_required"
    return SourceSufficiencyDecision(
        task_id=task_id,
        sufficient_for_initial_answer=legislation == "sufficient" and not refs,
        legislation_status=legislation,
        regulation_status=regulation,
        jurisprudence_status=jurisprudence,
        doctrine_status="not_needed",
        missing_questions=missing_questions,
        explicit_normative_references=[r["reference_text"] for r in refs],
        reason=("La législation récupérée contient les éléments nécessaires à une première réponse conditionnelle."
                if legislation == "sufficient" else "Une source ou un élément opérant reste à récupérer."),
    )


def build_claim_ledger(answer: str, selection: PrimaryAuthoritySelection,
                       source_texts: dict[str, str], *, task_id: str = "") -> ClaimLedger:
    claims: list[LegalClaim] = []
    for index, paragraph in enumerate(re.split(r"(?<=[.!?])\s+|\n+", answer or "")):
        text = paragraph.strip()
        if not text or not _LEGAL_MARKER_RE.search(text):
            continue
        cited = set(_ARTICLE_RE.findall(text))
        source_ids = [sid for sid in selection.primary_sources
                      if sid in source_texts and (not cited or sid.rsplit(":", 1)[-1] in cited)]
        if not source_ids:
            claims.append(LegalClaim(
                claim_id=f"claim-{index}", text=text,
                support_type="unsupported", verification_status="failed",
                failure_reason="aucune source principale récupérée ne soutient cette affirmation",
                task_id=task_id))
            continue
        direct = any(_fold(text[:80]) in _fold(source_texts[sid]) for sid in source_ids)
        claims.append(LegalClaim(
            claim_id=f"claim-{index}", text=text, source_ids=source_ids,
            support_type="direct" if direct else "reasonable_inference",
            verification_status="verified", task_id=task_id))
    return ClaimLedger(task_id=task_id, claims=claims)


def merge_failures(previous: list[dict[str, Any]], new: list[dict[str, Any]], *, node: str = "") -> list[dict[str, Any]]:
    """Append-only deterministic merge; resolved failures remain auditable."""
    merged = [dict(item) for item in previous or []]
    seen = {(str(item.get("failure_type", "")), str(item.get("claim", "")), str(item.get("reason", ""))) for item in merged}
    for item in new or []:
        value = dict(item)
        value.setdefault("status", "open")
        value.setdefault("resolved_at_node", "")
        key = (str(value.get("failure_type", "")), str(value.get("claim", "")), str(value.get("reason", "")))
        if key not in seen:
            merged.append(value)
            seen.add(key)
    return merged


def normalize_and_repair_tool_args(catalog: Any, tool: str, arguments: Any,
                                   *, active_task: dict[str, Any] | None = None,
                                   latest_user_message: str = "") -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Normalize planner arguments once, including reconstructable searches."""
    raw = dict(arguments) if isinstance(arguments, dict) else {}
    spec = getattr(catalog, "tools", {}).get(tool)
    if spec is not None:
        cleaned = {key: value for key, value in raw.items() if key in spec.properties}
    else:
        cleaned = raw
    removed = [key for key in raw if key not in cleaned]
    if tool in {"semantic_search_ccq", "semantic_search_cpc"} and not str(cleaned.get("query") or "").strip():
        task = active_task or {}
        query = str(task.get("normalized_query") or latest_user_message or task.get("summary") or "").strip()
        if query:
            cleaned["query"] = query
            repaired = ["query"]
        else:
            repaired = []
    else:
        repaired = []
    errors = list(catalog.validate_call(tool, cleaned)) if hasattr(catalog, "validate_call") else []
    audit = {"tool": tool, "removed_fields": removed, "repaired_fields": repaired,
             "remaining_arguments": cleaned, "errors": errors}
    return cleaned, audit, errors
