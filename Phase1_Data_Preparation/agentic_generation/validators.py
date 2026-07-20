# -*- coding: utf-8 -*-
"""Validations déterministes finales et métriques agentiques."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .schemas import Decision, PlannerDecision, Role, TrainingTrajectory
from .taxonomy import CATEGORIES, NO_JURISPRUDENCE, OFFICIAL_TEXT_REQUIRED
from .tool_catalog import ToolCatalog

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TEMP_PATH_RE = re.compile(r"workspaceStorage|content\.txt|(?:[A-Za-z]:\\[^\s]+)", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s<>\]\[\"')]+")
CITATION_MARK_RE = re.compile(r"\[\^?\d+\]|\b\d{4}\s+(?:SCC|CSC|FC|CF|FCA|CAF|QCCA|QCCS|QCCQ)\s+\d+\b")
ARTICLE_CITATION_RE = re.compile(r"\barticle\s+(\d{1,4}(?:\.\d+)?)\b", re.IGNORECASE)
CERTAINTY_RE = re.compile(r"\b(?:certainement|sans aucun doute|garanti|indiscutablement|assurément)\b", re.I)
PRECISE_ARTICLE_TOOLS = {
    "article_ccq_precis": "get_ccq_articles",
    "article_cpc_precis": "get_cpc_articles",
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_planner_decision(decision: PlannerDecision, catalog: ToolCatalog) -> list[str]:
    errors: list[str] = []
    if decision.decision == Decision.call_tool:
        if not decision.next_tool:
            errors.append("decision call_tool sans next_tool")
        else:
            errors.extend(catalog.validate_call(decision.next_tool, decision.arguments))
    elif decision.next_tool:
        errors.append(f"next_tool présent pour la décision {decision.decision.value}")
    if decision.decision == Decision.ask_clarification and not decision.clarification_question:
        errors.append("demande de clarification vide")
    return errors


def validate_tool_route(request_type: str, sequence: list[str]) -> list[str]:
    """Valide la route après décision, sans jamais la montrer au Planner."""
    category = CATEGORIES.get(request_type)
    if category is None:
        return [f"catégorie inconnue pour le routage : {request_type}"]
    expected = category.expected_route
    allowed = set(expected.allowed_tools())
    required = expected.required_tools()
    errors: list[str] = []
    if expected.no_tool and sequence:
        errors.append("outil interdit pour une demande sans recherche")
        return errors
    unexpected = [tool for tool in sequence if tool not in allowed]
    if unexpected:
        errors.append(f"outil hors route attendue : {unexpected}")
    missing = [tool for tool in required if tool not in sequence]
    if missing:
        errors.append(f"outil requis absent de la route : {missing}")
    positions = [sequence.index(tool) for tool in required if tool in sequence]
    if positions != sorted(positions):
        errors.append("ordre des outils requis incorrect")
    return errors


def _parse_tool_call(content: str) -> tuple[Optional[dict], list[str]]:
    errors: list[str] = []
    openings = content.count("<tool_call>")
    closings = content.count("</tool_call>")
    if openings != closings:
        errors.append("balises tool_call mal fermées")
        return None, errors
    if "<tool_response>" in content or "</tool_response>" in content:
        errors.append("tool_response produit par l'assistant")
    match = TOOL_CALL_RE.search(content)
    if not match:
        return None, errors
    try:
        payload = json.loads(match.group(1))
    except ValueError as exc:
        return None, [f"JSON tool_call invalide : {exc}"]
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        errors.append("tool_call doit contenir exactement name et arguments")
    return payload, errors


def validate_trajectory(trajectory: TrainingTrajectory, catalog: ToolCatalog,
                        allow_mock: bool = False, max_tool_calls: int = 4,
                        seen_fingerprints: Optional[set[str]] = None) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    pending: list[dict] = []
    observed_pairs: list[tuple[str, dict]] = []

    for index, message in enumerate(trajectory.messages):
        if TEMP_PATH_RE.search(message.content):
            errors.append(f"chemin local temporaire au message {index}")
        payload = None
        if message.role == Role.assistant:
            payload, tag_errors = _parse_tool_call(message.content)
            errors.extend(f"message {index}: {err}" for err in tag_errors)
        if message.role == Role.assistant and payload:
            name = payload.get("name", "")
            args = payload.get("arguments", {})
            errors.extend(catalog.validate_call(name, args))
            pending.append(payload)

        if message.role == Role.tool:
            if not pending:
                errors.append(f"tool message {index} sans tool_call")
                continue
            call = pending.pop(0)
            if message.name != call.get("name"):
                errors.append(f"outil du message {index} différent du tool_call")
            observed_pairs.append((message.name or "", call.get("arguments", {})))
    if pending:
        errors.append("tool_call sans tool message correspondant")

    if len(observed_pairs) > max_tool_calls:
        errors.append("nombre maximal d'appels dépassé")
    if len(observed_pairs) != len(trajectory.tool_trace):
        errors.append("tool_trace ne correspond pas aux messages tool")

    for idx, observation in enumerate(trajectory.tool_trace):
        if observation.mock and not allow_mock:
            errors.append(f"réponse MCP fabriquée/mock interdite à l'observation {idx}")
        if observation.content_hash == "":
            errors.append(f"observation {idx} sans content_hash")
        if observation.ok and observation.raw_response is None:
            errors.append(f"réponse MCP fabriquée à l'observation {idx}")
        if idx < len(observed_pairs):
            pair = observed_pairs[idx]
            if pair != (observation.tool_name, observation.arguments):
                errors.append(f"observation {idx} non liée à son tool_call")
        tool_messages = [m for m in trajectory.messages if m.role == Role.tool and m.name == observation.tool_name]
        if not any(m.content == observation.normalized_response for m in tool_messages):
            errors.append(f"réponse MCP de {observation.tool_name} absente des messages")

    sequence = [name for name, _ in observed_pairs]
    errors.extend(validate_tool_route(trajectory.request_type, sequence))
    # Une répétition strictement identique est une boucle, sauf pagination fetch_document.
    call_keys = [json.dumps([name, args], sort_keys=True, ensure_ascii=False) for name, args in observed_pairs]
    repeats = [k for k, count in Counter(call_keys).items() if count > 1]
    if repeats:
        errors.append("boucle d'appels identiques")
    if trajectory.request_type in NO_JURISPRUDENCE and "search_quebec_jurisprudence" in sequence:
        errors.append("recherche de jurisprudence inutile pour cette demande")
    if trajectory.request_type in OFFICIAL_TEXT_REQUIRED and not sequence:
        errors.append("réponse de mémoire à une demande de texte officiel")

    final = trajectory.final_answer().strip()
    if not final:
        errors.append("réponse finale vide")
    article_tool = PRECISE_ARTICLE_TOOLS.get(trajectory.request_type)
    if article_tool:
        official = next(
            (observation.normalized_response.strip()
             for observation in reversed(trajectory.tool_trace)
             if observation.tool_name == article_tool and observation.ok and
             not observation.truncated and observation.normalized_response.strip()),
            "",
        )
        if not official:
            errors.append("texte officiel complet de l'article indisponible")
        elif final != official:
            errors.append("texte d'article précis non reproduit intégralement mot pour mot")
    elif trajectory.request_type == "explication_article":
        official = next(
            (observation.normalized_response.strip()
             for observation in reversed(trajectory.tool_trace)
             if observation.tool_name == "get_ccq_articles" and observation.ok and
             not observation.truncated and observation.normalized_response.strip()),
            "",
        )
        if not official:
            errors.append("texte officiel complet de l'article indisponible")
        elif official not in final:
            errors.append("explication sans reproduction intégrale du texte officiel")
    if final.startswith("{") and final.endswith("}"):
        try:
            wrapped = json.loads(final)
        except ValueError:
            wrapped = None
        if isinstance(wrapped, dict) and "answer" in wrapped:
            errors.append("réponse finale enveloppée dans un objet JSON")
    if CERTAINTY_RE.search(final) and (not trajectory.tool_trace or any(o.error or o.truncated for o in trajectory.tool_trace)):
        errors.append("certitude non justifiée par les preuves disponibles")

    citable_observations = [
        observation for observation in trajectory.tool_trace
        if observation.tool_name not in {"semantic_search_ccq", "semantic_search_cpc"}
    ]
    available_urls = {u.rstrip(".,;)") for o in citable_observations for u in o.source_urls}
    final_urls = {u.rstrip(".,;)") for u in URL_RE.findall(final)}
    invented_urls = final_urls - available_urls
    if invented_urls:
        errors.append(f"URL absente des réponses d'outils : {sorted(invented_urls)}")
    available_citations = {c.casefold() for o in citable_observations for c in o.citations}
    for citation in CITATION_MARK_RE.findall(final):
        if citation.startswith("["):
            # Les appels de note sont permis uniquement si une URL récupérée suit dans la réponse.
            if not final_urls:
                errors.append(f"citation {citation} sans source récupérée")
        elif citation.casefold() not in available_citations:
            errors.append(f"citation absente des réponses d'outils : {citation}")
    evidence_text = "\n".join(
        o.normalized_response for o in citable_observations
    ).casefold()
    for article in ARTICLE_CITATION_RE.findall(final):
        if not re.search(rf"\barticle\s+{re.escape(article)}\b", evidence_text, re.IGNORECASE):
            errors.append(f"article {article} absent des réponses d'outils")

    resolved = trajectory.resolved_jurisdiction.casefold()
    expected = trajectory.expected_jurisdiction.casefold()
    if expected and expected not in ("indéterminée", "sans objet") and resolved:
        if "fédéral" in expected and "québec" in resolved and "fédéral" not in resolved:
            errors.append("juridiction contradictoire")
        if expected == "québec" and "fédéral" in resolved:
            errors.append("juridiction contradictoire")

    fingerprint = _fingerprint(trajectory)
    if seen_fingerprints is not None:
        tokens = set(fingerprint.split())
        near_duplicate = False
        for previous in seen_fingerprints:
            other = set(previous.split())
            union = tokens | other
            if union and len(tokens & other) / len(union) >= 0.90:
                near_duplicate = True
                break
        if fingerprint in seen_fingerprints:
            errors.append("doublon exact")
        elif near_duplicate:
            errors.append("quasi-duplicat")
        else:
            seen_fingerprints.add(fingerprint)
    return ValidationResult(valid=not errors, errors=list(dict.fromkeys(errors)), warnings=warnings)


def _fingerprint(trajectory: TrainingTrajectory) -> str:
    user = next((m.content for m in trajectory.messages if m.role == Role.user), "")
    normalized = re.sub(r"\W+", " ", user.casefold()).strip()
    return normalized + "|" + ",".join(o.tool_name for o in trajectory.tool_trace)


def compute_metrics(rows: Iterable[TrainingTrajectory], catalog: ToolCatalog) -> dict[str, float | dict]:
    from .taxonomy import CATEGORIES
    rows = list(rows)
    if not rows:
        return {}
    totals = Counter()
    reasons = Counter()
    for row in rows:
        result = validate_trajectory(row, catalog, allow_mock=True)
        for error in result.errors:
            reasons[error] += 1
        calls = row.tool_trace
        actual = [o.tool_name for o in calls]
        category = CATEGORIES.get(row.request_type)
        expected = category.expected_route.required_tools() if category else []
        allowed = set(category.expected_route.allowed_tools()) if category else set(expected)
        expected_set, actual_set = set(expected), set(actual)
        totals["tool_names"] += sum(1 for o in calls if o.tool_name in catalog.tools)
        totals["tool_args"] += sum(1 for o in calls if not catalog.validate_call(o.tool_name, o.arguments))
        totals["tool_calls"] += len(calls)
        totals["final"] += bool(row.final_answer().strip())
        totals["grounded"] += sum(1 for g in row.grounding if g.content_hash)
        totals["grounding"] += len(row.grounding)
        totals["fabricated"] += sum(1 for o in calls if not o.mock and o.raw_response is None and o.ok)
        totals["legal_accept"] += (row.quality.legal_critic_score or 0) >= 0.7
        totals["agentic_accept"] += (row.quality.agentic_critic_score or 0) >= 0.7
        totals["route_correct"] += all(tool in actual for tool in expected) and all(tool in allowed for tool in actual)
        totals["sequence_exact"] += actual == expected
        totals["sequence_tp"] += len(actual_set & expected_set)
        totals["sequence_pred"] += len(actual_set)
        totals["sequence_expected"] += len(expected_set)
        totals["unnecessary"] += len(actual_set - allowed)
        totals["missing"] += len(expected_set - actual_set)
        expected_clarification = bool(category and category.expected_route.requires_clarification)
        actual_clarification = any(m.role == Role.assistant and m.content.rstrip().endswith("?")
                                   for m in row.messages[:3])
        totals["clarification_correct"] += expected_clarification == actual_clarification
        totals["stopped_correctly"] += len(calls) <= 4 and not any("boucle" in e for e in result.errors)
        totals["loops"] += any("boucle" in e for e in result.errors)
        totals["unsupported"] += any("absente des réponses" in e or "URL absente" in e for e in result.errors)
        expected_j = row.expected_jurisdiction.casefold()
        resolved_j = row.resolved_jurisdiction.casefold()
        totals["jurisdiction_correct"] += (not expected_j or expected_j in {"indéterminée", "sans objet"}
                                             or ("fédéral" in expected_j) == ("fédéral" in resolved_j))
    n = len(rows)
    calls = totals["tool_calls"] or 1
    return {
        "tool_name_valid_rate": totals["tool_names"] / calls,
        "tool_arguments_valid_rate": totals["tool_args"] / calls,
        "final_answer_presence_rate": totals["final"] / n,
        "grounded_citation_rate": totals["grounded"] / (totals["grounding"] or 1),
        "MCP_response_fabrication_rate": totals["fabricated"] / calls,
        "route_accuracy": totals["route_correct"] / n,
        "tool_sequence_exact_match": totals["sequence_exact"] / n,
        "tool_sequence_precision": totals["sequence_tp"] / (totals["sequence_pred"] or 1),
        "tool_sequence_recall": totals["sequence_tp"] / (totals["sequence_expected"] or 1),
        "unnecessary_tool_call_rate": totals["unnecessary"] / calls,
        "missing_tool_call_rate": totals["missing"] / (totals["sequence_expected"] or 1),
        "clarification_accuracy": totals["clarification_correct"] / n,
        "stop_accuracy": totals["stopped_correctly"] / n,
        "loop_rate": totals["loops"] / n,
        "unsupported_claim_rate": totals["unsupported"] / n,
        "jurisdiction_accuracy": totals["jurisdiction_correct"] / n,
        "legal_critic_acceptance_rate": totals["legal_accept"] / n,
        "agentic_critic_acceptance_rate": totals["agentic_accept"] / n,
        "rejection_reasons": dict(reasons),
    }
