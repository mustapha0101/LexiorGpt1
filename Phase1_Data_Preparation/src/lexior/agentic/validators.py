# -*- coding: utf-8 -*-
"""Validations déterministes finales et métriques agentiques."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .schemas import Decision, ExpectedRoute, PlannerDecision, Role, RoutePolicy, TrainingTrajectory
from .taxonomy import REQUEST_TYPES, NO_JURISPRUDENCE, OFFICIAL_TEXT_REQUIRED
from .tool_catalog import ToolCatalog

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TEMP_PATH_RE = re.compile(r"workspaceStorage|content\.txt|(?:[A-Za-z]:\\[^\s]+)", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s<>\]\[\"')]+")
CITATION_MARK_RE = re.compile(r"\[\^?\d+\]|\b\d{4}\s+(?:SCC|CSC|FC|CF|FCA|CAF|QCCA|QCCS|QCCQ)\s+\d+\b")
ARTICLE_CITATION_RE = re.compile(r"\barticle\s+(\d{1,4}(?:\.\d+)?)\b", re.IGNORECASE)
CERTAINTY_RE = re.compile(r"\b(?:certainement|sans aucun doute|garanti|indiscutablement|assurément)\b", re.I)
PRECISE_ARTICLE_TOOLS = {
    "exact_text_retrieval": ("get_ccq_articles", "get_cpc_articles"),
}

# Backward compat
CATEGORIES = REQUEST_TYPES

# ── Jurisprudence query quality ──────────────────────────────────────────

_CONVERSATIONAL_PATTERNS = re.compile(
    r"\b(?:aimerais|voudrais|comprendre|expliquer|savoir|connaître|"
    r"s'il\s+vous\s+plaît|svp|merci|bonjour|pouvez-vous)\b",
    re.IGNORECASE,
)
_LEGAL_TERM_RE = re.compile(
    r"\b(?:responsabilité|obligation|contrat|préjudice|dommage|recours|"
    r"prescription|garantie|vice\s+caché|résolution|réduction|inexécution|"
    r"injonction|signification|appel|exécution|saisie|copropriété|"
    r"hypothèque|servitude|succession|testament|bail|louage|mandat|"
    r"tutelle|curatelle|assurance|faillite|divorce|pension|"
    r"droit\s+de\s+propriété|usufruit|copropriété|donation|"
    r"vente|achat|vendeur|acheteur|locataire|propriétaire|employeur|"
    r"salarié|congédiement|harcèlement|discrimination|Québec|fédéral)\b",
    re.IGNORECASE,
)


def validate_jurisprudence_query(query: str) -> list[str]:
    """Check that a jurisprudence search query is structured enough."""
    warnings: list[str] = []
    if not query or len(query.strip()) < 10:
        warnings.append("requête de jurisprudence trop courte")
        return warnings

    words = query.split()
    # Strip conversational fillers
    conversational_count = len(_CONVERSATIONAL_PATTERNS.findall(query))
    if conversational_count > 2:
        warnings.append(
            "requête de jurisprudence contient trop de formulations conversationnelles")

    legal_terms = _LEGAL_TERM_RE.findall(query)
    article_refs = ARTICLE_CITATION_RE.findall(query)

    if not legal_terms and not article_refs:
        warnings.append(
            "requête de jurisprudence sans terme juridique ni référence d'article")

    if len(words) < 4 and not article_refs:
        warnings.append("requête de jurisprudence trop générique")

    return warnings


# ── Search result classification ─────────────────────────────────────────

_CASE_CITATION_RE = re.compile(
    r"\b\d{4}\s+(?:QCCA|QCCS|QCCQ|QCTDP|QCRDL|SCC|CSC|FC|CF|FCA|CAF)\s+\d+\b"
)
_CASE_NAME_RE = re.compile(
    r"[A-ZÀ-Ÿ][\w'-]+\s+c\.\s+[A-ZÀ-Ÿ][\w'-]+"
)


def classify_search_result(
    tool_name: str,
    response: str,
    ok: bool,
    error: str | None = None,
) -> str:
    """Classify a tool result into one of the SearchResultStatus values."""
    if not ok:
        if error:
            return "tool_error"
        return "tool_error"

    stripped = (response or "").strip()
    if not stripped or stripped in ("[]", "{}", "null"):
        return "empty"

    if tool_name == "search_quebec_jurisprudence":
        has_citation = bool(_CASE_CITATION_RE.search(stripped))
        has_case_name = bool(_CASE_NAME_RE.search(stripped))
        if not has_citation and not has_case_name:
            if any(kw in stripped.lower() for kw in (
                "loi sur", "règlement sur", "code civil", "code de procédure"
            )):
                return "wrong_document_type"
            return "irrelevant"

    if len(stripped) < 20:
        return "empty"

    return "usable"


# ── Tool sequence logic ──────────────────────────────────────────────────

ARTICLE_FETCH_TOOLS = {"get_ccq_articles", "get_cpc_articles"}
JURISPRUDENCE_TOOLS = {"search_quebec_jurisprudence"}
SEMANTIC_SEARCH_TOOLS = {"semantic_search_ccq", "semantic_search_cpc"}


def validate_tool_sequence_logic(
    request_type: str, sequence: list[str],
) -> list[str]:
    """Check logical ordering of tool calls.

    For Quebec civil/procedural law:
    - Official article text should be retrieved before jurisprudence
      (unless the request is specifically about case law)
    - Semantic search should come before article retrieval when
      the article number is unknown
    """
    warnings: list[str] = []

    if request_type in ("case_law_research", "non_legal", "dataset_coverage",
                        "comparative_law"):
        return warnings

    jurisprudence_indices = [
        i for i, t in enumerate(sequence) if t in JURISPRUDENCE_TOOLS
    ]
    article_indices = [
        i for i, t in enumerate(sequence) if t in ARTICLE_FETCH_TOOLS
    ]

    if jurisprudence_indices and article_indices:
        first_jurisprudence = min(jurisprudence_indices)
        first_article = min(article_indices)
        if first_jurisprudence < first_article:
            warnings.append(
                "séquence: jurisprudence recherchée avant récupération du texte officiel")

    return warnings

# --------------- language mismatch detection ---------------

_CJK_RANGES = (
    ("一", "鿿"),
    ("㐀", "䶿"),
    ("぀", "ゟ"),
    ("゠", "ヿ"),
    ("가", "힯"),
)

_FOREIGN_SCRIPT_CATEGORIES = {"CYRILLIC", "ARABIC", "DEVANAGARI", "THAI", "GREEK", "HEBREW"}

_STOPWORDS: dict[str, set[str]] = {
    "fr": {"le", "la", "les", "de", "des", "du", "un", "une", "est", "sont",
            "en", "au", "aux", "ce", "cette", "il", "elle", "nous", "vous",
            "qui", "que", "dans", "pour", "par", "sur", "avec", "pas", "ne",
            "plus", "ou", "et", "mais", "donc", "car", "entre", "après",
            "avant", "selon", "lors", "ses", "leur", "leurs", "être", "avoir",
            "fait", "peut", "doit", "soit"},
    "en": {"the", "of", "and", "to", "in", "is", "it", "that", "was", "for",
            "on", "are", "with", "as", "at", "be", "this", "have", "from",
            "or", "an", "by", "not", "but", "what", "all", "were", "when",
            "can", "there", "their", "which", "do", "if", "will", "has",
            "been", "would", "could", "should", "may"},
}

_STRIP_RE = re.compile(r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>|https?://\S+|```.*?```", re.DOTALL)


def _detect_language_mismatch(text: str, expected_lang: str) -> Optional[str]:
    clean = _STRIP_RE.sub(" ", text)
    letters = [ch for ch in clean if ch.isalpha()]
    if len(letters) < 30:
        return None
    cjk_count = sum(
        1 for ch in letters
        if any(lo <= ch <= hi for lo, hi in _CJK_RANGES)
    )
    if cjk_count / len(letters) > 0.05:
        return "language mismatch : contenu CJK détecté"
    foreign_script_count = 0
    for ch in letters:
        script = unicodedata.name(ch, "").split()[0]
        if script in _FOREIGN_SCRIPT_CATEGORIES:
            foreign_script_count += 1
    if foreign_script_count / len(letters) > 0.10:
        return "language mismatch : écriture non latine détectée"
    words = re.findall(r"[a-zà-ÿœæ]+", clean.lower())
    if len(words) < 15:
        return None
    expected_stops = _STOPWORDS.get(expected_lang, set())
    other_langs = {k: v for k, v in _STOPWORDS.items() if k != expected_lang}
    expected_hits = sum(1 for w in words if w in expected_stops)
    for lang, stops in other_langs.items():
        foreign_hits = sum(1 for w in words if w in stops)
        if foreign_hits > expected_hits and foreign_hits / len(words) > 0.08:
            return f"language mismatch : langue détectée '{lang}' au lieu de '{expected_lang}'"
    return None


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


def validate_tool_route(
    request_type: str,
    sequence: list[str],
    exempt_tools: Optional[list[str]] = None,
) -> list[str]:
    rt = REQUEST_TYPES.get(request_type)
    if rt is None:
        return [f"type de demande inconnu pour le routage : {request_type}"]
    exempt = set(exempt_tools or [])
    policy = rt.route_policy
    if policy.required_capabilities:
        return _validate_capability_route(policy, rt.expected_route, sequence,
                                          exempt=exempt)
    expected = rt.expected_route
    allowed = set(expected.allowed_tools())
    required = expected.required_tools()
    errors: list[str] = []
    if expected.no_tool and sequence:
        errors.append("outil interdit pour une demande sans recherche")
        return errors
    unexpected = [tool for tool in sequence if tool not in allowed]
    if unexpected:
        errors.append(f"outil hors route attendue : {unexpected}")
    missing = [tool for tool in required
               if tool not in sequence and tool not in exempt]
    if missing:
        errors.append(f"outil requis absent de la route : {missing}")
    positions = [sequence.index(tool) for tool in required if tool in sequence]
    if positions != sorted(positions):
        errors.append("ordre des outils requis incorrect")
    return errors


def _validate_capability_route(
    policy: RoutePolicy,
    expected: ExpectedRoute,
    sequence: list[str],
    exempt: Optional[set[str]] = None,
) -> list[str]:
    errors: list[str] = []
    if policy.no_tool and sequence:
        errors.append("outil interdit pour une demande sans recherche")
        return errors
    forbidden_used = [t for t in sequence if t in policy.forbidden_tools]
    if forbidden_used:
        errors.append(f"outil interdit par la politique : {forbidden_used}")
    required = expected.required_tools()
    exempt = exempt or set()
    missing = [t for t in required
               if t not in sequence and t not in exempt]
    if missing:
        errors.append(f"outil requis absent de la route : {missing}")
    positions = [sequence.index(t) for t in required if t in sequence]
    if positions != sorted(positions):
        errors.append("ordre des outils requis incorrect")
    return errors


def validate_next_action(request_type: str, tool: str) -> list[str]:
    rt = REQUEST_TYPES.get(request_type)
    if rt is None:
        return [f"type de demande inconnu : {request_type}"]
    policy = rt.route_policy
    errors: list[str] = []
    if policy.no_tool:
        errors.append("outil interdit pour cette demande")
    if not policy.allows_tool(tool):
        errors.append(f"outil {tool} interdit par la politique de routage")
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
                        seen_fingerprints: Optional[set[str]] = None,
                        exempt_tools: Optional[list[str]] = None) -> ValidationResult:
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
        tool_messages = [m for m in trajectory.messages
                         if m.role == Role.tool and m.name == observation.tool_name]
        if not any(m.content == observation.normalized_response for m in tool_messages):
            errors.append(f"réponse MCP de {observation.tool_name} absente des messages")

    sequence = [name for name, _ in observed_pairs]
    errors.extend(validate_tool_route(trajectory.request_type, sequence,
                                       exempt_tools=exempt_tools))
    call_keys = [json.dumps([name, args], sort_keys=True, ensure_ascii=False)
                 for name, args in observed_pairs]
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

    article_tools = PRECISE_ARTICLE_TOOLS.get(trajectory.request_type)
    if article_tools:
        official = next(
            (o.normalized_response.strip()
             for o in reversed(trajectory.tool_trace)
             if o.tool_name in article_tools and o.ok and
             not o.truncated and o.normalized_response.strip()),
            "",
        )
        if not official:
            errors.append("texte officiel complet de l'article indisponible")
        elif final != official:
            errors.append("texte d'article précis non reproduit intégralement mot pour mot")
    elif trajectory.request_type == "article_explanation":
        official = next(
            (o.normalized_response.strip()
             for o in reversed(trajectory.tool_trace)
             if o.tool_name in ("get_ccq_articles", "get_cpc_articles")
             and o.ok and not o.truncated and o.normalized_response.strip()),
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
    if CERTAINTY_RE.search(final) and (
        not trajectory.tool_trace
        or any(o.error or o.truncated for o in trajectory.tool_trace)
    ):
        errors.append("certitude non justifiée par les preuves disponibles")

    citable_observations = [
        o for o in trajectory.tool_trace
        if o.tool_name not in {"semantic_search_ccq", "semantic_search_cpc"}
    ]
    available_urls = {u.rstrip(".,;)") for o in citable_observations
                      for u in o.source_urls}
    final_urls = {u.rstrip(".,;)") for u in URL_RE.findall(final)}
    invented_urls = final_urls - available_urls
    if invented_urls:
        errors.append(f"URL absente des réponses d'outils : {sorted(invented_urls)}")
    available_citations = {c.casefold() for o in citable_observations
                           for c in o.citations}
    for citation in CITATION_MARK_RE.findall(final):
        if citation.startswith("["):
            if not final_urls:
                errors.append(f"citation {citation} sans source récupérée")
        elif citation.casefold() not in available_citations:
            errors.append(f"citation absente des réponses d'outils : {citation}")
    evidence_text = "\n".join(
        o.normalized_response for o in citable_observations
    ).casefold()
    for article in ARTICLE_CITATION_RE.findall(final):
        if not re.search(rf"\barticle\s+{re.escape(article)}\b",
                         evidence_text, re.IGNORECASE):
            errors.append(f"article {article} absent des réponses d'outils")

    assistant_text = "\n".join(
        m.content for m in trajectory.messages if m.role == Role.assistant
    )
    lang_err = _detect_language_mismatch(assistant_text, trajectory.language)
    if lang_err:
        errors.append(lang_err)

    # --- v2.0 structural checks ---
    assistant_messages = [m for m in trajectory.messages
                          if m.role == Role.assistant]
    for idx, m in enumerate(assistant_messages):
        tc_payload, _ = _parse_tool_call(m.content)
        has_tool_call = tc_payload is not None
        has_substantial = len(m.content.replace(
            TOOL_CALL_RE.sub("", m.content) if not has_tool_call
            else "", "").strip()) > 100
        if has_tool_call and has_substantial and idx < len(assistant_messages) - 1:
            warnings.append(f"appel d'outil et texte substantiel au message assistant {idx}")

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
    return ValidationResult(
        valid=not errors, errors=list(dict.fromkeys(errors)), warnings=warnings)


def _fingerprint(trajectory: TrainingTrajectory) -> str:
    user = next((m.content for m in trajectory.messages
                 if m.role == Role.user), "")
    normalized = re.sub(r"\W+", " ", user.casefold()).strip()
    return normalized + "|" + ",".join(o.tool_name for o in trajectory.tool_trace)


def compute_metrics(rows: Iterable[TrainingTrajectory],
                    catalog: ToolCatalog) -> dict[str, float | dict]:
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
        rt = REQUEST_TYPES.get(row.request_type)
        expected = rt.expected_route.required_tools() if rt else []
        allowed = set(rt.expected_route.allowed_tools()) if rt else set(expected)
        expected_set, actual_set = set(expected), set(actual)
        totals["tool_names"] += sum(
            1 for o in calls if o.tool_name in catalog.tools)
        totals["tool_args"] += sum(
            1 for o in calls
            if not catalog.validate_call(o.tool_name, o.arguments))
        totals["tool_calls"] += len(calls)
        totals["final"] += bool(row.final_answer().strip())
        totals["grounded"] += sum(1 for g in row.grounding if g.content_hash)
        totals["grounding"] += len(row.grounding)
        totals["fabricated"] += sum(
            1 for o in calls if not o.mock and o.raw_response is None and o.ok)
        totals["legal_accept"] += (row.quality.legal_critic_score or 0) >= 0.7
        totals["agentic_accept"] += (row.quality.agentic_critic_score or 0) >= 0.7
        totals["route_correct"] += (
            all(t in actual for t in expected)
            and all(t in allowed for t in actual))
        totals["sequence_exact"] += actual == expected
        totals["sequence_tp"] += len(actual_set & expected_set)
        totals["sequence_pred"] += len(actual_set)
        totals["sequence_expected"] += len(expected_set)
        totals["unnecessary"] += len(actual_set - allowed)
        totals["missing"] += len(expected_set - actual_set)
        expected_clarification = bool(
            rt and rt.expected_route.requires_clarification)
        actual_clarification = any(
            m.role == Role.assistant and m.content.rstrip().endswith("?")
            for m in row.messages[:3])
        totals["clarification_correct"] += (
            expected_clarification == actual_clarification)
        totals["stopped_correctly"] += (
            len(calls) <= 4
            and not any("boucle" in e for e in result.errors))
        totals["loops"] += any("boucle" in e for e in result.errors)
        totals["lang_mismatch"] += any(
            "language mismatch" in e for e in result.errors)
        totals["unsupported"] += any(
            "absente des réponses" in e or "URL absente" in e
            for e in result.errors)
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
        "language_mismatch_rate": totals["lang_mismatch"] / n,
        "unsupported_claim_rate": totals["unsupported"] / n,
        "legal_critic_acceptance_rate": totals["legal_accept"] / n,
        "agentic_critic_acceptance_rate": totals["agentic_accept"] / n,
        "rejection_reasons": dict(reasons),
    }
