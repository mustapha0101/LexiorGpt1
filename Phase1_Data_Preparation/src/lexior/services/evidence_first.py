# -*- coding: utf-8 -*-
"""Generic evidence-first contracts shared by live and dataset runs.

The module never decides what a legal rule *should* contain from a catalogue
of legal concepts.  It projects retrieved official text into contracts and
keeps every operative proposition tied to an exact retrieved passage.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from lexior.agentic.schemas import (
    ClaimLedger,
    ClarificationDecision,
    LegalClaim,
    PlannerDecision,
    PrimaryAuthoritySelection,
    Decision,
    DecisionTrace,
    RuleContract,
    RuleElement,
    RuleFact,
    SourceRejection,
    SourceSufficiencyDecision,
    ToolObservation,
)


_REGULATION_RE = re.compile(
    r"[^.\n]{0,120}\b(?:prescrit|prévu|prevu|selon|déterminé|determine|"
    r"formulaire obligatoire|annexe|modalités|modalites)\b[^.\n]{0,180}\b"
    r"(?:règlement|reglement|annexe|formulaire)\b[^.\n]*", re.I)
_ARTICLE_RE = re.compile(r"\b(?:article|art\.)\s+(\d{1,4}(?:\.\d+)?)", re.I)
_LEGAL_MARKER_RE = re.compile(
    r"\b(?:article|articles|code civil|loi|règlement|reglement|doit|peut|"
    r"est tenu|a droit|prescription|délai|delai|responsabil|préjudice|"
    r"préjudice|dommage|réparer|reparer)\b", re.I)
_ARTICLE_BLOCK_RE = re.compile(
    r"(?ms)^\s*Article\s+(\d{1,4}(?:\.\d+)?)\s*\n(.*?)(?=^\s*Article\s+"
    r"\d{1,4}(?:\.\d+)?\s*\n|\Z)")
_CLAUSE_RE = re.compile(
    r"\b(?:si|lorsque|en cas de|du fait de|par le fait|sous réserve|"
    r"sous reserve|à moins que|a moins que|sauf)\b", re.I)
_EXCEPTION_RE = re.compile(r"\b(?:sauf|à moins que|a moins que|sans préjudice|"
                           r"sans prejudice|ne peut)\b", re.I)
_OPERATIVE_RE = re.compile(r"\b(?:doit|peut|est tenu|a droit|demander|"
                           r"réparer|reparer|interdit|obligatoire)\b", re.I)
_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "cette", "dans", "de", "des",
    "du", "elle", "en", "et", "il", "la", "le", "les", "leur", "leurs",
    "ne", "ou", "par", "pour", "qui", "que", "se", "son", "sur", "un",
    "une", "vous", "est", "sont", "doit", "peut", "pas", "the", "and",
}
_GENERIC_FACT_IDS = (
    "general_liability_basis", "fault", "causation", "prior_knowledge",
    "failure_to_take_reasonable_action", "damage_assessment",
)


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[\wÀ-ÿ]{3,}", _fold(value))
        if token not in _STOPWORDS
    }


def source_id(tool_name: str, article_number: str) -> str:
    prefix = ("ccq" if "ccq" in tool_name.casefold() else
              "cpc" if "cpc" in tool_name.casefold() else tool_name)
    return f"{prefix}:{article_number}"


def retrieved_articles(observations: Iterable[ToolObservation]) -> dict[str, tuple[str, str, int]]:
    """Return source_id -> (exact article text, label, observation index)."""
    result: dict[str, tuple[str, str, int]] = {}
    for index, observation in enumerate(observations):
        if observation.tool_name not in {"get_ccq_articles", "get_cpc_articles"} or not observation.ok:
            continue
        text = observation.normalized_response or ""
        numbers = list(dict.fromkeys(_ARTICLE_RE.findall(text)))
        if not numbers and isinstance(observation.arguments, dict):
            numbers = [str(raw) for raw in observation.arguments.get("articles", [])]
        blocks = re.split(r"(?=^\s*Article\s+\d)", text, flags=re.I | re.M)
        for number in numbers:
            match = next(
                (block for block in blocks if re.search(
                    rf"^\s*Article\s+{re.escape(number)}\b", block, re.I | re.M)),
                text,
            )
            result[source_id(observation.tool_name, number)] = (
                match.strip(), number, index)
    return result


def _case_text(case_description: str, facts: dict[str, Any] | None = None) -> str:
    values = [case_description]
    for key, value in (facts or {}).items():
        if key in {"user_statements", "entities", "case_entities"}:
            if key == "user_statements":
                values.extend(str(item) for item in value or [])
            continue
        if isinstance(value, dict):
            if value.get("value") is not None:
                values.append(str(value.get("value")))
        elif value not in (None, "", [], {}):
            values.append(str(value))
    return " ".join(values)


def _ranked_clauses(text: str) -> list[tuple[str, str]]:
    blocks = _ARTICLE_BLOCK_RE.findall(text or "")
    if not blocks:
        blocks = [("", text or "")]
    clauses: list[tuple[str, str]] = []
    for _number, body in blocks:
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", body):
            sentence = re.sub(r"^\s*[-–•]\s*", "", sentence).strip()
            if sentence and (_OPERATIVE_RE.search(sentence) or _CLAUSE_RE.search(sentence)):
                clauses.append((sentence, sentence))
    return clauses


def _element(source: str, text: str, kind: str, index: int) -> RuleElement:
    return RuleElement(
        id=f"{kind}_{index}",
        description=text,
        support_source_ids=[source],
        supporting_passages=[text],
        kind=kind,
        status="uncertain",
    )


def _source_bounded_contract_for_source(
    sid: str, text: str, facts: dict[str, Any]
) -> tuple[list[RuleElement], list[RuleElement], list[RuleElement], list[RuleElement], list[RuleElement], list[RuleElement], list[RuleFact], list[RuleFact], list[RuleFact], list[str]]:
    clauses = [sentence for _key, sentence in _ranked_clauses(text)]
    if not clauses and text.strip():
        clauses = [text.strip()[:1000]]
    all_elements: list[RuleElement] = []
    triggers: list[RuleElement] = []
    conditions: list[RuleElement] = []
    exceptions: list[RuleElement] = []
    consequences: list[RuleElement] = []
    subject: list[RuleElement] = []
    persons: list[RuleElement] = []
    for index, clause in enumerate(clauses):
        if _EXCEPTION_RE.search(clause):
            kind = "exception"
        elif _CLAUSE_RE.search(clause):
            kind = "trigger" if not _OPERATIVE_RE.search(clause) else "condition"
        elif _OPERATIVE_RE.search(clause):
            kind = "consequence"
        else:
            kind = "operative_text"
        item = _element(sid, clause, kind, index)
        all_elements.append(item)
        if kind == "trigger":
            triggers.append(item)
        elif kind == "condition":
            conditions.append(item)
        elif kind == "exception":
            exceptions.append(item)
        elif kind == "consequence":
            consequences.append(item)
        person_match = re.match(
            r"(.{2,140}?)\s+(?=(?:doit|peut|est tenu|a droit|demander|réparer|reparer)\b)",
            clause, re.I)
        if person_match:
            persons.append(_element(sid, person_match.group(1).strip(), "persons", index))
    if clauses:
        # These are source excerpts, not inferred legal categories.
        subject.append(_element(sid, clauses[0], "subject", 0))
    conditional_facts = [
        RuleFact(
            fact_id=f"conditional_{index}", description=item.description,
            source_ids=[sid], supporting_passages=item.supporting_passages,
            branches=[f"Si cette proposition s'applique : {item.description}"],
            status="conditional",
        )
        for index, item in enumerate([*triggers, *conditions])
    ]
    supporting_facts = [RuleFact(
        fact_id=f"supporting_{index}",
        description="Conserver les éléments de preuve liés à la proposition source.",
        source_ids=[sid], supporting_passages=item.supporting_passages,
        status="supporting",
    ) for index, item in enumerate(consequences)]
    # The extractor is deliberately conservative: it never turns generic
    # responsibility vocabulary into a condition.  Such legacy conditions
    # are recorded as not required unless the source explicitly contains a
    # matching proposition.
    source_folded = _fold(text)
    not_required = [
        fact_id for fact_id in _GENERIC_FACT_IDS
        if fact_id.replace("_", " ") not in source_folded
        and not (fact_id == "fault" and re.search(r"\bfaute\b", source_folded))
        and not (fact_id == "causation" and re.search(r"\bcausal", source_folded))
    ]
    return (all_elements, subject, persons, triggers, conditions, consequences,
            exceptions, conditional_facts, supporting_facts, not_required)


def _status_for_source(review: dict[str, Any], facts: dict[str, Any]) -> str:
    status = str(review.get("status", ""))
    return "present" if status == "applicable" else "uncertain"


def select_primary_authorities(
    observations: Iterable[ToolObservation],
    reviews: dict[str, dict[str, Any]],
    *, task_id: str = "", case_description: str = "",
    facts: dict[str, Any] | None = None,
    maximum_primary: int = 3,
    maximum_secondary: int = 3,
) -> PrimaryAuthoritySelection:
    """Rank by source/fact alignment; retrieval rank is only a weak signal."""
    query_tokens = _tokens(_case_text(case_description, facts))
    articles = retrieved_articles(observations)
    candidates: list[dict[str, Any]] = []
    for sid, (text, number, _index) in articles.items():
        review = reviews.get(number, {})
        status = str(review.get("status", ""))
        if status not in {"applicable", "conditionally_applicable", ""}:
            continue
        text_tokens = _tokens(text)
        overlap = len(query_tokens & text_tokens)
        directness = overlap / max(1, len(query_tokens))
        clauses = _ranked_clauses(text)
        score = (
            4.0 * directness
            + (1.0 if status == "applicable" else 0.5)
            + (0.35 if review.get("reviewed") else 0.0)
            + min(0.4, len(clauses) * 0.05)
            - min(0.3, float(review.get("rerank_rank", 10_000)) / 1000.0)
        )
        candidates.append({
            "source_id": sid, "number": number, "text": text,
            "review": review, "score": score, "overlap": overlap,
            "directness": directness,
            "completeness": bool(text.strip()) and not review.get("truncated", False),
            "rank": int(review.get("rerank_rank", 10_000)),
        })
    candidates.sort(key=lambda item: (
        -item["score"], -item["directness"], -int(item["completeness"]),
        item["rank"], -len(item["text"])))
    if candidates:
        # One source is the normal policy.  Add a second/third source only
        # when it materially aligns with the same facts and is complementary.
        primary = [candidates[0]["source_id"]]
        for candidate in candidates[1:maximum_primary]:
            if candidate["directness"] > 0 and candidate["score"] >= candidates[0]["score"] * 0.75:
                primary.append(candidate["source_id"])
    else:
        primary = []
    primary_set = set(primary)
    secondary = [item["source_id"] for item in candidates
                 if item["source_id"] not in primary_set][:maximum_secondary]
    selected = primary_set | set(secondary)
    rejected = []
    for sid, (_text, number, _index) in articles.items():
        if sid not in selected:
            rejected.append(SourceRejection(
                source_id=sid,
                reason=str(reviews.get(number, {}).get("reason")
                           or "alignement source-faits inférieur; non autorisée")))
    return PrimaryAuthoritySelection(
        task_id=task_id, primary_sources=primary, secondary_sources=secondary,
        rejected_sources=rejected,
        selection_reason=("Classement par correspondance entre les propositions "
                          "du texte récupéré et les faits de la tâche; le rang "
                          "de retrieval est secondaire."),
        confidence=(min(1.0, max(0.0, candidates[0]["directness"]))
                    if candidates else None),
    )


def build_rule_contract(
    selection: PrimaryAuthoritySelection,
    observations: Iterable[ToolObservation],
    reviews: dict[str, dict[str, Any]],
    facts: dict[str, Any],
    *, task_id: str = "",
) -> RuleContract:
    """Extract only propositions and explicit branches present in sources."""
    articles = retrieved_articles(observations)
    elements: list[RuleElement] = []
    subject: list[RuleElement] = []
    persons: list[RuleElement] = []
    triggers: list[RuleElement] = []
    conditions: list[RuleElement] = []
    consequences: list[RuleElement] = []
    exceptions: list[RuleElement] = []
    conditional_facts: list[RuleFact] = []
    supporting_facts: list[RuleFact] = []
    not_required: list[str] = []
    for sid in selection.primary_sources:
        if sid not in articles:
            continue
        text, number, _index = articles[sid]
        (source_elements, source_subject, source_persons, source_triggers,
         source_conditions, source_consequences, source_exceptions,
         source_conditional, source_supporting, source_not_required) = (
            _source_bounded_contract_for_source(sid, text, facts))
        elements.extend(source_elements)
        subject.extend(source_subject)
        persons.extend(source_persons)
        triggers.extend(source_triggers)
        conditions.extend(source_conditions)
        consequences.extend(source_consequences)
        exceptions.extend(source_exceptions)
        conditional_facts.extend(source_conditional)
        supporting_facts.extend(source_supporting)
        not_required.extend(source_not_required)
    blocking_facts: list[RuleFact] = []
    facts_not_required = list(dict.fromkeys(not_required))
    branches = list(dict.fromkeys(
        branch for fact in conditional_facts for branch in fact.branches))
    summary = " ".join(item.description for item in elements[:3]).strip()
    return RuleContract(
        task_id=task_id,
        rule_type="source_bounded_rule",
        rule_summary=summary,
        primary_source_ids=list(selection.primary_sources),
        elements=elements,
        supported_exceptions=exceptions,
        subject=subject,
        persons_covered=persons,
        trigger_events=triggers,
        positive_conditions=conditions,
        consequences_or_remedies=consequences,
        blocking_facts=blocking_facts,
        conditional_facts=conditional_facts,
        supporting_facts=supporting_facts,
        decisive_facts_needed=[],
        facts_not_required=facts_not_required,
        conditional_branches=branches,
        permitted_claims=[item.description for item in elements],
        prohibited_claims=[f"Ne pas présenter {item} comme une condition de la source sans passage explicite."
                           for item in facts_not_required],
        extraction_status="source_bounded" if elements else "fallback_limited",
        application_limits=([] if elements else
                            ["Aucune proposition opérante lisible n'a pu être extraite du texte récupéré."]),
    )


def validate_rule_contract(
    contract: RuleContract, retrieved_source_ids: set[str],
    rejected_source_ids: set[str] | None = None,
    source_texts: dict[str, str] | None = None,
) -> list[str]:
    """Validate source IDs and exact supporting passages."""
    rejected = rejected_source_ids or set()
    errors: list[str] = []
    for sid in contract.primary_source_ids:
        if sid not in retrieved_source_ids:
            errors.append(f"source principale absente: {sid}")
    all_elements = [*contract.elements, *contract.supported_exceptions,
                    *contract.subject, *contract.persons_covered,
                    *contract.trigger_events, *contract.positive_conditions,
                    *contract.consequences_or_remedies]
    for element in all_elements:
        if not element.support_source_ids or not element.supporting_passages:
            errors.append(f"élément sans passage source: {element.id}")
        for sid in element.support_source_ids:
            if sid not in retrieved_source_ids:
                errors.append(f"source absente pour {element.id}: {sid}")
            if sid in rejected:
                errors.append(f"source rejetée utilisée par {element.id}: {sid}")
            if source_texts and sid in source_texts:
                for passage in element.supporting_passages:
                    if _fold(passage) not in _fold(source_texts[sid]):
                        errors.append(f"passage absent pour {element.id}: {sid}")
    return list(dict.fromkeys(errors))


def build_clarification_decision(
    contract: RuleContract, facts: dict[str, Any],
    clarification_history: list[dict[str, Any]], *, task_id: str = "",
) -> ClarificationDecision:
    """Authorize only source-derived blocking facts, never legacy profiles."""
    asked_ids = {
        str(item.get("clarification_id")) for item in clarification_history
        if item.get("clarification_id")
    }
    asked_facts = {
        str(key) for item in clarification_history
        for key in item.get("fact_keys", [])
    }
    for fact in contract.blocking_facts:
        value = facts.get(fact.fact_id)
        if value not in (None, "", [], {}) and not (
                isinstance(value, dict) and value.get("value") is None):
            continue
        clarification_id = f"rule-fact-{fact.fact_id}"
        already = clarification_id in asked_ids or fact.fact_id in asked_facts
        status = next((str(item.get("status")) for item in clarification_history
                       if fact.fact_id in item.get("fact_keys", [])), "not_asked")
        return ClarificationDecision(
            needed=not already,
            blocking=True,
            answerable_conditionally=False,
            missing_fact_id=fact.fact_id,
            missing_rule_element_id=fact.fact_id,
            question=f"Pouvez-vous préciser le fait suivant : {fact.description}?",
            changes_legal_regime=True,
            reason="Le contrat source-bounded identifie ce fait comme bloquant.",
            already_asked=already,
            user_answer_status=status,
        )
    return ClarificationDecision(
        needed=False, blocking=False, answerable_conditionally=True,
        conditional_branches=list(contract.conditional_branches),
        reason="Aucun fait bloquant dérivé du texte; une réponse conditionnelle est possible.",
        already_asked=False,
    )


def authorize_planner_action(state: dict[str, Any], proposed_action: Any) -> PlannerDecision:
    """Deterministic policy boundary after the planner proposal."""
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
            thinking_text="Les sources officielles suffisent; aucune recherche jurisprudentielle n'est justifiée.",
            decision_trace=DecisionTrace(
                request_type=decision.request_type, jurisdiction=decision.jurisdiction,
                need="source suffisante", next_action="final_answer"),
        )
    if (decision.decision == Decision.call_tool and decision.next_tool
            and state.get("tool_history") and not state.get("information_gap")
            and decision.next_tool in {"search_quebec_jurisprudence",
                                       "search_quebec_regulations",
                                       "semantic_search_ccq", "semantic_search_cpc"}):
        return PlannerDecision(
            request_type=decision.request_type, jurisdiction=decision.jurisdiction,
            decision=Decision.final_answer,
            thinking_text="Aucune lacune explicite n'autorise une recherche supplémentaire.",
            decision_trace=DecisionTrace(
                request_type=decision.request_type, jurisdiction=decision.jurisdiction,
                need="gap explicite absent", next_action="final_answer"),
        )
    return decision


def normative_references(observations: Iterable[ToolObservation]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for sid, (text, _number, _index) in retrieved_articles(observations).items():
        for match in _REGULATION_RE.finditer(text):
            refs.append({"reference_type": "regulation",
                         "reference_text": match.group(0).strip(),
                         "target_title": "", "source_id": sid,
                         "status": "unresolved"})
    return refs


def decide_source_sufficiency(
    selection: PrimaryAuthoritySelection, rule_contract: RuleContract,
    observations: Iterable[ToolObservation], *, task_id: str = "",
    jurisprudence_requested: bool = False,
    regulation_resolved: bool = False,
) -> SourceSufficiencyDecision:
    refs = normative_references(observations)
    if not selection.primary_sources:
        legislation = "missing"
    elif rule_contract.elements and rule_contract.extraction_status == "source_bounded":
        legislation = "sufficient"
    else:
        legislation = "required"
    if regulation_resolved and refs:
        for reference in refs:
            reference["status"] = "resolved"
    regulation = ("sufficient" if refs and regulation_resolved else
                  "required" if refs else "not_needed")
    if jurisprudence_requested:
        jurisprudence = "required"
    else:
        jurisprudence = "not_needed" if legislation == "sufficient" else "conditionally_required"
    return SourceSufficiencyDecision(
        task_id=task_id,
        sufficient_for_initial_answer=legislation == "sufficient" and not refs,
        legislation_status=legislation, regulation_status=regulation,
        jurisprudence_status=jurisprudence, doctrine_status="not_needed",
        missing_questions=(
            ["Quel règlement ou quelle annexe est visé par le renvoi explicite de la source récupérée?"]
            if refs else []),
        explicit_normative_references=[item["reference_text"] for item in refs],
        reason=("Les propositions opérantes sont récupérées et bornées par des passages sources."
                if legislation == "sufficient" else
                "Une source officielle ou une proposition opérante reste à récupérer."),
    )


def _claim_supported_by_source(claim: str, source: str) -> tuple[str, str | None]:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return "unsupported", None
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", source) if part.strip()]
    best = max(sentences, key=lambda sentence: len(claim_tokens & _tokens(sentence)), default="")
    overlap = len(claim_tokens & _tokens(best)) / max(1, len(claim_tokens))
    normalized_claim = _fold(claim).strip(" .,:;\"'")
    if normalized_claim and normalized_claim in _fold(source):
        return "direct", best
    if overlap >= 0.75 and not re.search(r"\b(?:doit|peut|est tenu|a droit)\b", claim, re.I):
        return "reasonable_inference", best
    return "unsupported", None


def build_claim_ledger(answer: str, selection: PrimaryAuthoritySelection,
                       source_texts: dict[str, str], *, task_id: str = "") -> ClaimLedger:
    claims: list[LegalClaim] = []
    allowed = [sid for sid in [*selection.primary_sources, *selection.secondary_sources]
               if sid in source_texts]
    if not allowed:
        # Non-article official tools (regulations, fetched decisions and
        # official forms) use tool-scoped source IDs.  They remain bounded by
        # the caller's verified source_texts even when no CCQ/CPC authority
        # was selected.
        allowed = list(source_texts)
    for index, paragraph in enumerate(re.split(r"(?<=[.!?])\s+|\n+", answer or "")):
        text = paragraph.strip()
        if (not text or not _LEGAL_MARKER_RE.search(text)
                or "get_" in text.casefold()
                or text.casefold().startswith(("règles et documents", "regles et documents"))):
            continue
        if (re.search(r"\bArticle\s+\d", text, re.I)
                and not re.search(r"\b(?:prévoit|prevu|stipule|dispose|doit|peut|"
                                  r"est tenu|a droit|réparer|reparer|responsabil)\b",
                                  text, re.I)):
            continue
        if (text.casefold().startswith(("cet article", "explication"))
                and not re.search(r"\b(?:prévoit|prevu|stipule|dispose|doit|peut|"
                                  r"est tenu|a droit|réparer|reparer)\b", text, re.I)):
            continue
        cited = set(_ARTICLE_RE.findall(text))
        source_ids = [sid for sid in allowed
                      if not cited or sid.rsplit(":", 1)[-1] in cited]
        support_type = "unsupported"
        passage = None
        supporting_ids: list[str] = []
        for sid in source_ids:
            current_type, current_passage = _claim_supported_by_source(text, source_texts[sid])
            if current_type == "direct":
                support_type, passage = current_type, current_passage
                supporting_ids = [sid]
                break
            if current_type == "reasonable_inference" and support_type == "unsupported":
                support_type, passage = current_type, current_passage
                supporting_ids = [sid]
        verified = support_type != "unsupported"
        claims.append(LegalClaim(
            claim_id=f"claim-{index}", text=text,
            source_ids=supporting_ids,
            support_type=support_type,
            verification_status="verified" if verified else "failed",
            failure_reason=(None if verified else
                            "Aucun passage récupéré n'emporte cette proposition exacte."),
            premises=[passage] if support_type == "reasonable_inference" and passage else [],
            inference_explanation=("Les termes de l'affirmation sont présents dans le passage source; aucune condition nouvelle n'est ajoutée."
                                   if support_type == "reasonable_inference" else ""),
            task_id=task_id,
        ))
    return ClaimLedger(task_id=task_id, claims=claims)


def merge_failures(previous: list[dict[str, Any]], new: list[dict[str, Any]], *, node: str = "") -> list[dict[str, Any]]:
    """Append-only failure history. Resolved records are never deleted."""
    merged = [dict(item) for item in previous or []]
    seen = {(str(item.get("failure_type", "")), str(item.get("claim", "")),
             str(item.get("reason", ""))) for item in merged}
    for item in new or []:
        value = dict(item)
        value.setdefault("status", "open")
        value.setdefault("resolved_at_node", "")
        if node:
            value.setdefault("detected_at_node", node)
        key = (str(value.get("failure_type", "")), str(value.get("claim", "")),
               str(value.get("reason", "")))
        if key not in seen:
            merged.append(value)
            seen.add(key)
    return merged


def article_budget(config: Any, *, fetched_count: int = 0,
                   batch_index: int = 0, remaining_candidates: bool = True) -> dict[str, Any]:
    """Single source of truth for evidence-first candidate/fetch budgets."""
    enabled = bool(getattr(config, "evidence_first_enabled", False))
    if not enabled:
        initial = int(getattr(config, "initial_article_fetch_k", 6))
        batch = int(getattr(config, "article_fetch_batch_size", 6))
        maximum = int(getattr(config, "max_articles_per_issue", 20))
        return {"candidate_count": initial, "fetch_count": batch,
                "allow_next": remaining_candidates and fetched_count < maximum,
                "batch_index": batch_index}
    initial = int(getattr(config, "evidence_first_initial_fetch_count", 3))
    maximum_batches = int(getattr(config, "evidence_first_maximum_article_batches", 2))
    maximum = max(initial, maximum_batches * initial)
    allow_next = (remaining_candidates and batch_index + 1 < maximum_batches
                  and fetched_count < maximum)
    return {"candidate_count": int(getattr(config, "evidence_first_initial_candidate_count", 5)),
            "fetch_count": initial, "allow_next": allow_next,
            "batch_index": batch_index}


def normalize_and_repair_tool_args(catalog: Any, tool: str, arguments: Any,
                                   *, active_task: dict[str, Any] | None = None,
                                   latest_user_message: str = "",
                                   user_messages: Iterable[str] = ()) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Normalize arguments; reconstruct a query from the active task only."""
    raw = dict(arguments) if isinstance(arguments, dict) else {}
    spec = getattr(catalog, "tools", {}).get(tool)
    cleaned = ({key: value for key, value in raw.items() if key in spec.properties}
               if spec is not None else raw)
    removed = [key for key in raw if key not in cleaned]
    repaired: list[str] = []
    if tool in {"semantic_search_ccq", "semantic_search_cpc"} and not str(cleaned.get("query") or "").strip():
        task = active_task or {}
        candidates = [
            task.get("normalized_query"), task.get("active_issue"),
            *list(user_messages), latest_user_message,
            task.get("canonical_case_description"), task.get("summary"),
        ]
        query = next((str(item).strip() for item in candidates
                      if str(item or "").strip()
                      and _fold(item).strip() not in {"oui", "non", "je ne sais pas",
                                                     "probablement", "peut-etre"}), "")
        if query:
            cleaned["query"] = query
            repaired.append("query")
    errors = list(catalog.validate_call(tool, cleaned)) if hasattr(catalog, "validate_call") else []
    return cleaned, {"tool": tool, "removed_fields": removed,
                     "repaired_fields": repaired,
                     "remaining_arguments": cleaned, "errors": errors}, errors
