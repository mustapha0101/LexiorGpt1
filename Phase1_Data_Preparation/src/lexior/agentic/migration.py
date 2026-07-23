# -*- coding: utf-8 -*-
"""Migration de records agentic-1.0 vers agentic-2.0."""

from __future__ import annotations

from typing import Any

_OLD_TO_NEW: dict[str, dict[str, Any]] = {
    "article_ccq_precis": {
        "request_type": "exact_text_retrieval",
        "jurisdiction_status": "supported_quebec",
        "source_intent": "legislation",
    },
    "article_cpc_precis": {
        "request_type": "exact_text_retrieval",
        "jurisdiction_status": "supported_quebec",
        "source_intent": "procedural_rule",
    },
    "explication_article": {
        "request_type": "article_explanation",
        "jurisdiction_status": "supported_quebec",
    },
    "recherche_theme_ccq": {
        "request_type": "topic_research",
        "jurisdiction_status": "supported_quebec",
    },
    "recherche_theme_cpc": {
        "request_type": "topic_research",
        "source_intent": "procedural_rule",
    },
    "cas_civil_quebecois": {
        "request_type": "case_analysis",
        "jurisdiction_status": "supported_quebec",
    },
    "cas_procedure_quebecoise": {
        "request_type": "procedure_guidance",
    },
    "reglement_quebecois_connu": {
        "request_type": "law_or_regulation_identification",
        "source_identity": "exactly_known",
    },
    "reglement_quebecois_inconnu": {
        "request_type": "law_or_regulation_identification",
        "source_identity": "unknown_but_discoverable",
    },
    "jurisprudence_quebecoise": {
        "request_type": "case_law_research",
        "jurisdiction_status": "supported_quebec",
    },
    "loi_federale": {
        "request_type": "law_or_regulation_identification",
        "jurisdiction_status": "supported_federal",
    },
    "jurisprudence_federale": {
        "request_type": "case_law_research",
        "jurisdiction_status": "supported_federal",
    },
    "cas_federal_concret": {
        "request_type": "case_analysis",
        "jurisdiction_status": "supported_federal",
    },
    "comparaison_quebec_federal": {
        "request_type": "comparative_law",
    },
    "verification_entree_en_vigueur": {
        "request_type": "legislative_status_verification",
    },
    "couverture_dataset": {
        "request_type": "dataset_coverage",
    },
    "question_non_juridique": {
        "request_type": "non_legal",
    },
    # Old categories that were really metadata, not types
    "panne_mcp": {
        "request_type": "case_analysis",
        "planned_failure_mode": "tool_error",
    },
    "resultat_vide": {
        "request_type": "topic_research",
        "planned_failure_mode": "empty_result",
    },
    "source_trop_longue": {
        "request_type": "law_or_regulation_identification",
        "planned_failure_mode": "truncated_source",
    },
    "juridiction_ambigue": {
        "request_type": "case_analysis",
        "clarification_stage": "before_search",
    },
    "question_incomplete": {
        "request_type": "case_analysis",
        "clarification_stage": "before_search",
    },
    "clarification_puis_recherche": {
        "request_type": "topic_research",
        "clarification_stage": "before_search",
    },
}


def migrate_v1_to_v2(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 (agentic-1.0) record dict to v2 (agentic-2.0) structure.

    Preserves the original record under ``_v1_original``.
    """
    out = dict(record)
    out["_v1_original"] = dict(record)
    out["schema_version"] = "agentic-2.0"
    out["dataset_type"] = "agentic_legal_intermediate"

    old_type = record.get("request_type", "")
    mapping = _OLD_TO_NEW.get(old_type, {})
    if mapping:
        out["request_type"] = mapping["request_type"]
        for field in ("jurisdiction_status", "source_intent", "source_identity",
                      "clarification_stage", "planned_failure_mode"):
            if field in mapping:
                out.setdefault(field, mapping[field])

    facts_missing = record.get("facts_missing") or []
    if facts_missing and not record.get("facts_required_before_search"):
        out["facts_required_before_search"] = list(facts_missing)
        out.setdefault("facts_required_before_application", [])
        out.setdefault("facts_useful", [])
        out.setdefault("retrieval_targets", [])

    out.setdefault("search_evaluations", [])
    out.setdefault("final_claims", [])
    out.setdefault("clarification_stage", "none")
    out.setdefault("jurisdiction_status", "supported_quebec")
    out.setdefault("source_intent", "legislation")

    return out
