# -*- coding: utf-8 -*-
"""build_answer_contract — le contrat que la réponse doit honorer.

Fige, AVANT rédaction : la question exacte à laquelle répondre (le
dernier suivi, pas la question initiale), la juridiction établie, les
preuves utilisables — et le mode de réponse.

The writer receives ONLY:
  - usable_evidence (three-tier classified, level == usable)
  - explicitly labelled alternative_sources (when the contract allows)
It must NOT receive irrelevant candidate results as legal evidence.

Without usable evidence, the graph chooses: clarification, reformulation,
equivalent tool, coverage limitation, or cannot_conclude.
"""

from __future__ import annotations

from typing import Any

from lexior.services.modes import is_live
from lexior.services.assertion_grounding import (
    articles_incompatibles_deterministes,
    textes_recuperes,
)
from lexior.services.article_review import build_conditional_reasoning_contract
from lexior.agentic.schemas import RuleContract, SourceSufficiencyDecision

from ..context import GraphContext
from ..state import LexiorState, canonical_case_description, visible_tool_history

NAME = "build_answer_contract"

_SUBSTANTIVE_TYPES_NEEDING_EVIDENCE = {
    "exact_text_retrieval", "article_explanation", "topic_research",
    "case_analysis", "law_or_regulation_identification",
}

_COVERAGE_LIMITATION_FR = (
    "Je n'ai pas pu récupérer et vérifier {source_desc} à partir des "
    "sources actuellement disponibles dans ce système.")


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    # Le contrat voit les preuves déjà validées du dossier entier. Le budget
    # reste isolé au tour courant dans le planner.
    tool_history = visible_tool_history(state)
    exempt = ctx.services.validation.compute_exempt_tools(tool_history)
    live = is_live(state.get("mode", ""))

    # ── Validation de route (dataset, avant rédaction) ───────────────────
    if not live and state.get("stop_reason") != "clarification_required":
        route_errors = ctx.services.validation.validate_tool_route(
            state["scenario"].request_type,
            [o.tool_name for o in tool_history],
            exempt_tools=exempt,
        )
        if route_errors:
            return {
                "status": "rejected",
                "stop_reason": "; ".join(route_errors),
                "deterministic_blockers": list(route_errors),
                "exempt_tools": exempt,
            }

    # ── Evidence from the three-tier collections ─────────────────────────
    alternative_entries = state.get("alternative_sources", [])
    coverage_gaps = state.get("coverage_gaps", [])

    usable_idx = state.get("usable_evidence", [])
    usable_tools = [
        tool_history[i].tool_name for i in usable_idx
        if 0 <= i < len(tool_history)
    ]

    # Search returns candidates and official retrieval can return several
    # articles. Before substantive writing, an independent reviewer compares
    # their conditions with the facts. In live mode, a source that was not
    # reviewed is not authorized as a legal basis.
    textes_officiels = textes_recuperes(tool_history)
    texte_exact_demande = (
        state.get("requested_output_type") == "article_text"
        or state["scenario"].request_type == "exact_text_retrieval"
    )
    faits = canonical_case_description(state)
    service_selection = ctx.services.assertion_grounding
    # Le dossier live exige une revue explicite. Les trajectoires offline
    # n'ont pas de relecteur LLM : elles conservent leur contrat historique,
    # déjà couvert par les validations de route et de grounding.
    reviews_stored = (dict(state.get("article_reviews", {})) if live else {})
    selection_disponible = (
        bool(textes_officiels)
        and not texte_exact_demande
        and service_selection.disponible()
        and not reviews_stored
    )
    incompatibles_deterministes = (
        articles_incompatibles_deterministes(textes_officiels, faits=faits)
        if not texte_exact_demande else {})
    selection_articles = (
        service_selection.selectionner_articles(textes_officiels, faits=faits)
        if selection_disponible else {})
    filtre_articles_effectue = bool(
        incompatibles_deterministes or selection_disponible or reviews_stored)
    if texte_exact_demande:
        # A request for exact text reproduces the sources without drawing a
        # legal conclusion, so it does not need the applicability filter.
        articles_retenus = list(textes_officiels)
    elif reviews_stored:
        # La revue a eu lieu à la récupération. On consomme le même verdict
        # à l'étape de rédaction afin d'éviter tout changement de règle entre
        # recherche, jurisprudence et réponse finale.
        articles_retenus = [
            numero for numero in textes_officiels
            if (numero not in incompatibles_deterministes
                and reviews_stored.get(numero, {}).get(
                    "retrieval_group", "primary") == "primary"
                and reviews_stored.get(numero, {}).get("status") in {
                    "applicable", "conditionally_applicable"})
        ]
    elif selection_disponible:
        # Allowlist: no verdict means the batch could not be reviewed, not
        # that the article is applicable.
        # In a fact-specific case analysis, an ``incertain`` article has a
        # material condition missing from the user's facts. It cannot support
        # a legal conclusion until that fact is established. Topic research
        # may still retain such a source to describe its conditional scope.
        statuts_autorises = (
            {"applicable"}
            if (state.get("request_type") or state["scenario"].request_type)
            == "case_analysis"
            else {"applicable", "incertain"}
        )
        articles_retenus = [
            numero for numero, verdict in selection_articles.items()
            if (numero not in incompatibles_deterministes
                and verdict.statut in statuts_autorises)
        ]
    else:
        # Offline mode does not simulate an LLM relevance decision. The
        # deterministic exclusions remain active.
        articles_retenus = [
            numero for numero in textes_officiels
            if numero not in incompatibles_deterministes
        ]
    if filtre_articles_effectue and not articles_retenus:
        usable_idx = [
            index for index in usable_idx
            if not (0 <= index < len(tool_history)
                    and tool_history[index].tool_name in {
                        "get_ccq_articles", "get_cpc_articles"})
        ]
        usable_tools = [
            tool_history[i].tool_name for i in usable_idx
            if 0 <= i < len(tool_history)
        ]

    # Evidence-first allowlist: top-k candidates and retrieved articles are
    # not automatically equivalent legal authorities. The writer receives
    # only selected primary/secondary article numbers.
    authority_selection = state.get("primary_authority_selection")
    if isinstance(authority_selection, dict):
        selected_source_ids = [
            *authority_selection.get("primary_sources", []),
            *authority_selection.get("secondary_sources", []),
        ]
    else:
        selected_source_ids = [
            *getattr(authority_selection, "primary_sources", []),
            *getattr(authority_selection, "secondary_sources", []),
        ]
    selected_numbers = {
        str(source_id).rsplit(":", 1)[-1]
        for source_id in selected_source_ids if ":" in str(source_id)
    }
    if selected_numbers and articles_retenus:
        articles_retenus = [str(number) for number in articles_retenus
                            if str(number) in selected_numbers]

    # Les décisions vérifiées sont persistées par signature dans le dossier,
    # pas par un index de l'ancien tour. On reconstruit donc leur index dans la
    # vue visible courante avant de remettre la preuve au rédacteur.
    if live and state.get("case_law_search_status") == "verified":
        verified_hashes = {
            getattr(item, "content_hash", "")
            for item in state.get("case_law_verified", [])
        }
        usable_idx = list(dict.fromkeys(
            [*usable_idx, *[
                index for index, item in enumerate(tool_history)
                if item.tool_name == "get_quebec_regulation"
                and (not verified_hashes or item.content_hash in verified_hashes)
            ]]))
        usable_tools = [
            tool_history[i].tool_name for i in usable_idx
            if 0 <= i < len(tool_history)
        ]
    elif articles_retenus:
        # Les indices historiques ne sont plus fiables après la projection
        # ``preuves persistantes + tour courant``. On les reconstruit à partir
        # des textes réellement présents dans chaque observation officielle.
        official_indices = [
            index for index, observation in enumerate(tool_history)
            if (observation.tool_name in {"get_ccq_articles", "get_cpc_articles"}
                and any(number in articles_retenus for number in
                        textes_recuperes([observation])))
        ]
        usable_idx = list(dict.fromkeys([*usable_idx, *official_indices]))
        usable_tools = [
            tool_history[i].tool_name for i in usable_idx
            if 0 <= i < len(tool_history)
        ]

    unusable = [
        {"tool": e.tool_name, "status": e.result_status}
        for e in state.get("search_evaluations", [])
        if e.result_status not in ("usable", "exact_match", "truncated")
    ]

    attempted_research = bool(tool_history)
    # ``usable_entries`` reste l'historique de classification; ``usable_idx``
    # est la liste effectivement autorisée dans le contrat, après le filtre
    # d'applicabilité des articles officiels.
    has_usable = bool(usable_idx)
    needs_evidence = (
        state.get("request_type", "")
        in _SUBSTANTIVE_TYPES_NEEDING_EVIDENCE)

    if not attempted_research:
        answer_mode = "direct"
    elif has_usable:
        answer_mode = "grounded"
    elif needs_evidence:
        answer_mode = "no_evidence"
    else:
        answer_mode = "direct"

    directives: list[str] = [
        "Réponds à la DERNIÈRE question de l'utilisateur.",
    ]
    if state.get("refers_to_previous_answer"):
        directives.append(
            "C'est une question de suivi : ne répète pas la réponse "
            "précédente; fournis exactement ce qui est demandé "
            f"({state.get('requested_output_type', 'answer')}).")
    if answer_mode == "no_evidence":
        directives.append(
            "AUCUNE preuve utilisable n'a été récupérée : n'affirme "
            "aucune règle de fond; explique la limite de recherche et "
            "oriente vers les sources officielles (CanLII, SOQUIJ).")
    if filtre_articles_effectue:
        if articles_retenus:
            directives.append(
                "Les textes officiels ont été comparés aux faits : utilise "
                "uniquement les articles retenus dans le contrat et exprime "
                "comme condition tout fait encore incertain.")
        else:
            directives.append(
                "Les textes officiels récupérés sont incompatibles avec les "
                "faits connus : n'en déduis aucune règle de fond.")

    raisonnement_autorise = [
        {
            "article": numero,
            "statut": reviews_stored.get(numero, {}).get(
                "status", "applicable"),
            "motif": reviews_stored.get(numero, {}).get("reason", ""),
            "consigne": (
                "Présente l'application comme conditionnelle; ne conclus pas "
                "que les conditions sont remplies."
                if reviews_stored.get(numero, {}).get("status")
                == "conditionally_applicable"
                else "Explique seulement ce que le texte officiel permet de dire."
            ),
        }
        for numero in articles_retenus
    ]
    evidence_first = bool(ctx.config.evidence_first_enabled)
    conditional_reasoning_contract = []
    if not evidence_first:
        conditional_reasoning_contract = build_conditional_reasoning_contract(
            reviews_stored, textes_officiels, state.get("facts") or {})
        if conditional_reasoning_contract:
            directives.append(
                "Utilise le contrat de raisonnement conditionnel legacy et "
                "distingue les faits affirmés des faits vérifiés.")

    raw_rule_contract = state.get("rule_contract") or {}
    rule_contract = (raw_rule_contract if isinstance(raw_rule_contract, RuleContract)
                    else RuleContract.model_validate(raw_rule_contract))
    raw_sufficiency = state.get("source_sufficiency_decision") or {}
    sufficiency = (raw_sufficiency if isinstance(raw_sufficiency, SourceSufficiencyDecision)
                   else SourceSufficiencyDecision.model_validate(raw_sufficiency))
    if evidence_first:
        if rule_contract.conditional_branches:
            directives.append(
                "Présente les branches conditionnelles dérivées des passages "
                "sources avec une formulation si/alors.")
        if rule_contract.supporting_facts:
            directives.append(
                "Mentionne les éléments de preuve comme supporting facts; ils "
                "ne constituent pas des conditions bloquantes.")
        if sufficiency.sufficient_for_initial_answer:
            directives.append(
                "Réponds directement à l'objectif de l'utilisateur; n'ajoute "
                "pas une limitation générique si la source suffit.")

    # Coverage gap directives.
    for gap in coverage_gaps:
        desc = gap.get("requested_court_scope", "") or gap.get(
            "requested_document_type", "cette source")
        directives.append(_COVERAGE_LIMITATION_FR.format(
            source_desc=f"une décision de {desc}"))

    # Alternative sources directive.
    alternatives_for_contract = []
    if alternative_entries and answer_mode in ("grounded", "no_evidence"):
        for alt in alternative_entries:
            alternatives_for_contract.append({
                "tool": alt.get("tool_name", ""),
                "status": alt.get("detailed_status", "alternative_only"),
                "reason": alt.get("reason", ""),
            })
        directives.append(
            "Des sources ALTERNATIVES (non-équivalentes) sont "
            "disponibles. Tu peux les mentionner en les identifiant "
            "explicitement comme alternatives, jamais comme réponses "
            "directes à la demande.")

    contract = {
        "question_courante": (state.get("latest_user_intent")
                              or state["scenario"].user_query),
        "enjeu_actif": state.get("active_issue", ""),
        "question_de_suivi": state.get("refers_to_previous_answer", False),
        "type_de_sortie": state.get("requested_output_type", "answer"),
        "juridiction_etablie": (state.get("resolved_jurisdiction")
                                or "unknown"),
        "juridiction_verrouillee": state.get("jurisdiction_locked", False),
        "mode_de_reponse": answer_mode,
        "preuves_utilisables": usable_tools,
        # Indices, et non seulement noms d'outils : le rédacteur peut ainsi
        # recevoir exactement les observations classées utilisables.
        "indices_preuves_utilisables": [
            index for index in usable_idx
            if isinstance(index, int) and 0 <= index < len(tool_history)
        ],
        "preuves_inutilisables": unusable,
        "filtre_articles_officiels": filtre_articles_effectue,
        "articles_retenus": articles_retenus,
        "primary_rule_source_ids": list(
            authority_selection.get("primary_sources", [])
            if isinstance(authority_selection, dict)
            else getattr(authority_selection, "primary_sources", [])),
        "secondary_rule_source_ids": list(
            authority_selection.get("secondary_sources", [])
            if isinstance(authority_selection, dict)
            else getattr(authority_selection, "secondary_sources", [])),
        "primary_authority_selection": (
            authority_selection.model_dump(mode="json")
            if hasattr(authority_selection, "model_dump") else authority_selection),
        "rule_contract": rule_contract.model_dump(mode="json"),
        "source_sufficiency_decision": sufficiency.model_dump(mode="json"),
        "authorized_source_ids": list(selected_source_ids),
        "conditional_branches": list(rule_contract.conditional_branches),
        "supporting_facts": [item.model_dump(mode="json")
                             for item in rule_contract.supporting_facts],
        "limitations": list(rule_contract.application_limits),
        "permitted_claims": list(rule_contract.permitted_claims),
        "prohibited_claims": list(rule_contract.prohibited_claims),
        "raisonnement_autorise": raisonnement_autorise,
        "sources_alternatives": alternatives_for_contract,
        "lacunes_de_couverture": [g for g in coverage_gaps],
        "consignes": directives,
    }
    if not evidence_first and conditional_reasoning_contract:
        contract["conditional_reasoning_contract"] = conditional_reasoning_contract

    return {
        "answer_contract": contract,
        "exempt_tools": exempt,
        "status": "answering",
    }
