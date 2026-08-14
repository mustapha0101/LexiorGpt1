# -*- coding: utf-8 -*-
"""validate_plan — le vérificateur déterministe autorise la route.

Le planner n'a AUCUNE autorité pour contourner ces contrôles :

  1. borne de décisions (max_tool_calls + 2);
  2. clarification bornée (dataset : 1; live : 2 puis synthèse forcée);
  3. budget d'outils (dépassement → synthèse, jamais un appel de plus);
  4. structure et politique de route (dataset — routes scriptées);
  5. juridiction : la valeur résolue/verrouillée écrase la proposition,
     et les outils exclusivement québécois sont bloqués hors Québec.
"""

from __future__ import annotations

import re
from typing import Any

from lexior.agentic.schemas import Decision, DecisionTrace, PlannerDecision, RuleContract
from lexior.services.evidence import CoverageGap
from lexior.services.article_review import (
    build_clarification, question_for_fact_keys,
)
from lexior.services.jurisdiction import (
    QC_ONLY_TOOLS,
    allows_quebec_tools,
    coverage_action,
    is_federal,
)
from lexior.services.modes import is_live
from lexior.services.text_folding import fold_text
from lexior.services.tool_coverage import get_coverage
from lexior.services.evidence_first import (
    authorize_planner_action,
    build_clarification_decision,
    normalize_and_repair_tool_args,
)
from lexior.services.validation import ProposalVerdict

from ..context import GraphContext
from ..state import LexiorState, canonical_case_description

NAME = "validate_plan"

# Faits manquants qui portent sur la juridiction : une fois celle-ci résolue,
# les redemander revient à ignorer la réponse de l'usager.
_RE_FAIT_JURIDICTION = re.compile(
    r"juridiction|province|québec|quebec|fédéral|federal|canada",
    re.IGNORECASE)
# Recherches par le sens : leurs deux champs décrivent une SITUATION. Un
# numéro d'article n'y sert à rien — l'index compare du texte, pas des
# références — et sa présence signale que le modèle part d'une croyance au
# lieu de décrire les faits. Vu en mesure : legal_terms « responsabilité du
# propriétaire d'un animal, dommages causés, article 1465 CCQ », alors que
# l'article des animaux est 1466.
_CHAMPS_DE_RECHERCHE = {"semantic_search_ccq": ("query", "legal_terms"),
                        "semantic_search_cpc": ("query", "legal_terms")}

_NUM = r"\d{1,4}(?:\.\d+)?"
_CODE = r"C\.?\s?c\.?\s?Q\.?|CCQ|C\.?\s?p\.?\s?c\.?|CPC"

_RE_NUMERO_ARTICLE = re.compile(
    # « article 1465 CCQ », « articles 1457 et 1458 », « art. 1457, 1458 à 1460 »
    rf"\b(?:articles?|art\.?)\s*{_NUM}"
    rf"(?:\s*(?:,|;|et|and|ou|à|au|-|–)\s*{_NUM})*"
    rf"(?:\s*(?:du\s+|selon\s+le\s+)?(?:{_CODE}))?"
    # « 1177 C.c.Q. » sans le mot « article »
    rf"|\b{_NUM}\s*(?:{_CODE})\b",
    re.IGNORECASE)

# Résidus laissés par la suppression : « selon . », « ( ) », « ,, ».
_RE_RESIDU = re.compile(
    r"\(\s*\)|\[\s*\]"                       # parenthèses vidées
    r"|\b(?:selon|suivant|prévu\s+(?:à|par)|en\s+vertu\s+d[eu])\s*(?=[.,;)]|$)",
    re.IGNORECASE)


def _sans_numero(valeur: str) -> str:
    nettoye = _RE_NUMERO_ARTICLE.sub(" ", valeur or "")
    nettoye = _RE_RESIDU.sub(" ", nettoye)
    nettoye = re.sub(r"\s*([,;])\s*(?=[,;])", "", nettoye)
    nettoye = re.sub(r"\s+([,;.)])", r"\1", nettoye)
    return re.sub(r"\s{2,}", " ", nettoye).strip(" ,;.-")


def _retirer_numeros_des_recherches(decision: PlannerDecision) -> None:
    """Retire les références d'article des arguments de recherche sémantique."""
    champs = _CHAMPS_DE_RECHERCHE.get(decision.next_tool or "")
    if not champs or not decision.arguments:
        return
    for champ in champs:
        valeur = decision.arguments.get(champ)
        if not isinstance(valeur, str) or not valeur:
            continue
        nettoye = _sans_numero(valeur)
        if nettoye != valeur:
            decision.arguments[champ] = nettoye


def _forced(decision: PlannerDecision, jurisdiction: str, need: str,
            thinking: str, action: Decision,
            question: str = "") -> PlannerDecision:
    effective_question = question or decision.clarification_question
    return PlannerDecision(
        request_type=decision.request_type,
        jurisdiction=jurisdiction,
        legal_regime=decision.legal_regime,
        missing_critical_facts=decision.missing_critical_facts,
        required_sources=decision.required_sources,
        decision=action,
        clarification_question=effective_question,
        thinking_text=thinking,
        legal_terms=decision.legal_terms,
        clarification_scope=decision.clarification_scope,
        clarification_blocking=decision.clarification_blocking,
        answerable_conditionally=decision.answerable_conditionally,
        clarification_fact_keys=decision.clarification_fact_keys,
        decision_trace=DecisionTrace(
            request_type=decision.request_type,
            jurisdiction=jurisdiction,
            need=need,
            next_action=action.value),
    )


def _forced_final(decision: PlannerDecision, jurisdiction: str,
                  need: str, thinking: str) -> PlannerDecision:
    return _forced(decision, jurisdiction, need, thinking,
                   Decision.final_answer)


def _pending_clarification(state: LexiorState,
                           decision: PlannerDecision,
                           ctx: GraphContext,
                           *, evidence_first: bool = False) -> dict[str, Any]:
    """Construit le contrat rendu à l'utilisateur avant l'interrupt()."""
    question = (decision.clarification_question or "").strip()
    clarification_scope = decision.clarification_scope
    if clarification_scope == "request_intent":
        return {
            "clarification_id": "request-intent",
            "category": "request_intent",
            "fact_keys": [],
            "question": question or ctx.services.clarification.build_question(
                decision),
            "answer_type": "free_text",
            "blocking": True,
            "answerable_conditionally": False,
            "source_articles": [],
            "status": "pending",
        }
    if clarification_scope == "jurisdiction":
        return {
            "clarification_id": "jurisdiction-province",
            "category": "jurisdiction",
            "fact_keys": decision.clarification_fact_keys or ["jurisdiction"],
            "question": question or ctx.services.clarification.build_question(
                decision),
            "answer_type": "province_or_federal",
            "blocking": True,
            "answerable_conditionally": False,
            "source_articles": [],
            "status": "pending",
        }
    if clarification_scope == "legal_regime":
        location_missing = not state.get("resolved_jurisdiction")
        return {
            "clarification_id": "legal-regime-employment",
            "category": "legal_regime",
            "fact_keys": (decision.clarification_fact_keys or (
                ["work_location", "employment_sector"]
                if location_missing else ["employment_sector"])),
            "question": question or ctx.services.clarification.build_question(
                decision,
                ["work_location", "employment_sector"]
                if location_missing else ["employment_sector"]),
            "answer_type": "location_and_employment_sector",
            "blocking": True,
            "answerable_conditionally": False,
            "source_articles": [],
            "status": "pending",
        }

    if evidence_first:
        raw_contract = state.get("rule_contract") or {}
        contract = (raw_contract if isinstance(raw_contract, RuleContract)
                    else RuleContract.model_validate(raw_contract))
        choice = build_clarification_decision(
            contract, state.get("facts") or {},
            state.get("clarification_history") or [],
            task_id=state.get("task_id", ""),
        )
        if choice.needed and choice.question:
            return {
                "clarification_id": f"rule-fact-{choice.missing_fact_id}",
                "category": "rule_element",
                "fact_keys": [str(choice.missing_fact_id)],
                "missing_rule_element_id": choice.missing_rule_element_id,
                "question": choice.question,
                "answer_type": "yes_no_or_explanation",
                "blocking": choice.blocking,
                "answerable_conditionally": choice.answerable_conditionally,
                "source_ids": list(dict.fromkeys([
                    *contract.primary_source_ids,
                    *contract.secondary_source_ids,
                ])),
                "status": "pending",
            }
        return {}

    context = state.get("case_context") or {}
    facts = dict(context.get("facts") or state.get("facts") or {})
    statements = facts.get("user_statements") or []
    selected = build_clarification(
        state.get("article_reviews") or {}, facts,
        state.get("clarification_history") or [], statements)
    if selected:
        return selected

    rule_contract = state.get("rule_contract") or {}
    if hasattr(rule_contract, "model_dump"):
        rule_contract = rule_contract.model_dump(mode="json")
    decisive = [str(item) for item in rule_contract.get(
        "decisive_facts_needed", []) if str(item).strip()]
    facts = state.get("facts") or {}
    missing_rule_facts = [item for item in decisive if facts.get(item) in (None, "", [], {})]
    if missing_rule_facts:
        source_ids = list(dict.fromkeys([
            *rule_contract.get("primary_source_ids", []),
            *rule_contract.get("secondary_source_ids", []),
        ]))
        return {
            "clarification_id": "rule-element-" + missing_rule_facts[0],
            "category": "rule_element",
            "fact_keys": [missing_rule_facts[0]],
            "missing_rule_element_id": missing_rule_facts[0],
            "question": question_for_fact_keys([missing_rule_facts[0]]),
            "answer_type": "yes_no_or_explanation",
            "source_ids": source_ids,
            "source_articles": [sid.rsplit(":", 1)[-1] for sid in source_ids],
            "status": "pending",
        }

    fact_keys = [str(value) for value in state.get(
        "missing_critical_facts", []) if str(value).strip()]
    if not fact_keys:
        fact_keys = ["user_provided_fact"]
    safe_question = question
    if (not safe_question or "que pouvez-vous confirmer" in safe_question.casefold()
            or "reason" in safe_question.casefold()
            or "article" in safe_question.casefold()):
        safe_question = question_for_fact_keys(fact_keys)
    return {
        "clarification_id": "fact-" + "-".join(fact_keys[:3]),
        "category": "fact",
        "fact_keys": fact_keys[:3],
        "question": safe_question,
        "answer_type": "yes_no_or_explanation",
        "blocking": False,
        "answerable_conditionally": True,
        "source_articles": [],
        "status": "pending",
    }


def _is_federal_matter(state: LexiorState,
                       decision: PlannerDecision) -> bool:
    """Le droit applicable est-il fédéral, quelle que soit la province ?"""
    state_regime = str(state.get("legal_regime", "")).casefold()
    return (state_regime == "federal"
            or is_federal(state.get("substantive_law", ""))
            or is_federal(state.get("expected_jurisdiction", ""))
            or decision.legal_regime == "federal"
            or is_federal(decision.jurisdiction))


def _distinct_legal_terms(legal_terms: str, query: str) -> str:
    """Seconde formulation, ou rien si elle recopie la question.

    ``semantic_search_*`` réunit DEUX formulations. Recopier la question
    dans ``legal_terms`` rend l'union dégénérée : les deux canaux voient le même texte,
    et le jeu de candidats se réduit aux premiers rangs denses — les
    candidats purement lexicaux et les rangs denses suivants disparaissent.
    Une chaîne vide est la façon documentée de dire « une seule
    formulation ».
    """
    terms = (legal_terms or "").strip()
    if not terms:
        return ""
    folded_terms = fold_text(terms)
    folded_query = fold_text(query)
    if folded_terms == folded_query:
        return ""
    # Recopie déguisée : la question entière plus quelques mots.
    if folded_query and folded_query in folded_terms:
        reste = folded_terms.replace(folded_query, " ").split()
        if len(reste) < 3:
            return ""
    return terms


def _initial_research_for_nonblocking_fact(
        state: LexiorState, decision: PlannerDecision,
        ctx: GraphContext) -> PlannerDecision:
    """Remplace une question factuelle prématurée par la première recherche.

    Cette fonction ne devine pas le droit applicable : elle n'est appelée
    qu'après résolution du lieu et du régime lorsque celui-ci est nécessaire.
    """
    query = str(
        state.get("active_issue")
        or state.get("latest_user_intent")
        or state.get("latest_user_message")
        or state["scenario"].user_query
    ).strip()
    legal_terms = str(decision.legal_terms or query).strip()
    federal = _is_federal_matter(state, decision)
    if federal and "search_legal_documents" in ctx.catalog.tools:
        tool = "search_legal_documents"
        arguments = {"query": legal_terms or query, "doc_type": "laws",
                     "search_language": "fr"}
    else:
        tool = ("semantic_search_cpc"
                if state.get("request_type") == "procedure_guidance"
                else "semantic_search_ccq")
        if tool not in ctx.catalog.tools:
            return _forced_final(
                decision, state.get("resolved_jurisdiction", ""),
                need="aucun outil initial compatible",
                thinking="Aucun outil compatible n'est disponible pour poursuivre.")
        arguments = {"query": query,
                     "legal_terms": _distinct_legal_terms(legal_terms, query)}
    decision.decision = Decision.call_tool
    decision.next_tool = tool
    decision.arguments = arguments
    decision.clarification_question = None
    decision.clarification_scope = "none"
    decision.clarification_blocking = False
    decision.answerable_conditionally = True
    decision.decision_trace.need = "rechercher la règle avant son application"
    decision.decision_trace.next_action = f"call_tool:{tool}"
    decision.thinking_text = (
        "Le fait manquant concerne seulement l'application. Je recherche "
        "d'abord la règle, puis je répondrai conditionnellement.")
    return decision


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    raw = state.get("latest_decision")
    if not raw:
        return {"status": "rejected",
                "stop_reason": "décision du planner absente"}

    decision = PlannerDecision.model_validate(raw)
    live = is_live(state.get("mode", ""))
    step = state.get("step", 0)
    max_steps = state.get("max_planner_decisions", 12)
    resolved = state.get("resolved_jurisdiction", "")
    intent = state.get("request_intent", "ambiguous")

    if live:
        # La classification structurée est calculée avant le planner et reste
        # la source de vérité. Le planner choisit une action, pas le domaine.
        decision.request_type = state.get("request_type", "unknown")
        decision.decision_trace.request_type = decision.request_type

        if intent == "ambiguous":
            decision = _forced(
                decision, resolved,
                need="intention de la demande à préciser",
                thinking=("Le message ne contient pas encore assez de contexte "
                          "pour déterminer la demande."),
                action=Decision.ask_clarification,
            )
            decision.clarification_scope = "request_intent"
            decision.clarification_blocking = True
            decision.answerable_conditionally = False
            decision.clarification_question = None
        elif intent in {"greeting", "non_legal"}:
            if decision.decision != Decision.final_answer:
                decision = _forced_final(
                    decision, resolved,
                    need="réponse conversationnelle sans recherche juridique",
                    thinking=("La demande ne nécessite ni juridiction ni "
                              "source juridique."))
            decision.clarification_question = None
            decision.clarification_scope = "none"
            decision.clarification_blocking = False
        elif decision.decision == Decision.ask_clarification:
            # Les deux dépendances structurelles priment sur une question
            # factuelle proposée par le planner. Le scope vient des champs
            # sémantiques, jamais du vocabulaire de sa question.
            if (state.get("employment_regime_material", False)
                    and state.get("legal_regime", "unknown") == "unknown"):
                decision.clarification_scope = "legal_regime"
                decision.clarification_blocking = True
            elif (state.get("jurisdiction_material", False) and not resolved):
                decision.clarification_scope = "jurisdiction"
                decision.clarification_blocking = True
            elif decision.clarification_scope == "none":
                decision.clarification_scope = "application_fact"
                decision.clarification_blocking = False
            if (decision.clarification_scope == "legal_regime"
                    and not state.get("employment_regime_material", False)):
                decision.clarification_scope = "application_fact"
                decision.clarification_blocking = False
            if (decision.clarification_scope == "jurisdiction"
                    and not state.get("jurisdiction_material", False)):
                decision.clarification_scope = "application_fact"
                decision.clarification_blocking = False
            if (decision.clarification_scope == "legal_regime"
                    and state.get("employment_regime_material", False)):
                decision.clarification_question = None
                decision.clarification_fact_keys = (
                    ["work_location", "employment_sector"]
                    if not resolved else ["employment_sector"])
                decision.answerable_conditionally = False
            elif (decision.clarification_scope == "jurisdiction"
                  and state.get("jurisdiction_material", False)):
                decision.clarification_question = None
                decision.clarification_fact_keys = ["jurisdiction"]
                decision.answerable_conditionally = False

    # Une question d'application ne doit jamais suspendre le flux. Avant la
    # première recherche, elle est remplacée par une recherche compatible;
    # après récupération de sources, elle devient une réponse conditionnelle.
    if (live and intent == "legal"
            and decision.decision == Decision.ask_clarification
            and decision.clarification_scope == "application_fact"
            and not decision.clarification_blocking):
        if state.get("tool_history"):
            decision = _forced_final(
                decision, resolved,
                need="fait d'application non bloquant",
                thinking=("La règle peut être exposée avec les faits connus; "
                          "les inconnues seront posées à la fin."))
        else:
            decision = _initial_research_for_nonblocking_fact(
                state, decision, ctx)

    # 1. Borne de décisions du planner.
    if step >= max_steps:
        if live:
            final = _forced_final(
                decision, resolved,
                need="budget de décisions live épuisé",
                thinking=("La limite de décisions live est atteinte. Je "
                          "termine avec les preuves déjà retenues."))
            return {"step": step, "latest_decision": final.model_dump(mode="json"),
                    "status": "planning",
                    "stop_reason": "planner_budget_exhausted"}
        return {
            "status": "rejected",
            "stop_reason": (
                f"limite de {max_steps} décisions Planner atteinte"),
        }

    # 5a. Juridiction autoritaire : la valeur résolue écrase la
    # proposition; en dataset, la proposition du planner raffine la
    # valeur non verrouillée (comportement historique).
    locked = state.get("jurisdiction_locked", False)
    updates: dict[str, Any] = {"step": step}

    # One deterministic argument boundary for both modes. In particular, a
    # missing semantic-search query is reconstructable from the active task;
    # only fields that cannot be reconstructed remain blocking errors.
    if decision.decision == Decision.call_tool and decision.next_tool:
        repaired_args, audit, arg_errors = normalize_and_repair_tool_args(
            ctx.catalog, decision.next_tool, decision.arguments,
            active_task={
                **(state.get("case_context") or {}),
                "normalized_query": state.get("active_issue") or "",
                "active_issue": state.get("active_issue") or "",
                "canonical_case_description": canonical_case_description(state),
            },
            latest_user_message=state.get("latest_user_message", ""),
            user_messages=[
                message.content for message in state.get("messages", [])
                if getattr(message.role, "value", message.role) == "user"
            ],
        )
        decision.arguments = repaired_args
        if audit.get("removed_fields") or audit.get("repaired_fields"):
            updates["last_tool_normalization"] = audit
        if arg_errors:
            return {
                **updates,
                "status": "rejected",
                "stop_reason": "; ".join(arg_errors),
                "deterministic_blockers": list(arg_errors),
            }
        if ctx.config.evidence_first_enabled:
            decision = authorize_planner_action(state, decision)

    sufficiency = state.get("source_sufficiency_decision") or {}
    if hasattr(sufficiency, "model_dump"):
        sufficiency = sufficiency.model_dump(mode="json")
    if (decision.decision == Decision.call_tool
            and decision.next_tool == "search_quebec_jurisprudence"
            and ctx.config.evidence_first_jurisprudence_requires_justification
            and state.get("request_type") != "case_law_research"
            and sufficiency.get("legislation_status") == "sufficient"
            and sufficiency.get("jurisprudence_status") == "not_needed"):
        decision = _forced_final(
            decision, resolved,
            need="sources officielles suffisantes; jurisprudence non demandée",
            thinking=("Les sources officielles récupérées couvrent la première "
                      "réponse. Je n'ajoute pas une recherche jurisprudentielle "
                      "sans lacune juridique explicite."))

    unresolved_refs = [ref for ref in state.get("normative_references", [])
                       if ref.get("status", "unresolved") != "resolved"]
    if (decision.decision == Decision.call_tool
            and decision.next_tool != "search_quebec_regulations"
            and unresolved_refs
            and ctx.config.evidence_first_follow_normative_references
            and "search_quebec_regulations" in ctx.catalog.tools):
        decision.decision = Decision.call_tool
        decision.next_tool = "search_quebec_regulations"
        decision.arguments = {"query": str(
            unresolved_refs[0].get("reference_text", ""))}
        decision.decision_trace.need = (
            "renvoi normatif explicite non résolu; combler ce gap avant la jurisprudence")
        decision.decision_trace.next_action = "call_tool:search_quebec_regulations"
    proposed_regime = decision.legal_regime
    if locked and resolved:
        decision.jurisdiction = resolved
    elif decision.jurisdiction and not live:
        # Dataset seulement : les routes sont scriptées et le scénario fait
        # foi. En live, laisser la proposition du planner s'installer comme
        # juridiction résolue reviendrait à DEVINER la province — c'est
        # précisément ce que la clarification obligatoire empêche, et
        # resolve_jurisdiction reste l'unique écrivain de ce champ.
        updates["resolved_jurisdiction"] = decision.jurisdiction
        updates["jurisdiction_status"] = decision.jurisdiction
        resolved = decision.jurisdiction

    # 5d. Couverture de juridiction — quatre cas, quatre comportements.
    # Le comportement manquait : les catégories n'existaient que dans le
    # YAML des distributions.
    needs_regime = bool(
        live and intent == "legal"
        and state.get("employment_regime_material", False)
        and state.get("legal_regime", "unknown") == "unknown")
    needs_jurisdiction = bool(
        live and intent == "legal"
        and state.get("jurisdiction_material", False))
    if (needs_regime and decision.decision not in {
            Decision.ask_clarification, Decision.cannot_conclude}):
        decision = _forced(
            decision, resolved,
            need="régime juridique de l'emploi inconnu",
            thinking=("Le secteur de l'employeur peut déterminer les sources "
                      "fédérales ou provinciales à rechercher."),
            action=Decision.ask_clarification,
        )
        decision.clarification_scope = "legal_regime"
        decision.clarification_blocking = True
        decision.answerable_conditionally = False
        decision.clarification_fact_keys = (
            ["work_location", "employment_sector"]
            if not resolved else ["employment_sector"])
        decision.clarification_question = None
    elif (needs_jurisdiction
          and decision.decision != Decision.ask_clarification):
        action = coverage_action(
            resolved, federal_matter=_is_federal_matter(state, decision))
        clarifications = state.get("clarification_count", 0)
        clarification_limit = min(
            state.get("max_clarifications", ctx.config.max_clarifications_live),
            ctx.config.evidence_first_maximum_clarifications,
        )
        if action == "clarify" and clarifications < clarification_limit:
            decision = _forced(
                decision, resolved,
                need="juridiction inconnue",
                thinking=("La juridiction applicable n'est pas établie et le "
                          "droit varie d'une province à l'autre : je la "
                          "demande plutôt que de la supposer."),
                action=Decision.ask_clarification,
            )
            decision.clarification_scope = "jurisdiction"
            decision.clarification_blocking = True
            decision.answerable_conditionally = False
            decision.clarification_fact_keys = ["jurisdiction"]
            decision.clarification_question = None
        elif action == "decline":
            decision = _forced(
                decision, resolved,
                need="juridiction hors couverture",
                thinking=(f"La situation relève du droit de {resolved}, hors "
                          "du droit québécois et du droit fédéral canadien. "
                          "Je ne peux pas répondre sur ce fondement."),
                action=Decision.cannot_conclude)
            updates["stop_reason"] = "jurisdiction_not_covered"

    # 2bis. Faits critiques manquants — clarification FORCÉE.
    # Le vérificateur décide; le planner n'a pas à y penser de lui-même.
    missing = [fact for fact in state.get("missing_facts_before_search", [])
               if str(fact).strip()]
    # Ne pas redemander ce qui vient d'être répondu. Après une clarification
    # de juridiction, analyze_facts laisse « juridiction applicable » dans la
    # liste alors que resolve_jurisdiction l'a établie : le tour 2 redemandait
    # la province qu'on venait de recevoir, en affichant « Québec » dans la
    # même trace.
    if resolved and coverage_action(
            resolved,
            federal_matter=_is_federal_matter(state, decision)) != "clarify":
        missing = [fact for fact in missing
                   if not _RE_FAIT_JURIDICTION.search(str(fact))]
    if (not live and not ctx.config.evidence_first_enabled and missing
            and decision.decision not in (Decision.ask_clarification,
                                          Decision.cannot_conclude)
            and state.get("clarification_count", 0) < min(
                state.get("max_clarifications", 2),
                ctx.config.evidence_first_maximum_clarifications)
            and not state.get("tool_history")):
        decision = _forced(
            decision, resolved,
            need="faits critiques manquants",
            thinking=("Des faits indispensables manquent "
                      f"({', '.join(missing[:3])}) : je les demande avant de "
                      "chercher, sinon la réponse porterait sur une "
                      "situation supposée."),
            action=Decision.ask_clarification,
            question=question_for_fact_keys(missing[:3]))

    # Une règle conditionnellement compatible signale une lacune FACTUELLE,
    # distincte de la juridiction. La catégorie est enregistrée par
    # handle_clarification et empêche de répéter cette étape au tour suivant.
    structured_clarification = None
    if ctx.config.evidence_first_enabled:
        raw_contract = state.get("rule_contract") or {}
        contract = (raw_contract if isinstance(raw_contract, RuleContract)
                    else RuleContract.model_validate(raw_contract))
        structured_decision = build_clarification_decision(
            contract, state.get("facts") or {},
            state.get("clarification_history") or [],
            task_id=state.get("task_id", ""),
        )
        if structured_decision.needed and structured_decision.question:
            structured_clarification = {
                "clarification_id": f"rule-fact-{structured_decision.missing_fact_id}",
                "fact_keys": [str(structured_decision.missing_fact_id)],
                "question": structured_decision.question,
            }
    else:
        facts_context = dict((state.get("case_context") or {}).get(
            "facts") or state.get("facts") or {})
        user_statements = facts_context.get("user_statements") or []
        structured_clarification = build_clarification(
            state.get("article_reviews") or {}, facts_context,
            state.get("clarification_history") or [], user_statements)
    # Les clarifications issues du RuleContract portent sur l'application de
    # la règle. Elles alimentent la réponse conditionnelle mais ne suspendent
    # jamais le graphe.

    raw_contract_for_clarification = state.get("rule_contract") or {}
    contract_elements = (getattr(raw_contract_for_clarification, "elements", [])
                         if not isinstance(raw_contract_for_clarification, dict)
                         else raw_contract_for_clarification.get("elements", []))
    if (ctx.config.evidence_first_enabled
            and decision.decision == Decision.ask_clarification
            and decision.clarification_scope not in {
                "request_intent", "jurisdiction", "legal_regime"}
            and not decision.clarification_blocking
            and bool(contract_elements)
            and not structured_clarification):
        decision = _forced_final(
            decision, resolved,
            need="réponse conditionnelle possible; clarification non bloquante",
            thinking=("Le RuleContract ne dérive aucun fait bloquant et le point "
                      "peut être présenté par branches conditionnelles."))

    # 2. Clarification bornée.
    if decision.decision == Decision.ask_clarification:
        count = state.get("clarification_count", 0)
        if not live and count >= 1:
            return {
                "status": "rejected",
                "stop_reason": ("clarification répétée après la réponse "
                                "de l'utilisateur"),
            }
        if live and count >= min(
                state.get("max_clarifications", 2),
                ctx.config.evidence_first_maximum_clarifications):
            decision = _forced_final(
                decision, resolved,
                need="clarifications épuisées",
                thinking=("Deux clarifications ont déjà été posées; je "
                          "réponds au mieux avec les éléments connus."))

    if decision.decision == Decision.call_tool and decision.next_tool:
        # 3. Budget d'outils : dépassement → synthèse forcée.
        if len(state.get("tool_history", [])) >= state.get(
                "max_tool_calls", 4):
            decision = _forced_final(
                decision, resolved,
                need="budget d'outils épuisé",
                thinking=("Le budget d'appels d'outils est épuisé; je "
                          "synthétise à partir des résultats obtenus."))
            updates["stop_reason"] = "max_tool_calls"
        # 5b. Coverage gate — check tool availability/coverage.
        elif decision.next_tool:
            coverage = get_coverage(decision.next_tool)
            if coverage and not coverage.is_available(
                    "live" if live else "dataset"):
                gap = CoverageGap(
                    requested_document_type=(
                        coverage.document_types[0]
                        if coverage.document_types else ""),
                    requested_court_scope=(
                        coverage.court_scopes[0]
                        if coverage.court_scopes else ""),
                    requested_jurisdiction=(
                        coverage.legal_jurisdictions[0]
                        if coverage.legal_jurisdictions else ""),
                    reason=(coverage.availability_reason
                            or f"{decision.next_tool} unavailable"),
                )
                gaps = list(state.get("coverage_gaps", []))
                gaps.append(gap.to_dict())
                decision = _forced_final(
                    decision, resolved,
                    need="outil indisponible",
                    thinking=(
                        f"L'outil « {decision.next_tool} » n'est pas "
                        f"disponible : {coverage.availability_reason}. "
                        "Je signale la limite de couverture."))
                updates["coverage_gaps"] = gaps

        # 5c. Outils québécois bloqués hors Québec (déterministe).
        if (decision.decision == Decision.call_tool
                and decision.next_tool
                and live
                and decision.next_tool in QC_ONLY_TOOLS
                and not allows_quebec_tools(resolved)):
            decision = _forced_final(
                decision, resolved,
                need="outils québécois inapplicables hors Québec",
                thinking=(f"L'utilisateur n'est pas au Québec ({resolved}) :"
                          " les outils CCQ/CPC ne s'appliquent pas. Je "
                          "réponds avec le droit fédéral applicable."))
        # 4. Structure + politique de route (dataset uniquement — les
        # routes scriptées n'existent pas en live).
        elif not live:
            proposal = ctx.services.validation.verify_proposal(
                decision,
                state["scenario"].request_type,
                state.get("tool_history", []),
                state.get("max_tool_calls", 4),
            )
            if proposal.verdict == ProposalVerdict.reject:
                return {
                    "status": "rejected",
                    "stop_reason": "; ".join(proposal.errors),
                    "deterministic_blockers": list(proposal.errors),
                }
            decision = proposal.decision

    if decision.decision in (Decision.final_answer,
                             Decision.cannot_conclude):
        updates.setdefault("stop_reason",
                           state.get("stop_reason") or
                           decision.decision.value)

    if decision.decision == Decision.ask_clarification and live:
        pending = _pending_clarification(
            state, decision, ctx,
            evidence_first=ctx.config.evidence_first_enabled)
        updates["pending_clarification"] = pending
        context = dict(state.get("case_context") or {})
        context["pending_clarification"] = pending
        updates["case_context"] = context

    if proposed_regime in {"federal", "provincial"}:
        updates.setdefault("planner_proposed_legal_regime", proposed_regime)

    # Budget PROPRE à la reprise jurisprudentielle. Il partageait
    # « reformulation_count » avec la reformulation de recherche
    # sémantique, dont le maximum est 1 : une seule reprise ici épuisait
    # tout le budget de reformulation, et route_after_classification ne
    # pouvait plus jamais reformuler un résultat non pertinent.
    if (live and decision.decision == Decision.call_tool
            and decision.next_tool == "search_quebec_jurisprudence"
            and state.get("case_law_search_status") in {
                "irrelevant", "empty", "failed", "tool_error",
                "coverage_gap", "candidates_without_url"}
            and state.get("case_law_retry_count", 0) < state.get(
                "max_search_reformulations", 1)):
        updates["case_law_retry_count"] = state.get(
            "case_law_retry_count", 0) + 1

    _retirer_numeros_des_recherches(decision)
    updates["latest_decision"] = decision.model_dump(mode="json")
    return updates
