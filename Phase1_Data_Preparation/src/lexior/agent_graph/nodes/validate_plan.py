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

from lexior.agentic.schemas import Decision, DecisionTrace, PlannerDecision
from lexior.services.evidence import AcceptanceBlocker, CoverageGap
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
from lexior.services.tool_coverage import get_coverage, has_equivalent_coverage
from lexior.services.validation import ProposalVerdict

from ..context import GraphContext
from ..state import LexiorState

NAME = "validate_plan"

# Faits manquants qui portent sur la juridiction : une fois celle-ci résolue,
# les redemander revient à ignorer la réponse de l'usager.
_RE_FAIT_JURIDICTION = re.compile(r"juridiction|province", re.IGNORECASE)

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
    return PlannerDecision(
        request_type=decision.request_type,
        jurisdiction=jurisdiction,
        missing_critical_facts=decision.missing_critical_facts,
        required_sources=decision.required_sources,
        decision=action,
        clarification_question=question or decision.clarification_question,
        thinking_text=thinking,
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
                           decision: PlannerDecision) -> dict[str, Any]:
    """Construit le contrat rendu à l'utilisateur avant l'interrupt()."""
    question = (decision.clarification_question or "").strip()
    if _RE_FAIT_JURIDICTION.search(question):
        return {
            "clarification_id": "jurisdiction-province",
            "category": "jurisdiction",
            "fact_keys": ["jurisdiction"],
            "question": question or (
                "Dans quelle province êtes-vous? La réponse dépend du droit applicable."),
            "answer_type": "province_or_federal",
            "source_articles": [],
            "status": "pending",
        }

    context = state.get("case_context") or {}
    facts = dict(context.get("facts") or state.get("facts") or {})
    statements = facts.get("user_statements") or []
    selected = build_clarification(
        state.get("article_reviews") or {}, facts,
        state.get("clarification_history") or [], statements)
    if selected:
        return selected

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
        "source_articles": [],
        "status": "pending",
    }


def _is_federal_matter(state: LexiorState,
                       decision: PlannerDecision) -> bool:
    """Le droit applicable est-il fédéral, quelle que soit la province ?"""
    return (is_federal(state.get("expected_jurisdiction", ""))
            or is_federal(decision.jurisdiction))


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
    if live and decision.decision != Decision.ask_clarification:
        action = coverage_action(
            resolved, federal_matter=_is_federal_matter(state, decision))
        clarifications = state.get("clarification_count", 0)
        if action == "clarify" and clarifications < 2:
            decision = _forced(
                decision, resolved,
                need="juridiction inconnue",
                thinking=("La juridiction applicable n'est pas établie et le "
                          "droit varie d'une province à l'autre : je la "
                          "demande plutôt que de la supposer."),
                action=Decision.ask_clarification,
                question=("Dans quelle province êtes-vous? La réponse dépend "
                          "du droit applicable."))
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
    if (live and missing
            and decision.decision not in (Decision.ask_clarification,
                                          Decision.cannot_conclude)
            and state.get("clarification_count", 0) < state.get(
                "max_clarifications", 2)
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
    facts_context = dict((state.get("case_context") or {}).get(
        "facts") or state.get("facts") or {})
    user_statements = facts_context.get("user_statements") or []
    structured_clarification = build_clarification(
        state.get("article_reviews") or {}, facts_context,
        state.get("clarification_history") or [], user_statements)
    if (live and state.get("request_type") == "case_analysis"
            and structured_clarification
            and state.get("clarification_count", 0) < state.get(
                "max_clarifications", 2)
            and decision.decision not in (Decision.ask_clarification,
                                          Decision.cannot_conclude)):
        decision = _forced(
            decision, resolved,
            need="fait matériel requis par la règle revue",
            thinking=("Un texte officiel est potentiellement pertinent, mais "
                      "sa revue indique qu'un fait nécessaire reste à établir."),
            action=Decision.ask_clarification,
            question=structured_clarification["question"])

    # 2. Clarification bornée.
    if decision.decision == Decision.ask_clarification:
        count = state.get("clarification_count", 0)
        if not live and count >= 1:
            return {
                "status": "rejected",
                "stop_reason": ("clarification répétée après la réponse "
                                "de l'utilisateur"),
            }
        if live and count >= state.get("max_clarifications", 2):
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
        pending = _pending_clarification(state, decision)
        updates["pending_clarification"] = pending
        context = dict(state.get("case_context") or {})
        context["pending_clarification"] = pending
        updates["case_context"] = context

    if (live and decision.decision == Decision.call_tool
            and decision.next_tool == "search_quebec_jurisprudence"
            and state.get("case_law_search_status") in {
                "irrelevant", "empty", "failed", "tool_error",
                "coverage_gap", "candidates_without_url"}
            and state.get("reformulation_count", 0) < state.get(
                "max_search_reformulations", 1)):
        updates["reformulation_count"] = state.get(
            "reformulation_count", 0) + 1

    _retirer_numeros_des_recherches(decision)
    updates["latest_decision"] = decision.model_dump(mode="json")
    return updates
