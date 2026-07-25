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

from typing import Any

from lexior.agentic.schemas import Decision, DecisionTrace, PlannerDecision
from lexior.services.evidence import AcceptanceBlocker, CoverageGap
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
    max_steps = state.get("max_tool_calls", 4) + 2

    # 1. Borne de décisions du planner.
    if step > max_steps:
        return {
            "status": "rejected",
            "stop_reason": (
                f"limite de {max_steps} décisions Planner atteinte"),
        }

    # 5a. Juridiction autoritaire : la valeur résolue écrase la
    # proposition; en dataset, la proposition du planner raffine la
    # valeur non verrouillée (comportement historique).
    resolved = state.get("resolved_jurisdiction", "")
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
    if (live and missing
            and decision.decision not in (Decision.ask_clarification,
                                          Decision.cannot_conclude)
            and state.get("clarification_count", 0) < 2
            and not state.get("tool_history")):
        decision = _forced(
            decision, resolved,
            need="faits critiques manquants",
            thinking=("Des faits indispensables manquent "
                      f"({', '.join(missing[:3])}) : je les demande avant de "
                      "chercher, sinon la réponse porterait sur une "
                      "situation supposée."),
            action=Decision.ask_clarification,
            question=(f"Pour répondre précisément, il me manque : "
                      f"{', '.join(missing[:3])}. Pouvez-vous préciser?"))

    # 2. Clarification bornée.
    if decision.decision == Decision.ask_clarification:
        count = state.get("clarification_count", 0)
        if not live and count >= 1:
            return {
                "status": "rejected",
                "stop_reason": ("clarification répétée après la réponse "
                                "de l'utilisateur"),
            }
        if live and count >= 2:
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

    updates["latest_decision"] = decision.model_dump(mode="json")
    return updates
