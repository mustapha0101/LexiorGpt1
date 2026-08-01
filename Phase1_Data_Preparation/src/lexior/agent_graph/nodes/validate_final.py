# -*- coding: utf-8 -*-
"""validate_final — validation déterministe de la trajectoire complète.

Dataset et live : une affirmation juridique non soutenue par le texte
récupéré est un bloqueur. Le live conserve les avertissements de qualité,
mais ne livre jamais une règle que le contrôle factuel a invalidée.
"""

from __future__ import annotations

import re

from typing import Any

from lexior.agentic.error_codes import ErrorCode, tag
from lexior.services.assertion_grounding import (
    VerdictAffirmation,
    textes_recuperes,
)
from lexior.services.evidence_first import (
    build_claim_ledger,
    merge_failures,
    retrieved_articles,
)
from lexior.agentic.schemas import PrimaryAuthoritySelection
from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import (
    LexiorState,
    canonical_case_description,
    to_trajectory,
    visible_tool_history,
)

NAME = "validate_final"


_RE_REGLE_ENONCEE = re.compile(
    r"\barticles?\s+\d{1,4}"                 # « l'article 1465 »
    r"|\ble\s+Code\s+civil\s+(?:du\s+Québec\s+)?(?:stipule|prévoit|dispose)"
    r"|\bla\s+loi\s+(?:québécoise|canadienne)\s+(?:stipule|prévoit|dispose)"
    r"|\bla\s+responsabilité\s+(?:civile\s+)?(?:peut\s+être\s+engagée|"
    r"est\s+engagée|doit\s+être\s+engagée)"
    r"|\best\s+tenu\s+de\s+réparer"
    r"|\bsauf\s+s'?il\s+prouve", re.IGNORECASE)
_RE_ARTICLE_NUMBER = re.compile(r"\d{1,4}(?:\.\d+)?")
_RE_CITED_ARTICLE = re.compile(
    r"\b(?:articles?|art\.)\s*(\d{1,4}(?:\.\d+)?)", re.IGNORECASE)


def _source_bounded_fallback(textes: dict[str, str],
                             articles: list[str]) -> str:
    """Produit un repli utile sans transformer une source en conclusion.

    Il ne s'agit pas d'une nouvelle analyse juridique : le texte officiel est
    reproduit tel que rÃ©cupÃ©rÃ© et l'application aux faits reste explicitement
    ouverte. Cette sortie gÃ©nÃ©rique Ã©vite qu'une paraphrase non soutenue
    transforme une rÃ©ponse de droit en erreur technique pour l'utilisateur.
    """
    blocs = [
        f"Article {numero}\n{textes[numero].strip()}"
        for numero in articles if numero in textes and textes[numero].strip()
    ]
    if not blocs:
        return ""
    return (
        "Je ne peux pas d\u00e9terminer, \u00e0 partir des seuls faits fournis, si les "
        "conditions de responsabilit\u00e9 sont r\u00e9unies. Voici le texte officiel "
        "r\u00e9cup\u00e9r\u00e9 qui peut servir \u00e0 l'analyse :\n\n"
        + "\n\n".join(blocs[:2])
        + "\n\nConservez les \u00e9l\u00e9ments factuels et les communications utiles; "
          "un professionnel peut ensuite appliquer ces textes \u00e0 votre situation."
    )


def _conditional_evidence_fallback(
        textes: dict[str, str], contract: dict[str, Any]) -> str:
    """Return a source-bounded conditional answer before the generic fallback."""
    entries = contract.get("conditional_reasoning_contract") or []
    retained = {str(number) for number in contract.get("articles_retenus", [])}
    entries = [entry for entry in entries
               if str(entry.get("article", "")) in retained
               and str(entry.get("article", "")) in textes]
    if not entries:
        return ""
    paragraphs = [
        "Les textes officiels retenus peuvent être pertinents pour analyser la situation, mais ils ne permettent pas à eux seuls de conclure à une responsabilité.",
    ]
    asserted: list[str] = []
    unresolved: list[str] = []
    for entry in entries:
        for item in entry.get("user_asserted_facts", []):
            fact = str(item.get("fact", "")).strip()
            if fact and fact not in asserted:
                asserted.append(fact)
        for condition in entry.get("unresolved_conditions", []):
            condition = str(condition).strip()
            if condition and condition not in unresolved:
                unresolved.append(condition)
        article = str(entry.get("article", ""))
        proposition = (entry.get("source_propositions") or [])
        proposition = str(proposition[0]).strip() if proposition else textes[article].strip()
        if proposition:
            paragraphs.append(f"Article {article} : le texte officiel énonce notamment : {proposition}")
    if asserted:
        paragraphs.append("Vous avez affirmé, sans vérification indépendante : "
                          + "; ".join(asserted) + ".")
    if unresolved:
        paragraphs.append("Il reste notamment à établir : " + "; ".join(unresolved) + ".")
    paragraphs.append(
        "La responsabilité n'est donc pas automatique : l'application dépend des faits à établir et du lien avec le préjudice. Conservez les photos, les échanges, la date et les estimations utiles.")
    return "\n\n".join(paragraphs)


def _safe_unresolved_fallback() -> str:
    """Réponse sûre quand aucun texte vérifié ne peut être affiché.

    Le contrôle a écarté le brouillon et le contrat ne contient aucune
    disposition autorisée. Le flux live doit rester utile, sans déguiser ce
    manque de preuve en erreur technique ni tirer de conclusion juridique.
    """
    return (
        "Je ne peux pas déterminer, à partir des faits et des sources "
        "vérifiées disponibles, si les conditions juridiques sont réunies. "
        "Conservez les photos, la date, les échanges et les estimations de "
        "réparation; un professionnel pourra ensuite examiner ces éléments "
        "et les sources officielles applicables."
    )


def _articles_cites(reponse: str) -> list[str]:
    return list(dict.fromkeys(_RE_CITED_ARTICLE.findall(reponse or "")))


def _regle_sans_source(state: LexiorState) -> list[str]:
    """Une règle énoncée alors qu'aucune source n'a été retenue.

    Le contrat porte déjà la directive « n'affirme aucune règle de fond »
    quand answer_mode vaut no_evidence. Le modèle l'a ignorée : sur une
    question de morsure de chien, il a énoncé la règle de l'article 1465
    (« sauf s'il prouve n'avoir commis aucune faute ») ET écrit « je n'ai pas
    trouvé d'articles spécifiques » dans la même réponse. Les deux ne peuvent
    pas être vraies. Une consigne de plus ne suffira pas — il en existait
    déjà une, comme « jamais de loi de mémoire » — donc on le constate.
    """
    contrat = state.get("answer_contract") or {}
    # ``answer_mode`` est le nom historique, conservé pour les trajectoires
    # déjà exportées; les nouveaux contrats utilisent ``mode_de_reponse``.
    mode = contrat.get("mode_de_reponse", contrat.get("answer_mode"))
    if mode != "no_evidence":
        return []
    reponse = state.get("final_answer") or ""
    if not _RE_REGLE_ENONCEE.search(reponse):
        return []
    return [tag(ErrorCode.ANSWER_FROM_MEMORY,
                "règle de fond énoncée alors qu'aucune preuve utilisable "
                "n'a été retenue")]


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    live = is_live(state.get("mode", ""))
    trajectory = to_trajectory(state)
    exempt = state.get("exempt_tools") or (
        ctx.services.validation.compute_exempt_tools(
            state.get("tool_history", [])))

    validation = ctx.services.validation.validate_trajectory(
        trajectory,
        allow_mock=(ctx.config.offline or ctx.config.dry_run or live),
        max_tool_calls=state.get("max_tool_calls", 4),
        exempt_tools=exempt,
    )
    validation.warnings.extend(ctx.services.validation.sequence_warnings(
        state["scenario"].request_type,
        [o.tool_name for o in state.get("tool_history", [])],
    ))
    # Une règle tirée de la mémoire alors que le contrat ne retient aucune
    # preuve est une erreur de fond. La laisser parmi les avertissements
    # permettait au flux live de livrer précisément la réponse interdite.
    memory_blockers = _regle_sans_source(state)
    validation.errors.extend(memory_blockers)

    # Le NUMÉRO cité est vérifié ailleurs ; ici c'est l'AFFIRMATION qui est
    # confrontée au texte réellement récupéré. Un échec technique devient une
    # erreur : ne pas avoir pu vérifier n'est pas avoir vérifié.
    # Les faits sont nécessaires : les conditions temporelles, matérielles et
    # personnelles d'une règle doivent correspondre à la situation décrite.
    textes = textes_recuperes(visible_tool_history(state))
    reponse_finale = state.get("final_answer") or ""
    verdicts = ctx.services.assertion_grounding.verifier(
        reponse_finale,
        textes,
        faits=canonical_case_description(state))
    contract = state.get("answer_contract") or {}
    if contract.get("filtre_articles_officiels"):
        retained = {str(numero) for numero in contract.get(
            "articles_retenus", [])}
        for numero in _articles_cites(reponse_finale):
            if numero in textes and numero not in retained:
                verdicts.append(VerdictAffirmation(
                    article=numero,
                    affirmation=f"citation de l'article {numero}",
                    soutenue=False,
                    motif="l'article est exclu du contrat d'applicabilité",
                ))
    grounding_blockers: list[str] = []
    grounding_failures: list[dict[str, str]] = []
    for verdict in verdicts:
        if not verdict.soutenue:
            problem = tag(ErrorCode.UNGROUNDED_ARTICLE, verdict.probleme())
            validation.errors.append(problem)
            grounding_blockers.append(problem)
            grounding_failures.append({
                "claim": str(verdict.affirmation or ""),
                "failure_type": "ungrounded_article",
                "source_article": str(verdict.article or ""),
                "reason": str(verdict.motif or ""),
            })

    selection = state.get("primary_authority_selection")
    if isinstance(selection, dict):
        selection = PrimaryAuthoritySelection.model_validate(selection)
    elif selection is None:
        selection = PrimaryAuthoritySelection(task_id=state.get("task_id", ""))
    source_texts = {
        sid: item[0]
        for sid, item in retrieved_articles(visible_tool_history(state)).items()
    }
    for observation in visible_tool_history(state):
        if (observation.ok and observation.normalized_response.strip()
                and not ctx.catalog.is_local(observation.tool_name)
                and observation.tool_name not in {
                    "semantic_search_ccq", "semantic_search_cpc",
                    "search_quebec_jurisprudence", "search_quebec_regulations",
                }):
            source_texts.setdefault(
                f"tool:{observation.tool_name}:{observation.content_hash}",
                observation.normalized_response)
    ledger = build_claim_ledger(
        reponse_finale, selection, source_texts,
        task_id=state.get("task_id", ""),
    )
    ledger_failures = [
        {
            "claim": claim.text,
            "failure_type": "ungrounded_claim",
            "reason": claim.failure_reason or "affirmation juridique non vérifiée",
        }
        for claim in ledger.claims if claim.verification_status == "failed"
    ]
    grounding_failures = merge_failures(
        state.get("grounding_failures", []),
        [*grounding_failures, *ledger_failures], node=NAME)

    # A failed free-form paraphrase or a rule from memory must not become a
    # technical error shown to the user. In live mode, replace it with an
    # evidence-only fallback; it asserts no application of the rule.
    fallback = ""
    safety_blockers = memory_blockers + grounding_blockers
    if safety_blockers and live:
        cited_numbers = list(dict.fromkeys(
            number
            for verdict in verdicts if not verdict.soutenue
            for number in _RE_ARTICLE_NUMBER.findall(verdict.article or "")
        ))
        if contract.get("filtre_articles_officiels"):
            # Le contrat résulte de la revue d'applicabilité : son ordre est
            # plus fiable que les citations du brouillon que l'on vient de
            # rejeter. Le repli ne doit donc pas perpétuer une citation
            # marginale ou exclue simplement parce qu'elle figurait dans la
            # réponse invalide.
            cited_numbers = [
                str(number) for number in contract.get("articles_retenus", [])
                if str(number) in textes
            ]
        fallback = (
            ("" if ctx.config.evidence_first_enabled
             else _conditional_evidence_fallback(textes, contract))
            or _source_bounded_fallback(textes, cited_numbers)
            or _safe_unresolved_fallback())
        validation.errors = [error for error in validation.errors
                             if error not in safety_blockers]
        memory_blockers = []
        grounding_blockers = []

    return {
        "thread_id": state.get("thread_id", ""),
        "task_id": state.get("task_id", ""),
        "claim_events": [
            {"event_name": "claim_verified" if claim.verification_status == "verified"
             else "claim_failed", "claim_id": claim.claim_id,
             "status": claim.verification_status}
            for claim in ledger.claims
        ],
        **({"final_answer": fallback,
            "final_reasoning_summary": ""} if fallback else {}),
        "validation_result": validation,
        "validation_issues": list(validation.errors)
        + list(validation.warnings),
        "grounding_failures": grounding_failures,
        "claim_ledger": ledger,
        "failure_history": merge_failures(
            state.get("failure_history", []), grounding_failures, node=NAME),
        # Le dataset reste strict sur tous les validateurs. En live, seuls
        # les échecs de grounding d'une affirmation juridique bloquent : une
        # erreur de route historique ne doit pas transformer une réponse
        # générale non juridique en panne utilisateur.
        "deterministic_blockers": (
            grounding_blockers if live else list(validation.errors)),
        "deterministic_validation": bool(validation.valid),
        "exempt_tools": exempt,
    }
