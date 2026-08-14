# -*- coding: utf-8 -*-
"""Planner : une seule prochaine action, validée contre le catalogue."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .prompts import CHAT_PLANNER_SUPPLEMENT, planner_system_prompt
from .taxonomy_conditions import (
    GardeContexte, etape_facultative_retenue, juridiction_compatible,
)
from lexior.services.provenance import (
    a_une_provenance, numero_demande, numeros_demandes, reponses_reussies,
)
from lexior.services.article_review import (
    assess_legislative_sufficiency,
    format_facts_for_query,
)
from lexior.services.evidence_first import article_budget
from lexior.services.evidence_first import normalize_and_repair_tool_args
from .schemas import Decision, DecisionTrace, PlannerDecision, ResearchState
from .tool_catalog import MAX_ARTICLES_PAR_APPEL, ToolCatalog
from .validators import validate_next_action, validate_planner_decision

ARTICLE_RE = re.compile(r"\b(?:article\s+)?(\d{1,4}(?:\.\d+)?)\b", re.I)
ARTICLE_LABEL_RE = re.compile(r"\barticle\s+(\d{1,4}(?:\.\d+)?)\b", re.I)
NO_RESULT_RE = re.compile(
    r"(?:aucun(?:e)?\s+(?:article|résultat|document|décision)|"
    r"rien\s+trouvé|no\s+results?)",
    re.I,
)

# ── Garde-fou juridictionnel du mode chat ────────────────────────────────
# Implémentation canonique : lexior.services.jurisdiction (une seule
# source de vérité; ré-exportée ici pour compatibilité).
from .citations import CASE_CITATION_RE  # noqa: E402
from lexior.services.jurisdiction import (  # noqa: E402
    QC_ONLY_TOOLS,
    QuebecToolsBlocked,
    detect_jurisdiction_hint,
)

# ---------------------------------------------------------------------------
# Thinking-text extraction helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchIntent:
    keywords: list[str]
    target_type: str   # "cases" | "laws" | "auto"
    case_name: str

_CASE_NAME_RE = re.compile(
    r"(?:affaire|cause|décision|arrêt)\s+"
    r"([A-ZÀ-Ÿ][\w'-]+(?:\s+(?:c\.|contre|v\.)\s+[A-ZÀ-Ÿ][\w'-]+)?)"
    r"|([A-ZÀ-Ÿ][\w'-]+)\s+(?:c\.|contre|v\.)\s+([A-ZÀ-Ÿ][\w'-]+)",
    re.UNICODE,
)

_CASE_INDICATOR_WORDS = frozenset({
    "jurisprudence", "décision", "décisions", "arrêt", "arrêts",
    "jugement", "jugements", "cause", "causes", "affaire", "affaires",
    "tribunal", "tribunaux", "cour",
})

_LAW_INDICATOR_WORDS = frozenset({
    "loi", "lois", "code", "article", "articles", "disposition",
    "dispositions", "législatif", "législative", "texte", "statut",
})

_THINKING_STOPWORDS = frozenset({
    "avec", "avoir", "cela", "ceci", "cette", "comme", "comment", "dans",
    "donner", "elle", "elles", "entre", "faire", "leur", "leurs", "mais",
    "même", "nous", "notre", "plus", "pour", "pouvez", "quel", "quelle",
    "quelles", "quels", "quoi", "sans", "sont", "sous", "suis", "aussi",
    "après", "avant", "bien", "chez", "donc", "alors", "être", "très",
    "tout", "toute", "tous", "vers", "cette", "savoir", "matière",
    "partie", "encore", "déjà",
    "chercher", "rechercher", "trouver", "identifier", "utiliser",
    "appeler", "lancer", "outil", "outils", "recherche", "résultat",
    "résultats", "doit", "dois", "devrait", "peut", "peux", "faut",
    "nécessaire", "information", "informations", "question", "demande",
    "utilisateur", "pertinent", "pertinente", "pertinents", "premier",
    "première", "suivant", "suivante", "tour", "étape", "récupérer",
    "obtenir", "besoin", "passer", "concernant",
})

FEDERAL_STATUTES = (
    (("faillit", "insolv"), "Bankruptcy and Insolvency Act",
     ("bankruptcy and insolvency act", "loi sur la faillite et l insolvabilite")),
    (("banque", "bancaire"), "Bank Act",
     ("bank act", "loi sur les banques")),
    (("marque",), "Trademarks Act",
     ("trademarks act", "loi sur les marques de commerce")),
    (("brevet",), "Patent Act",
     ("patent act", "loi sur les brevets")),
    (("maritime",), "Marine Liability Act",
     ("marine liability act", "loi sur la responsabilite en matiere maritime")),
    (("immigr", "réfugié", "asile"), "Immigration and Refugee Protection Act",
     ("immigration and refugee protection act",
      "loi sur l immigration et la protection des refugies")),
    (("criminel", "infraction", "pénale", "meurtre", "vol qualifié", "agression"),
     "Criminal Code",
     ("criminal code", "code criminel")),
    (("droit d'auteur", "copyright"), "Copyright Act",
     ("copyright act", "loi sur le droit d auteur")),
    (("concurrence", "antitrust"), "Competition Act",
     ("competition act", "loi sur la concurrence")),
    (("impôt", "fiscal", "revenu"), "Income Tax Act",
     ("income tax act", "loi de l impot sur le revenu")),
    (("travail", "normes du travail fédéral", "congédiement fédéral"),
     "Canada Labour Code",
     ("canada labour code", "code canadien du travail")),
    (("environnement", "pollution"), "Canadian Environmental Protection Act, 1999",
     ("canadian environmental protection act",
      "loi canadienne sur la protection de l environnement")),
    (("divorce",), "Divorce Act",
     ("divorce act", "loi sur le divorce")),
    (("douane", "tarif", "importation", "exportation"), "Customs Act",
     ("customs act", "loi sur les douanes")),
    (("drogue", "stupéfiant", "cannabis", "substance contrôlée"),
     "Controlled Drugs and Substances Act",
     ("controlled drugs and substances act",
      "loi reglementant certaines drogues et autres substances")),
    (("transport", "aérien", "aviation"), "Canada Transportation Act",
     ("canada transportation act", "loi sur les transports au canada")),
    (("société", "entreprise fédérale", "société par actions"),
     "Canada Business Corporations Act",
     ("canada business corporations act",
      "loi canadienne sur les societes par actions")),
    (("pension", "retraite", "rpc"), "Canada Pension Plan",
     ("canada pension plan", "regime de pensions du canada")),
    (("assurance-emploi", "chômage", "prestation"), "Employment Insurance Act",
     ("employment insurance act", "loi sur l assurance emploi")),
    (("télécommunication", "radiodiffusion", "crtc"),
     "Telecommunications Act",
     ("telecommunications act", "loi sur les telecommunications")),
    (("accès à l'information", "renseignements personnels", "vie privée"),
     "Privacy Act",
     ("privacy act", "loi sur la protection des renseignements personnels")),
)

FEDERAL_KNOWN_CITATIONS: dict[str, str] = {
    "Bankruptcy and Insolvency Act": "LRC 1985, c B-3",
    "Bank Act": "LC 1991, c 46",
    "Trademarks Act": "LRC 1985, c T-13",
    "Patent Act": "LRC 1985, c P-4",
    "Marine Liability Act": "LC 2001, c 6",
    "Immigration and Refugee Protection Act": "LC 2001, c 27",
    "Criminal Code": "LRC 1985, c C-46",
    "Copyright Act": "LRC 1985, c C-42",
    "Competition Act": "LRC 1985, c C-34",
    "Income Tax Act": "LRC 1985, c 1 (5e suppl)",
    "Canada Labour Code": "LRC 1985, c L-2",
    "Canadian Environmental Protection Act, 1999": "LC 1999, c 33",
    "Divorce Act": "LRC 1985, c 3 (2e suppl)",
    "Customs Act": "LRC 1985, c 1 (2e suppl)",
    "Controlled Drugs and Substances Act": "LC 1996, c 19",
    "Canada Transportation Act": "LC 1996, c 10",
    "Canada Business Corporations Act": "LRC 1985, c C-44",
    "Canada Pension Plan": "LRC 1985, c C-8",
    "Employment Insurance Act": "LC 1996, c 23",
    "Telecommunications Act": "LC 1993, c 38",
    "Privacy Act": "LRC 1985, c P-21",
}


class PlannerAgent:
    def __init__(self, catalog: ToolCatalog, client=None, offline: bool = False,
                 chat_mode: bool = False, *, initial_article_fetch_k: int = 6,
                 article_fetch_batch_size: int = 6,
                 max_articles_per_issue: int = 20,
                 evidence_first_enabled: bool = False,
                 evidence_first_initial_candidate_count: int = 5,
                 evidence_first_initial_fetch_count: int = 3,
                 evidence_first_maximum_article_batches: int = 2):
        self.catalog = catalog
        self.client = client
        self.offline = offline
        # chat_mode : requête libre sans route scriptée — les gardes fondés
        # sur expected_route/request_type ne s'appliquent pas.
        self.chat_mode = chat_mode
        self.last_live_normalization: dict[str, object] = {}
        self.initial_article_fetch_k = max(1, min(
            int(initial_article_fetch_k), MAX_ARTICLES_PAR_APPEL))
        self.article_fetch_batch_size = max(1, min(
            int(article_fetch_batch_size), MAX_ARTICLES_PAR_APPEL))
        self.max_articles_per_issue = max(1, min(
            int(max_articles_per_issue), MAX_ARTICLES_PAR_APPEL))
        self.evidence_first_enabled = bool(evidence_first_enabled)
        self.evidence_first_initial_candidate_count = int(
            evidence_first_initial_candidate_count)
        self.evidence_first_initial_fetch_count = int(
            evidence_first_initial_fetch_count)
        self.evidence_first_maximum_article_batches = int(
            evidence_first_maximum_article_batches)

    def decide(self, state: ResearchState) -> PlannerDecision:
        if self.offline:
            decision = self._offline_decide(state)
            decision = self._raise_if_invalid(state, decision)
            return decision
        feedback = ""
        while True:
            try:
                decision = self._teacher_decide(state, feedback=feedback)
                if not self.chat_mode:
                    decision = self._guard_clarification(state, decision)
                decision = self._guard_federal_fetch(state, decision)
                decision = self._guard_failed_tool(state, decision)
                if (decision.decision == Decision.call_tool
                        and decision.next_tool and not self.chat_mode):
                    # chat : les arguments du modèle passent tels quels; la
                    # reconstruction déterministe est calibrée pour les
                    # scénarios scriptés et détruit les requêtes libres.
                    decision = self._validate_arguments(state, decision)
                if not self.chat_mode:
                    decision = self._guard_tool_compatibility(state, decision)
                    decision = self._guard_required_tools(state, decision)
                else:
                    decision = self._guard_chat_jurisdiction(
                        state, decision, retried=bool(feedback))
                    decision = self._guard_progressive_article_review(
                        state, decision)
                    decision = self._guard_live_tool_chain_compatibility(
                        state, decision)
                # Toute comparaison de doublon porte sur les arguments finaux
                # du contrat outil, après legal_terms, candidats et schéma.
                decision = self._prepare_decision_arguments(state, decision)
                decision = self._guard_duplicate_call(state, decision)
                decision = self._guard_budget(state, decision)
                decision = self._raise_if_invalid(state, decision, prepare=False)
                return decision
            except ValueError as exc:
                # ValidationError pydantic incluse (sous-classe de ValueError).
                if feedback:
                    if (self.chat_mode
                            and self._has_usable_official_evidence(state)):
                        return self._safe_final_after_invalid_live_call(
                            state, decision)
                    raise
                feedback = str(exc) or "décision invalide"

    def _raise_if_invalid(self, state: ResearchState,
                          decision: PlannerDecision,
                          *, prepare: bool = True) -> PlannerDecision:
        # Les appels directs historiques à ce helper attendent encore que les
        # candidats soient attachés ici. Le flux planner principal passe
        # ``prepare=False`` : il prépare une seule fois avant déduplication.
        if prepare:
            decision = self._prepare_decision_arguments(state, decision)
        errors = validate_planner_decision(decision, self.catalog)
        if (decision.decision == Decision.call_tool and decision.next_tool
                and not self.chat_mode):
            errors.extend(validate_next_action(
                state.scenario.request_type, decision.next_tool))
        if (state.scenario.expected_route.no_tool
                and decision.decision == Decision.call_tool
                and not self.chat_mode):
            errors.append("appel d'outil interdit pour cette demande")
        if errors:
            if self.chat_mode and all("outil inconnu" in e for e in errors):
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    missing_critical_facts=decision.missing_critical_facts,
                    required_sources=decision.required_sources,
                    decision=Decision.final_answer,
                    thinking_text=(
                        f"L'outil « {decision.next_tool} » n'est pas "
                        "disponible en mode chat. Je réponds avec les "
                        "informations déjà obtenues."),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="outil indisponible en chat",
                        next_action="final_answer"),
                )
            raise ValueError("décision Planner invalide : " + "; ".join(errors))
        return decision

    def _prepare_decision_arguments(
            self, state: ResearchState,
            decision: PlannerDecision) -> PlannerDecision:
        """Prépare une décision exactement une fois avant les guards d'arguments."""
        self._attach_semantic_legal_terms(decision)
        self._attach_retrieved_candidates(state, decision)
        self.last_live_normalization = {}
        if (decision.decision == Decision.call_tool and decision.next_tool):
            normalized_args, audit, _errors = normalize_and_repair_tool_args(
                self.catalog, decision.next_tool, decision.arguments,
                active_task={"normalized_query": state.case_description,
                             "summary": state.case_description},
                latest_user_message=state.scenario.user_query,
                user_messages=[
                    message.content for message in state.messages
                    if getattr(message.role, "value", message.role) == "user"
                ],
            )
            decision.arguments = normalized_args
            if audit.get("removed_fields") or audit.get("repaired_fields"):
                self.last_live_normalization = {
                    "tool": decision.next_tool,
                    "removed_fields": list(audit.get("removed_fields", [])),
                    "repaired_fields": list(audit.get("repaired_fields", [])),
                    "remaining_arguments": dict(normalized_args),
                }
        return decision

    @staticmethod
    def _has_usable_official_evidence(state: ResearchState) -> bool:
        return bool(state.official_rule_retrieved or any(
            observation.ok and observation.tool_name in {
                "get_ccq_articles", "get_cpc_articles"}
            for observation in state.tool_history
        ))

    @staticmethod
    def _safe_final_after_invalid_live_call(
            state: ResearchState, decision: PlannerDecision) -> PlannerDecision:
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            decision=Decision.final_answer,
            thinking_text=(
                "L'appel d'outil proposé reste invalide après correction, mais "
                "des sources officielles ont déjà été récupérées. Je réponds "
                "prudemment à partir de ces seules sources."),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="appel live invalide; preuves officielles conservées",
                next_action="final_answer"),
        )

    @staticmethod
    def _attach_semantic_legal_terms(decision: PlannerDecision) -> None:
        """Recopie la qualification du plan dans l'appel de recherche.

        ``legal_terms`` appartient au schéma du Planner et au schéma de
        l'outil. Les deux emplacements représentent la même information :
        le Planner peut la fournir au niveau de sa décision, alors que
        l'exécuteur ne lit que les arguments de l'outil. Cette normalisation
        ne crée aucune qualification; en son absence, la validation du
        contrat demande au modèle de corriger sa décision.
        """
        if decision.next_tool not in {
                "semantic_search_ccq", "semantic_search_cpc"}:
            return
        terms = (decision.legal_terms or "").strip()
        if not terms:
            return
        arguments = dict(decision.arguments or {})
        if not str(arguments.get("legal_terms") or "").strip():
            arguments["legal_terms"] = terms
            decision.arguments = arguments

    def _semantic_candidates_for(self, state: ResearchState,
                                 fetch_tool: str) -> list[int | float]:
        """Candidats dans l'ordre réel du dernier classement sémantique."""
        search_tool = {
            "get_ccq_articles": "semantic_search_ccq",
            "get_cpc_articles": "semantic_search_cpc",
        }.get(fetch_tool)
        if not search_tool:
            return []
        latest_search = next(
            (observation for observation in reversed(state.tool_history)
             if observation.tool_name == search_tool and observation.ok),
            None,
        )
        if latest_search is None:
            return []
        numbers: list[int | float] = []
        seen: set[float] = set()
        for raw in ARTICLE_LABEL_RE.findall(
                latest_search.normalized_response or ""):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value in seen:
                continue
            seen.add(value)
            numbers.append(int(value) if value.is_integer() else value)
        return numbers

    def _article_fetch_arguments(
            self, state: ResearchState, fetch_tool: str,
            *, batch_size: Optional[int] = None) -> Optional[dict]:
        """Prochain lot non lu, sans changer l'ordre de classement RAG."""
        if not self.chat_mode:
            return self._legacy_article_fetch_arguments(state, fetch_tool)
        candidates = self._semantic_candidates_for(state, fetch_tool)
        if self.evidence_first_enabled:
            candidates = candidates[:self.evidence_first_initial_candidate_count]
        if not candidates:
            # Une demande précise peut légitimement mentionner un article
            # avant toute recherche. Elle ne reçoit aucune complétion de la
            # mémoire : seuls les numéros écrits dans la conversation servent.
            query = "\n".join(message.content for message in state.messages
                              if message.role.value == "user")
            candidates = [
                int(float(number)) if float(number).is_integer()
                else float(number)
                for number in ARTICLE_RE.findall(query)
            ]
        if not candidates:
            return None
        fetched = {
            str(number)
            for observation in state.tool_history
            if observation.tool_name == fetch_tool and observation.ok
            for number in numeros_demandes(
                observation.tool_name, observation.arguments)
        }
        remaining = [
            value for value in candidates if str(value) not in fetched
        ]
        remaining_capacity = self.max_articles_per_issue - len(fetched)
        if remaining_capacity <= 0 or not remaining:
            return None
        batch_index = sum(
            observation.tool_name == fetch_tool and observation.ok
            for observation in state.tool_history)
        if self.evidence_first_enabled:
            budget = article_budget(
                self, fetched_count=len(fetched), batch_index=batch_index,
                remaining_candidates=bool(remaining))
            if not budget["allow_next"] and fetched:
                return None
            width = budget["fetch_count"]
        else:
            width = (batch_size if batch_size is not None else
                     (self.initial_article_fetch_k if not fetched
                      else self.article_fetch_batch_size))
        width = min(max(1, width), remaining_capacity,
                    MAX_ARTICLES_PAR_APPEL)
        return {"articles": remaining[:width]}

    @staticmethod
    def _legacy_article_fetch_arguments(
            state: ResearchState, fetch_tool: str) -> Optional[dict]:
        """Compatibilité des trajectoires dataset historiques (plage compacte)."""
        search_tool = ("semantic_search_ccq" if fetch_tool ==
                       "get_ccq_articles" else "semantic_search_cpc")
        search_text = "\n".join(
            observation.normalized_response for observation in state.tool_history
            if observation.ok and observation.tool_name == search_tool)
        candidates = ARTICLE_LABEL_RE.findall(search_text)
        if not candidates:
            candidates = ARTICLE_RE.findall("\n".join(
                message.content for message in state.messages
                if message.role.value == "user"))
        already_fetched = {
            str(number) for observation in state.tool_history
            if observation.tool_name == fetch_tool and observation.ok
            for number in numeros_demandes(
                observation.tool_name, observation.arguments)
        }
        values = []
        for candidate in candidates:
            if candidate in already_fetched:
                continue
            try:
                value = float(candidate)
            except ValueError:
                continue
            if value not in values:
                values.append(value)
            if len(values) == 3:
                break
        if not values:
            return None
        primary = values[0]
        nearby = [value for value in values if abs(value - primary) <= 5]
        start, end = min(nearby), max(nearby)
        def as_json(value: float) -> int | float:
            return int(value) if value.is_integer() else value
        arguments = {"start_article": as_json(start)}
        if end != start:
            arguments["end_article"] = as_json(end)
        return arguments

    def _attach_retrieved_candidates(self, state: ResearchState,
                                     decision: PlannerDecision) -> None:
        """Transforme des candidats sémantiques en textes à vérifier.

        Une recherche sémantique ne prouve rien; son résultat est un ensemble
        de candidats. Lorsque le tour suivant demande leurs textes officiels,
        conserver uniquement le premier rang ferait perdre les autres pistes
        avant toute lecture de la loi. Cette règle transporte donc les numéros
        réellement renvoyés par la dernière recherche vers ``get_*_articles``
        sans décider de leur pertinence juridique.
        """
        if not self.chat_mode or decision.next_tool not in {
                "get_ccq_articles", "get_cpc_articles"}:
            return
        arguments = self._article_fetch_arguments(state, decision.next_tool)
        if arguments:
            decision.arguments = arguments

    def _guard_clarification(self, state: ResearchState,
                             decision: PlannerDecision) -> PlannerDecision:
        """Force clarification when required but not yet asked."""
        if not state.scenario.expected_route.requires_clarification:
            if decision.decision == Decision.ask_clarification:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.final_answer,
                    thinking_text=(
                        "Cette catégorie ne nécessite pas de clarification. "
                        "Je réponds directement avec les informations disponibles."
                    ),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="pas de clarification requise",
                        next_action="final_answer"),
                )
            return decision
        if self._clarification_answered(state):
            return decision
        if decision.decision == Decision.ask_clarification:
            return decision
        question = (decision.clarification_question
                    or "Pouvez-vous préciser les faits essentiels "
                       "(lieu, dates, montants) afin que je puisse "
                       "identifier la règle applicable?")
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            missing_critical_facts=state.scenario.facts_missing or ["faits essentiels"],
            decision=Decision.ask_clarification,
            clarification_question=question,
            thinking_text=(
                "Des informations essentielles manquent pour identifier la règle "
                "applicable. Je dois poser une question de clarification avant "
                "de lancer une recherche."
            ),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="clarification avant recherche",
                next_action="ask_clarification"),
        )

    def _guard_failed_tool(self, state: ResearchState,
                           decision: PlannerDecision) -> PlannerDecision:
        """Skip a tool that has already failed 2+ times."""
        if decision.decision != Decision.call_tool or not decision.next_tool:
            return decision
        fail_count = sum(1 for o in state.tool_history
                         if o.tool_name == decision.next_tool and not o.ok)
        if fail_count < 2:
            return decision
        route = self._effective_route(state)
        for candidate in route:
            if candidate == decision.next_tool:
                continue
            if any(o.tool_name == candidate and o.ok for o in state.tool_history):
                continue
            args = self._arguments(candidate, state)
            if args is not None:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.call_tool,
                    next_tool=candidate,
                    arguments=args,
                    thinking_text=(
                        f"L'outil {decision.next_tool} a échoué plusieurs fois. "
                        f"Je passe à {candidate}."
                    ),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="outil défaillant, redirection",
                        next_action=f"call_tool:{candidate}"),
                )
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            decision=Decision.final_answer,
            thinking_text=(
                f"L'outil {decision.next_tool} a échoué plusieurs fois et "
                "aucun outil alternatif n'est disponible."
            ),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="aucun outil disponible",
                next_action="final_answer"),
        )

    def _guard_federal_fetch(self, state: ResearchState,
                             decision: PlannerDecision) -> PlannerDecision:
        """Redirect to fetch_document when search already returned results."""
        if decision.decision != Decision.call_tool:
            return decision
        if decision.next_tool != "search_legal_documents":
            return decision
        jurisdiction = getattr(state.scenario, "jurisdiction_status", "")
        is_federal = (jurisdiction == "supported_federal"
                      or state.scenario.request_type == "comparative_law")
        if not is_federal:
            return decision
        search_calls = [
            o for o in state.tool_history
            if o.tool_name == "search_legal_documents" and o.ok
            and o.normalized_response.strip() not in ("", "[]", "{}")
        ]
        if not search_calls:
            return decision
        already_fetched = any(
            o.tool_name == "fetch_document" for o in state.tool_history
        )
        if already_fetched:
            return PlannerDecision(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                decision=Decision.final_answer,
                thinking_text=(
                    "J'ai déjà effectué une recherche et récupéré le document "
                    "fédéral. J'ai assez d'information pour répondre."
                ),
                decision_trace=DecisionTrace(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    need="sources fédérales récupérées",
                    next_action="final_answer"),
            )
        args = self._arguments("fetch_document", state)
        if not args:
            return PlannerDecision(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                decision=Decision.final_answer,
                thinking_text=(
                    "La recherche fédérale a retourné des résultats mais je ne "
                    "parviens pas à identifier une citation précise pour "
                    "récupérer le document complet. Je réponds avec les "
                    "informations disponibles dans les résultats de recherche."
                ),
                decision_trace=DecisionTrace(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    need="réponse basée sur résultats de recherche",
                    next_action="final_answer"),
            )
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            decision=Decision.call_tool,
            next_tool="fetch_document",
            arguments=args,
            thinking_text=(
                "La recherche a déjà retourné des résultats. "
                "Je passe à fetch_document pour récupérer le texte officiel "
                "au lieu de relancer une recherche."
            ),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="texte officiel du document fédéral",
                next_action="call_tool:fetch_document"),
        )

    def _guard_required_tools(self, state: ResearchState,
                              decision: PlannerDecision) -> PlannerDecision:
        """Prevent final_answer when required tools haven't been called."""
        if decision.decision not in {Decision.final_answer, Decision.cannot_conclude}:
            return decision
        required = state.scenario.expected_route.required_tools()
        called = {o.tool_name for o in state.tool_history}
        missing = [t for t in required if t not in called]
        if not missing:
            return decision
        for tool in missing:
            args = self._arguments(tool, state)
            if args is not None:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.call_tool,
                    next_tool=tool,
                    arguments=args,
                    thinking_text=(
                        f"Je dois encore appeler {tool} avant de répondre, "
                        "car c'est un outil requis pour cette catégorie."
                    ),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="outil requis non encore appelé",
                        next_action=f"call_tool:{tool}"),
                )
        return decision

    @staticmethod
    def _chat_jurisdiction_hint(state: ResearchState) -> Optional[str]:
        """Juridiction déduite DÉTERMINISTIQUEMENT de la conversation.

        Délègue à ``lexior.services.jurisdiction.detect_jurisdiction_hint``
        (implémentation canonique, partagée avec le graphe central).
        """
        return detect_jurisdiction_hint(state.messages)

    def _guard_chat_jurisdiction(self, state: ResearchState,
                                 decision: PlannerDecision,
                                 retried: bool) -> PlannerDecision:
        """Bloque les outils québécois quand l'utilisateur n'est pas au Québec.

        La juridiction déduite écrase aussi decision.jurisdiction pour que
        le rédacteur reçoive une juridiction_etablie fiable, quel que soit
        le modèle qui planifie.
        """
        hint = self._chat_jurisdiction_hint(state)
        if hint is None:
            return decision
        decision.jurisdiction = hint
        if hint == "Québec":
            return decision
        if (decision.decision != Decision.call_tool
                or decision.next_tool not in QC_ONLY_TOOLS):
            return decision
        if not retried:
            raise QuebecToolsBlocked(
                f"outil québécois {decision.next_tool} interdit : "
                f"l'utilisateur n'est pas au Québec ({hint}); utilise "
                "search_legal_documents ou fetch_document (droit fédéral "
                "ou autre province), ou passe à final_answer")
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=hint,
            decision=Decision.final_answer,
            thinking_text=(
                f"L'utilisateur n'est pas au Québec ({hint}) : les outils "
                "CCQ/CPC ne s'appliquent pas. Je réponds avec le droit "
                "fédéral applicable et les orientations générales."
            ),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=hint,
                need="outils québécois inapplicables hors Québec",
                next_action="final_answer"),
        )

    def _guard_progressive_article_review(
            self, state: ResearchState,
            decision: PlannerDecision) -> PlannerDecision:
        """Élargit la lecture si le premier lot ne soutient pas l'analyse.

        Cette garde ne choisit aucun article par son numéro : elle avance dans
        l'ordre produit par le RAG et s'arrête dès qu'une règle est revue comme
        applicable. Une règle seulement conditionnelle justifie un lot
        supplémentaire, car elle peut révéler une disposition plus directe.
        """
        if decision.decision not in {
                Decision.final_answer, Decision.cannot_conclude}:
            return decision
        if state.scenario.request_type != "case_analysis":
            return decision
        if self.evidence_first_enabled:
            fetched_batches = sum(
                observation.tool_name in {"get_ccq_articles", "get_cpc_articles"}
                and observation.ok for observation in state.tool_history)
            source_bounded = any(
                review.get("extraction_mode") == "source_bounded"
                and review.get("status") in {"applicable", "conditionally_applicable"}
                for review in state.article_reviews.values())
            candidates = bool(
                self._semantic_candidates_for(state, "get_ccq_articles")
                or self._semantic_candidates_for(state, "get_cpc_articles"))
            if source_bounded or fetched_batches >= self.evidence_first_maximum_article_batches:
                return self._guard_live_source_completeness(state, decision)
            fetch_tool = next((tool for tool in (
                "get_ccq_articles", "get_cpc_articles")
                if self._semantic_candidates_for(state, tool)), None)
            if candidates and fetch_tool:
                arguments = self._article_fetch_arguments(state, fetch_tool)
                if arguments:
                    return PlannerDecision(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        decision=Decision.call_tool,
                        next_tool=fetch_tool,
                        arguments=arguments,
                        thinking_text=(
                            "Les propositions du premier lot ne permettent pas "
                            "encore de retenir une autorité suffisante; un seul "
                            "lot evidence-first supplémentaire est autorisé."),
                        decision_trace=DecisionTrace(
                            request_type=decision.request_type,
                            jurisdiction=decision.jurisdiction,
                            need="lacune de source explicite",
                            next_action=f"call_tool:{fetch_tool}"),
                    )
            return self._guard_live_source_completeness(state, decision)
        sufficiency = assess_legislative_sufficiency(
            state.article_reviews, state.case_description, state.case_facts,
            remaining_candidates=bool(
                self._semantic_candidates_for(state, "get_ccq_articles")
                or self._semantic_candidates_for(state, "get_cpc_articles")))
        if sufficiency.sufficient:
            return self._guard_live_source_completeness(state, decision)
        fetch_tool = next((tool for tool in (
            "get_ccq_articles", "get_cpc_articles")
            if self._semantic_candidates_for(state, tool)), None)
        if fetch_tool:
            arguments = self._article_fetch_arguments(
                state, fetch_tool,
                batch_size=self.article_fetch_batch_size)
            if arguments:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.call_tool,
                    next_tool=fetch_tool,
                    arguments=arguments,
                    thinking_text=(
                        "Le premier lot de textes ne contient aucune règle "
                        "revue comme directement applicable. Je lis le lot "
                        "suivant du même classement avant de conclure."),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="élargir la couverture des articles candidats",
                        next_action=f"call_tool:{fetch_tool}"),
                )
        return self._guard_live_source_completeness(state, decision)

    def _guard_live_tool_chain_compatibility(
            self, state: ResearchState,
            decision: PlannerDecision) -> PlannerDecision:
        """Applique la provenance du candidat également en mode live."""
        if decision.decision != Decision.call_tool or not decision.next_tool:
            return decision
        has_qc_case_search = any(
            obs.tool_name == "search_quebec_jurisprudence" and obs.ok
            for obs in state.tool_history)
        has_qc_regulation_search = any(
            obs.tool_name == "search_quebec_regulations" and obs.ok
            for obs in state.tool_history)
        if decision.next_tool == "fetch_document" and has_qc_regulation_search:
            args = self._arguments("get_quebec_regulation", state)
            if args:
                decision.next_tool = "get_quebec_regulation"
                decision.arguments = args
                return decision
            return PlannerDecision(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                decision=Decision.final_answer,
                thinking_text=(
                    "La recherche réglementaire québécoise ne fournit pas "
                    "d'URL acceptée pour un texte complet."),
                decision_trace=DecisionTrace(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    need="URL réglementaire acceptée absente",
                    next_action="final_answer"))
        if decision.next_tool == "fetch_document" and has_qc_case_search:
            accepted = [item for item in state.usable_case_sources
                        if (item.get("source_url", "") if isinstance(item, dict)
                            else getattr(item, "source_url", ""))]
            if not accepted:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.final_answer,
                    thinking_text=(
                        "Aucune décision québécoise n'a été retenue par le "
                        "gate; je ne récupère pas une URL du résultat rejeté."),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="candidat jurisprudentiel non accepté",
                        next_action="final_answer"))
            return PlannerDecision(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                decision=Decision.call_tool,
                next_tool="get_quebec_regulation",
                arguments={"url": str(
                    accepted[0].get("source_url", "")
                    if isinstance(accepted[0], dict)
                    else accepted[0].source_url)},
                thinking_text=(
                    "Le candidat québécois accepté doit être récupéré par "
                    "get_quebec_regulation avant toute utilisation."),
                decision_trace=DecisionTrace(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    need="chaîne jurisprudentielle québécoise",
                    next_action="call_tool:get_quebec_regulation"))
        if decision.next_tool == "get_quebec_regulation":
            args = self._arguments("get_quebec_regulation", state)
            if not args:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.final_answer,
                    thinking_text=(
                        "Aucun document québécois accepté ne possède d'URL "
                        "récupérable; je conserve les textes déjà retenus."),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="URL de source acceptée absente",
                        next_action="final_answer"))
            decision.arguments = args
        return decision

    def _guard_live_source_completeness(
            self, state: ResearchState,
            decision: PlannerDecision) -> PlannerDecision:
        """Evidence-first does not add case law without an explicit gap."""
        if (self.evidence_first_enabled
                and state.scenario.request_type != "case_law_research"):
            return decision
        if decision.decision not in {
                Decision.final_answer, Decision.cannot_conclude}:
            return decision
        if state.scenario.request_type != "case_analysis":
            return decision
        has_case_search = any(
            obs.tool_name == "search_quebec_jurisprudence"
            for obs in state.tool_history)
        has_case_content = state.case_law_search_status == "verified" or any(
            obs.tool_name == "get_quebec_regulation" and obs.ok
            for obs in state.tool_history)
        if has_case_content:
            return decision
        statuses = {review.get("status") for review in
                    state.article_reviews.values()}
        if not statuses & {"applicable", "conditionally_applicable"}:
            return decision
        if (state.case_law_search_status in {
                "candidates_pending_fetch", "candidate_pending_fetch"}
                and state.usable_case_sources):
            arguments = self._arguments("get_quebec_regulation", state)
            if arguments:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.call_tool,
                    next_tool="get_quebec_regulation",
                    arguments=arguments,
                    thinking_text=(
                        "La recherche a identifié une décision candidate. Je "
                        "récupère son contenu intégral avant de la citer ou de "
                        "l'utiliser comme preuve."),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="contenu intégral d'une décision candidate",
                        next_action="call_tool:get_quebec_regulation"),
                )
            return decision
        # Compatibility for ResearchState objects created by integrations
        # predating the explicit case-law gate status. This branch is limited
        # to the default ``not_required`` state; a real gate result such as
        # ``irrelevant`` or ``candidates_without_url`` can never use a raw URL.
        if state.case_law_search_status == "not_required":
            legacy_search = next(
                (obs for obs in reversed(state.tool_history)
                 if obs.tool_name == "search_quebec_jurisprudence"
                 and obs.ok and obs.source_urls),
                None,
            )
            if legacy_search:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.call_tool,
                    next_tool="get_quebec_regulation",
                    arguments={"url": legacy_search.source_urls[0]},
                    thinking_text=(
                        "État legacy sans statut de gate : la source sera "
                        "vérifiée avant toute utilisation."),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="compatibilité d'état sans statut de gate",
                        next_action="call_tool:get_quebec_regulation"),
                )
        if (has_case_search and state.case_law_search_status in {
                "irrelevant", "empty", "failed", "tool_error",
                "coverage_gap", "candidates_without_url"}):
            if state.reformulation_count < state.max_search_reformulations:
                arguments = self._arguments("search_quebec_jurisprudence", state)
                if arguments:
                    return PlannerDecision(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        decision=Decision.call_tool,
                        next_tool="search_quebec_jurisprudence",
                        arguments=arguments,
                        thinking_text=(
                            "Le gate n'a pas retenu de décision. Je tente une "
                            "seule reformulation avec le dossier complet."),
                        decision_trace=DecisionTrace(
                            request_type=decision.request_type,
                            jurisdiction=decision.jurisdiction,
                            need="reformulation jurisprudentielle bornée",
                            next_action="call_tool:search_quebec_jurisprudence"))
            return decision
        factual_answers = [entry for entry in state.clarification_history
                           if entry.get("category") == "fact"
                           and entry.get("answered")]
        if not factual_answers and "applicable" not in statuses:
            return decision
        arguments = self._arguments("search_quebec_jurisprudence", state)
        if not arguments:
            return decision
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            decision=Decision.call_tool,
            next_tool="search_quebec_jurisprudence",
            arguments=arguments,
            thinking_text=(
                "Une règle officielle a été revue et les faits disponibles "
                "permettent une recherche ciblée de jurisprudence québécoise."),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="jurisprudence ciblée par disposition et faits",
                next_action="call_tool:search_quebec_jurisprudence"),
        )

    def _guard_budget(self, state: ResearchState,
                      decision: PlannerDecision) -> PlannerDecision:
        """Budget d'outils épuisé : forcer la synthèse au lieu d'un appel."""
        if decision.decision != Decision.call_tool:
            return decision
        if state.tool_calls_made() < state.max_tool_calls:
            return decision
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            missing_critical_facts=decision.missing_critical_facts,
            required_sources=decision.required_sources,
            decision=Decision.final_answer,
            thinking_text=(
                "Le budget d'appels d'outils est épuisé; je synthétise "
                "à partir des résultats déjà obtenus."
            ),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="budget d'outils épuisé",
                next_action="final_answer"),
        )

    def _guard_duplicate_call(self, state: ResearchState,
                             decision: PlannerDecision) -> PlannerDecision:
        """Prevent calling the same tool with identical arguments."""
        if decision.decision != Decision.call_tool or not decision.next_tool:
            return decision
        for obs in state.tool_history:
            if obs.tool_name == decision.next_tool and obs.arguments == decision.arguments:
                route = self._effective_route(state)
                for candidate in route:
                    if candidate == decision.next_tool:
                        continue
                    if any(o.tool_name == candidate and o.ok for o in state.tool_history):
                        continue
                    fail_count = sum(1 for o in state.tool_history
                                     if o.tool_name == candidate and not o.ok)
                    if fail_count >= 2:
                        continue
                    args = self._arguments(candidate, state)
                    if args is not None:
                        return PlannerDecision(
                            request_type=decision.request_type,
                            jurisdiction=decision.jurisdiction,
                            decision=Decision.call_tool,
                            next_tool=candidate,
                            arguments=args,
                            thinking_text=(
                                f"L'outil {decision.next_tool} a déjà été appelé "
                                "avec ces mêmes arguments. Je passe à l'outil "
                                f"suivant dans la route : {candidate}."
                            ),
                            decision_trace=DecisionTrace(
                                request_type=decision.request_type,
                                jurisdiction=decision.jurisdiction,
                                need="éviter appel identique",
                                next_action=f"call_tool:{candidate}"),
                        )
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.final_answer,
                    thinking_text=(
                        f"L'outil {decision.next_tool} a déjà été appelé avec "
                        "les mêmes arguments. Je réponds avec les informations "
                        "déjà récupérées."
                    ),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="appel identique détecté",
                        next_action="final_answer"),
                )
        return decision

    def _guard_tool_compatibility(self, state: ResearchState,
                                  decision: PlannerDecision) -> PlannerDecision:
        """Redirect to a compatible tool when the teacher picks a forbidden one."""
        if decision.decision != Decision.call_tool or not decision.next_tool:
            return decision
        if state.scenario.expected_route.no_tool:
            return PlannerDecision(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                decision=Decision.final_answer,
                thinking_text=(
                    decision.thinking_text or
                    "Aucun outil n'est attendu pour cette catégorie. "
                    "Je réponds directement."
                ),
                decision_trace=DecisionTrace(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    need="pas d'outil attendu",
                    next_action="final_answer"),
            )
        from .taxonomy import REQUEST_TYPES
        rt = REQUEST_TYPES.get(state.scenario.request_type)
        policy = rt.route_policy if rt else None
        # La garde de juridiction s'applique AUSSI au choix libre du planner,
        # pas seulement au chemin de repli : en live, policy.allows_tool()
        # laissait passer fetch_document — déclaré fédéral dans tool_coverage —
        # avec la citation « CCQ 1466 » sur un scénario québécois. L'appel
        # revenait « document fédéral vide », après avoir consommé un tour.
        juridiction = (getattr(state, "jurisdiction_status", "")
                       or getattr(state.scenario, "jurisdiction_status", ""))
        if not juridiction_compatible(decision.next_tool, juridiction):
            pass                      # incompatible : repli sur la route
        elif not self._numero_a_une_provenance(decision, state):
            pass                      # article sorti de mémoire : repli
        elif policy and policy.required_capabilities:
            if policy.allows_tool(decision.next_tool):
                return decision
        else:
            allowed = set(state.scenario.expected_route.allowed_tools())
            if decision.next_tool in allowed:
                return decision
        route = self._effective_route(state)
        for candidate in route:
            if any(o.tool_name == candidate and o.ok for o in state.tool_history):
                continue
            fail_count = sum(1 for o in state.tool_history
                             if o.tool_name == candidate and not o.ok)
            if fail_count >= 2:
                continue
            args = self._arguments(candidate, state)
            if args is not None:
                return PlannerDecision(
                    request_type=decision.request_type,
                    jurisdiction=decision.jurisdiction,
                    decision=Decision.call_tool,
                    next_tool=candidate,
                    arguments=args,
                    thinking_text=decision.thinking_text or (
                        f"Redirection vers {candidate} car l'outil "
                        f"{decision.next_tool} n'est pas dans la route."
                    ),
                    decision_trace=DecisionTrace(
                        request_type=decision.request_type,
                        jurisdiction=decision.jurisdiction,
                        need="redirection vers outil compatible",
                        next_action=f"call_tool:{candidate}"),
                )
        return PlannerDecision(
            request_type=decision.request_type,
            jurisdiction=decision.jurisdiction,
            decision=Decision.final_answer,
            thinking_text=(
                decision.thinking_text or
                "Aucun outil compatible disponible dans la route. "
                "Je réponds avec les informations déjà récupérées."
            ),
            decision_trace=DecisionTrace(
                request_type=decision.request_type,
                jurisdiction=decision.jurisdiction,
                need="aucun outil compatible restant",
                next_action="final_answer"),
        )

    def _validate_arguments(self, state: ResearchState,
                            decision: PlannerDecision) -> PlannerDecision:
        """Valide et corrige les arguments du Teacher sans changer l'outil choisi.

        Le Teacher décide genuinement de l'outil (pas de route guard). Mais
        ses arguments peuvent être invalides (numéros d'articles halluccinés,
        mauvais types). On reconstruit les arguments de façon déterministe
        quand c'est possible, sinon on garde ceux du Teacher.
        """
        if not decision.next_tool:
            return decision
        reconstructed = self._arguments(decision.next_tool, state,
                                        legal_terms=decision.legal_terms,
                                         thinking=decision.thinking_text)
        if reconstructed is not None:
            decision.arguments = reconstructed
        return decision

    @staticmethod
    def _clarification_answered(state: ResearchState) -> bool:
        scenario_answer = state.scenario.clarification_answer
        if scenario_answer and any(
            message.role.value == "user" and message.content == scenario_answer
            for message in state.messages
        ):
            return True
        return any(
            state.messages[index].role.value == "assistant" and
            state.messages[index].content.rstrip().endswith("?") and
            state.messages[index + 1].role.value == "user"
            for index in range(len(state.messages) - 1)
        )

    def _teacher_decide(self, state: ResearchState,
                        feedback: str = "") -> PlannerDecision:
        if self.client is None:
            raise RuntimeError("client Teacher requis hors mode offline")
        visible = {
            "user_query": state.scenario.user_query,
            "work_location": state.work_location,
            "jurisdiction_status": state.jurisdiction_status,
            "legal_regime": state.legal_regime,
            "employment_sector": state.employment_sector,
            "request_intent": state.request_intent,
            "request_type": state.scenario.request_type,
            "legal_domain": state.legal_domain,
            "jurisdiction_material": state.jurisdiction_material,
            "employment_regime_material": state.employment_regime_material,
            "classification_confidence": state.classification_confidence,
            "messages": [m.model_dump(mode="json") for m in state.messages],
            "clarification_already_answered": any(
                state.messages[index].role.value == "assistant" and
                state.messages[index].content.rstrip().endswith("?") and
                state.messages[index + 1].role.value == "user"
                for index in range(len(state.messages) - 1)
            ),
            "tool_history": [{
                "tool_name": o.tool_name,
                "arguments": o.arguments,
                "normalized_response": o.normalized_response,
                "source_urls": o.source_urls,
                "citations": o.citations,
                "truncated": o.truncated,
                "error": o.error,
            } for o in state.tool_history],
            "remaining_tool_calls": state.max_tool_calls - state.tool_calls_made(),
        }
        system_content = planner_system_prompt(self.catalog)
        if self.chat_mode:
            system_content += CHAT_PLANNER_SUPPLEMENT
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
        ]
        if feedback:
            messages.append({
                "role": "user",
                "content": (
                    f"Ta décision précédente était invalide : {feedback}. "
                    "Renvoie un JSON corrigé qui respecte strictement le "
                    "schéma demandé (next_tool obligatoire pour call_tool, "
                    "aucune valeur null)."
                ),
            })
        raw = self.client.complete_json("planner", messages, temperature=0.0)
        # Les null renvoyés par le Teacher retombent sur les défauts du schéma.
        raw = {k: v for k, v in raw.items() if v is not None}
        raw_decision = raw.get("decision")
        if isinstance(raw_decision, str) and raw_decision in self.catalog.tools:
            raw.setdefault("next_tool", raw_decision)
            raw["decision"] = Decision.call_tool.value
        elif not raw_decision and raw.get("next_tool") in self.catalog.tools:
            raw["decision"] = Decision.call_tool.value
        thinking = raw.pop("thinking_text", "")
        if not thinking:
            thinking = raw.pop("thinking", "")
        decision = PlannerDecision.model_validate(raw)
        decision.thinking_text = thinking
        return decision

    def _offline_decide(self, state: ResearchState) -> PlannerDecision:
        scenario = state.scenario
        request_type = scenario.request_type
        jurisdiction = self._infer_jurisdiction(scenario.user_query, request_type)
        clarified = any(m.role.value == "user" and m.content == scenario.clarification_answer
                        for m in state.messages if scenario.clarification_answer)
        needs_clarification = (
            scenario.expected_route.requires_clarification
            or scenario.clarification_stage in ("before_search", "after_initial_research")
        )
        if needs_clarification and not clarified:
            return PlannerDecision(
                request_type=request_type, jurisdiction=jurisdiction,
                missing_critical_facts=scenario.facts_missing or ["faits essentiels"],
                decision=Decision.ask_clarification,
                clarification_question="Pouvez-vous préciser la juridiction, le lieu et les faits essentiels concernés?",
                thinking_text=(
                    f"L'utilisateur pose une question de type {request_type} mais des "
                    f"informations essentielles manquent : {', '.join(scenario.facts_missing or ['faits essentiels'])}. "
                    "Avant de lancer une recherche, je dois clarifier ces éléments car ils "
                    "pourraient changer la juridiction ou la règle applicable."
                ),
                decision_trace=DecisionTrace(request_type=request_type, jurisdiction=jurisdiction,
                                             need="clarification avant recherche",
                                             next_action="ask_clarification"))
        if scenario.expected_route.no_tool:
            return PlannerDecision(
                request_type=request_type, jurisdiction=jurisdiction,
                decision=Decision.final_answer,
                thinking_text=(
                    "Cette demande n'est pas de nature juridique. Aucun outil de "
                    "recherche n'est nécessaire, je peux répondre directement."
                ),
                decision_trace=DecisionTrace(request_type=request_type, jurisdiction=jurisdiction,
                                             need="aucune source juridique nécessaire",
                                             next_action="final_answer"))
        if state.tool_calls_made() >= state.max_tool_calls:
            return PlannerDecision(
                request_type=request_type, jurisdiction=jurisdiction,
                decision=Decision.cannot_conclude,
                thinking_text=(
                    "J'ai atteint la limite d'appels d'outils autorisés. Je dois "
                    "répondre avec les sources déjà récupérées, même si elles sont incomplètes."
                ),
                decision_trace=DecisionTrace(request_type=request_type, jurisdiction=jurisdiction,
                                             need="limite d'appels atteinte",
                                             next_action="cannot_conclude"))
        route = self._effective_route(state)
        if state.tool_calls_made() >= len(route):
            return PlannerDecision(
                request_type=request_type, jurisdiction=jurisdiction,
                decision=Decision.final_answer,
                thinking_text=(
                    "J'ai récupéré toutes les sources nécessaires. Les réponses "
                    "d'outils contiennent les informations juridiques pertinentes pour "
                    "répondre à la question de l'utilisateur."
                ),
                decision_trace=DecisionTrace(request_type=request_type, jurisdiction=jurisdiction,
                                             need="sources prévues récupérées",
                                             next_action="final_answer"))
        tool = route[state.tool_calls_made()]
        terms = self._offline_legal_terms(state)
        args = self._arguments(tool, state, legal_terms=terms)
        if args is None:
            return PlannerDecision(
                request_type=request_type, jurisdiction=jurisdiction,
                decision=Decision.cannot_conclude,
                thinking_text=(
                    f"Je devrais appeler {tool} mais je ne trouve pas d'identifiant "
                    "fiable dans les résultats précédents pour construire les arguments. "
                    "Je ne peux pas conclure de façon fiable."
                ),
                decision_trace=DecisionTrace(request_type=request_type, jurisdiction=jurisdiction,
                                             need="identifiant non présent dans les résultats MCP",
                                             next_action="cannot_conclude"))
        thinking = self._generate_offline_thinking(tool, args, state)
        return PlannerDecision(
            request_type=request_type, jurisdiction=jurisdiction,
            required_sources=getattr(scenario, "source_intent", []),
            decision=Decision.call_tool, next_tool=tool, arguments=args,
            thinking_text=thinking,
            decision_trace=DecisionTrace(request_type=request_type, jurisdiction=jurisdiction,
                                         need=f"source via {tool}", next_action=f"call_tool:{tool}"))

    @staticmethod
    def _offline_legal_terms(state: ResearchState) -> str:
        """Qualification disponible sans modèle dans un scénario synthétique.

        Le mode hors ligne sert à vérifier les routes et ne doit ni inventer
        de règle de droit, ni inscrire un numéro d'article dans la recherche.
        Il réutilise seulement le domaine et l'intention de source déclarés
        par le scénario, qui décrivent déjà le type de recherche attendu.
        """
        scenario = state.scenario
        parts = [
            str(getattr(scenario, "legal_domain", "") or "").strip(),
            *(str(value).strip() for value in
              (getattr(scenario, "source_intent", []) or []) if str(value).strip()),
        ]
        terms = ", ".join(dict.fromkeys(part for part in parts if part))
        prior_searches = sum(
            observation.tool_name in {"semantic_search_ccq", "semantic_search_cpc"}
            for observation in state.tool_history)
        if prior_searches:
            # Le marqueur distingue une nouvelle tentative pour le cache. Il
            # ne qualifie pas juridiquement les faits et ne modifie pas les
            # candidats d'un RAG réel : le mode offline ne fait ici que
            # rejouer les routes synthétiques sans appel de modèle.
            return f"{terms}, analyse complémentaire"
        return terms

    @staticmethod
    def _generate_offline_thinking(tool: str, args: dict, state: ResearchState) -> str:
        """Génère un thinking en langue naturelle pour le mode offline."""
        request_type = state.scenario.request_type
        step = state.tool_calls_made()

        if tool in {"get_ccq_articles", "get_cpc_articles"}:
            code = "Code civil du Québec" if "ccq" in tool else "Code de procédure civile"
            art = args.get("start_article", "")
            if step == 0:
                return (
                    f"L'utilisateur demande le texte officiel de l'article {art} du {code}. "
                    f"Je dois utiliser l'outil {tool} pour récupérer le texte officiel "
                    "directement, sans passer par une recherche sémantique puisque le "
                    "numéro d'article est déjà connu."
                )
            return (
                f"La recherche sémantique a identifié l'article {art} comme pertinent. "
                f"Je dois maintenant récupérer le texte officiel via {tool} pour fonder "
                "ma réponse sur la source législative authentique."
            )
        if tool in {"semantic_search_ccq", "semantic_search_cpc"}:
            code = "CCQ" if "ccq" in tool else "CPC"
            if step > 0:
                return (
                    f"La première recherche n'a pas retourné de résultats pertinents. "
                    f"Je reformule la recherche dans le {code} pour identifier les "
                    "articles applicables à cette situation."
                )
            return (
                f"L'utilisateur pose une question sur le {code} sans mentionner de "
                "numéro d'article précis. Je lance une recherche sémantique pour "
                "identifier les articles les plus pertinents avant de récupérer "
                "leur texte officiel."
            )
        if tool == "search_legal_documents":
            return (
                "La question relève du droit fédéral canadien. Je lance une recherche "
                "dans la base A2AJ/CanLII pour identifier les lois ou décisions "
                "fédérales pertinentes."
            )
        if tool == "fetch_document":
            return (
                "La recherche a identifié un document pertinent. Je récupère son "
                "contenu complet pour pouvoir fonder ma réponse sur le texte officiel."
            )
        if tool == "search_quebec_jurisprudence":
            article_nums = PlannerAgent._extract_article_nums_from_history(state)
            if article_nums:
                arts = ", ".join(f"article {n}" for n in article_nums[:2])
                return (
                    f"J'ai identifié la règle applicable ({arts}). Je cherche "
                    "maintenant comment les tribunaux ont appliqué cette disposition "
                    "à des situations factuelles similaires, pour déterminer les "
                    "exceptions et conditions d'application concrètes."
                )
            return (
                "Les faits de l'utilisateur justifient une recherche de jurisprudence "
                "québécoise pour voir comment les tribunaux ont appliqué les règles "
                "dans des situations similaires."
            )
        return (
            f"Pour répondre à cette question de type {request_type}, j'utilise "
            f"l'outil {tool} afin de récupérer les sources juridiques nécessaires."
        )

    def _numero_a_une_provenance(
        self, decision: PlannerDecision, state: ResearchState,
    ) -> bool:
        """Le numéro demandé a-t-il été produit par une recherche ou l'usager ?

        Un numéro sorti de la mémoire du modèle fait retomber sur la route :
        le planner devra chercher avant de récupérer. Voir services/provenance.
        """
        numero = numero_demande(decision.next_tool, decision.arguments)
        if numero is None:
            return True
        return a_une_provenance(
            numero,
            getattr(state.scenario, "user_query", "") or "",
            reponses_reussies(state.tool_history))

    def _effective_route(self, state: ResearchState) -> list[str]:
        request_type = state.scenario.request_type
        steps = state.scenario.expected_route.steps
        jurisdiction = getattr(state.scenario, "jurisdiction_status", "")
        # Les conditions déclarées sur les étapes facultatives sont désormais
        # LUES (taxonomy_conditions), au lieu d'être réencodées ici en trois
        # paires codées en dur qui en couvraient trois sur onze.
        contexte = GardeContexte(
            request_type=request_type,
            jurisdiction_status=jurisdiction,
            user_query=getattr(state.scenario, "user_query", "") or "",
            tool_history=tuple(state.tool_history),
        )
        route: list[str] = []
        for step in steps:
            if not step.optional:
                route.append(step.tool)
                continue
            if etape_facultative_retenue(step.tool, step.condition, contexte):
                route.append(step.tool)
        for search_tool in ("semantic_search_ccq", "semantic_search_cpc"):
            if search_tool not in route or route.count(search_tool) > 1:
                continue
            searches = [
                o for o in state.tool_history
                if o.tool_name == search_tool
            ]
            if searches and (not searches[0].ok or self._no_result(
                    searches[0].normalized_response)):
                route.insert(route.index(search_tool) + 1, search_tool)
        _SEARCH_TO_FETCH = {
            "semantic_search_ccq": "get_ccq_articles",
            "semantic_search_cpc": "get_cpc_articles",
        }
        for search_tool, fetch_tool in _SEARCH_TO_FETCH.items():
            if fetch_tool not in route:
                continue
            searches = [o for o in state.tool_history if o.tool_name == search_tool]
            if len(searches) >= 2 and all(
                self._no_result(s.normalized_response) for s in searches
            ):
                route = [t for t in route if t != fetch_tool]
        return route

    def _build_quebec_case_law_query(self, state: ResearchState) -> str:
        """Construit une requête stable à partir du dossier complet."""
        reviews = [
            (number, review) for number, review in state.article_reviews.items()
            if review.get("retrieval_group") == "primary"
            and review.get("status") in {"applicable", "conditionally_applicable"}
        ]
        reviews.sort(key=lambda item: (
            int(item[1].get("rerank_rank", 10_000)), str(item[0])))
        role_terms: list[str] = []
        for _number, review in reviews:
            role_terms.extend(str(role).replace("_", " ")
                              for role in review.get("rule_roles", []))
        fact_terms = format_facts_for_query(state.case_facts or {})
        article_segment = "articles : " + " ".join(
            f"article {number}" for number, _review in reviews[:6])
        operation_segment = "operations : " + " ".join(
            dict.fromkeys(role_terms))
        segments = [
            article_segment,
            operation_segment,
            "evenement : " + state.case_description.strip(),
            "dommages : " + (state.case_description.strip() or "dommages decrits"),
            "faits confirmes : " + " | ".join(fact_terms),
            "juridiction : Quebec",
        ]
        if state.reformulation_count:
            segments.append("operation jurisprudentielle comparable")
        query = " ".join(segment for segment in segments if segment.strip())
        if len(query) <= 500:
            return query
        # Preserve articles and operations; drop lower-priority tail segments
        # instead of cutting a sentence at an arbitrary character.
        preserved = [article_segment, operation_segment]
        for segment in segments[2:]:
            candidate = " ".join([*preserved, segment])
            if len(candidate) <= 500:
                preserved.append(segment)
        return " ".join(preserved)

    def _arguments(self, tool: str, state: ResearchState,
                   thinking: str = "",
                   legal_terms: str = "") -> Optional[dict]:
        query = "\n".join(
            message.content for message in state.messages
            if message.role.value == "user"
        ) or state.scenario.user_query
        if tool in {"get_ccq_articles", "get_cpc_articles"}:
            return self._article_fetch_arguments(state, tool)
        if tool in {"semantic_search_ccq", "semantic_search_cpc"}:
            # La question part telle quelle, et sa traduction en vocabulaire
            # du Code part À CÔTÉ — jamais à sa place. Les deux recherches
            # sont réunies dans legal_rag.search().
            #
            # Ce qui occupait cette place avant : une phrase CONSTANTE
            # (« identifier les règles, recours, conditions et exceptions
            # juridiquement équivalents »), identique pour toutes les
            # questions et ajoutée seulement à partir du deuxième appel.
            # Mesurée sur les huit questions sans mot commun avec leur
            # article : améliore cinq fois, dégrade trois fois, médiane
            # +1 rang. Du bruit.
            #
            # Une vraie traduction, mesurée sur les mêmes huit : 8/8
            # améliorées, médiane +75 rangs, deux entrées dans le top-10
            # portées à six. En REMPLACEMENT elle dégraderait dix des
            # trente-deux questions de contrôle — celles où l'usager
            # employait déjà le mot juste.
            arguments: dict[str, object] = {"query": query.strip()}
            terms = (legal_terms or "").strip()
            previous = sum(
                observation.tool_name == tool
                for observation in state.tool_history)
            if terms:
                arguments["legal_terms"] = terms
            elif previous:
                # Reprise sans traduction fournie. La phrase constante est
                # conservée FAUTE DE MIEUX MESURÉ : sans elle, le second
                # appel serait identique au premier, donc servi par le
                # cache — la reformulation deviendrait un non-événement.
                # Son effet propre est nul (médiane +1 rang sur les huit
                # cas sans mot commun); elle ne tient ici que le rôle de
                # différenciateur. À remplacer par une vraie traduction du
                # planner, une fois celle-ci mesurée en reprise.
                arguments["query"] = query.strip() + (
                    "\nReformulation de recherche: identifier les règles, "
                    "recours, conditions et exceptions juridiquement "
                    "équivalents.")
            return arguments
        if tool in {"search_ccq_keywords", "search_cpc_keywords", "search_quebec_regulations"}:
            candidates = self._keyword_candidates(tool, query, thinking)
            previous = sum(
                observation.tool_name == tool for observation in state.tool_history
            )
            keyword = candidates[min(previous, len(candidates) - 1)]
            return {"keyword": keyword}
        if tool == "search_quebec_jurisprudence":
            return {"query": self._build_quebec_case_law_query(state)}
        if tool == "get_quebec_regulation":
            accepted = [item for item in state.usable_case_sources
                        if (item.get("source_url", "") if isinstance(item, dict)
                            else getattr(item, "source_url", ""))]
            if accepted and state.case_law_search_status in {
                    "candidates_pending_fetch", "candidate_pending_fetch"}:
                url = (accepted[0].get("source_url", "")
                       if isinstance(accepted[0], dict)
                       else accepted[0].source_url)
                return {"url": str(url)}
            # Les règlements québécois ont leur propre chaîne de provenance;
            # une URL est permise seulement depuis un résultat réussi de cet
            # outil, jamais depuis une recherche jurisprudentielle rejetée.
            for obs in reversed(state.tool_history):
                if obs.tool_name != "search_quebec_regulations" or not obs.ok:
                    continue
                if obs.source_urls:
                    return {"url": obs.source_urls[0]}
                embedded = re.search(r"https?://[^\s\"'<>]+",
                                     obs.normalized_response or "")
                if embedded:
                    return {"url": embedded.group(0).rstrip(".,;)")}
            return None
        if tool == "get_quebec_legal_info":
            return {"type": "eevlois"}
        if tool == "coverage":
            return {"doc_type": "cases"}
        if tool == "search_legal_documents":
            intent = self._extract_search_intent(thinking)
            req = state.scenario.request_type
            if req == "case_analysis":
                prior_searches = [o for o in state.tool_history
                                  if o.tool_name == "search_legal_documents"]
                if prior_searches:
                    doc_type = "cases"
                elif intent.case_name or intent.target_type == "cases":
                    doc_type = "cases"
                else:
                    doc_type = "laws"
            elif req == "law_or_regulation_identification":
                doc_type = "laws"
            elif req == "case_law_research":
                doc_type = "cases"
            else:
                doc_type = "cases"

            if intent.case_name:
                return {
                    "query": intent.case_name,
                    "search_type": "name",
                    "doc_type": doc_type,
                    "search_language": "fr",
                    "size": 5,
                }

            target = self._federal_statute_target(query) if doc_type == "laws" else None
            if target:
                return {
                    "query": target[0], "search_type": "name",
                    "doc_type": "laws", "search_language": "en",
                    "dataset": "LEGISLATION-FED", "size": 5,
                }
            if doc_type == "laws":
                return {
                    "query": query[:180], "doc_type": "laws",
                    "search_language": "fr",
                    "dataset": "LEGISLATION-FED", "size": 5,
                }
            if req == "case_law_research":
                target = self._federal_statute_target(query)
                if target:
                    return {"query": target[0], "doc_type": "cases",
                            "search_language": "en", "size": 5}
            search_q = " ".join(intent.keywords[:3]) if intent.keywords else query[:180]
            return {"query": search_q, "doc_type": doc_type,
                    "search_language": "fr", "size": 5}
        if tool == "fetch_document":
            is_law = state.scenario.request_type in {
                "law_or_regulation_identification", "case_analysis",
            }
            if is_law:
                citation = self._validated_federal_law_citation(state, query)
            else:
                citations = [c for o in state.tool_history for c in o.citations]
                citation = citations[0] if citations else ""
            if not citation:
                all_citations = [c for o in state.tool_history for c in o.citations]
                citation = all_citations[0] if all_citations else ""
            if not citation:
                citation = self._any_search_citation(state)
            if not citation and is_law:
                target = self._federal_statute_target(query)
                if target and target[0] in FEDERAL_KNOWN_CITATIONS:
                    citation = FEDERAL_KNOWN_CITATIONS[target[0]]
            if not citation:
                return None
            args = {"citation": citation, "output_language": "fr",
                    "doc_type": "laws" if is_law else "cases"}
            target = self._federal_statute_target(query) if is_law else None
            if (state.scenario.request_type == "case_analysis" and target and
                    target[0] == "Bankruptcy and Insolvency Act"):
                # L'art. 49 décrit la cession volontaire, les documents, le
                # séquestre officiel et la nomination du syndic. Une section
                # ciblée évite de tronquer les quelque 400 articles de la loi.
                args["section"] = "49"
            previous_fetches = [o for o in state.tool_history if o.tool_name == "fetch_document"]
            if len(previous_fetches) >= 2:
                return None
            if previous_fetches:
                args.update({"start_char": 6000, "end_char": 12000})
            elif getattr(state.scenario, "planned_failure_mode", None) == "truncated_source":
                args.update({"start_char": 0, "end_char": 6000})
            return args
        return None

    @classmethod
    def _any_search_citation(cls, state: ResearchState) -> str:
        """Extract any citation from search_legal_documents results."""
        for obs in reversed(state.tool_history):
            if obs.tool_name != "search_legal_documents" or not obs.ok:
                continue
            for result in cls._extract_search_results(obs):
                citation = result.get("citation_fr") or result.get("citation_en")
                if citation:
                    return str(citation)
        for obs in reversed(state.tool_history):
            if obs.tool_name != "search_legal_documents" or not obs.ok:
                continue
            for c in obs.citations:
                if c.strip():
                    return c.strip()
            citation_re = re.compile(
                CASE_CITATION_RE.pattern
                + r"|(?:RSC|LRC|SC|LC),?\s+\d{4},?\s*c\.?\s*[A-Z]?-?\d+"
            )
            text = obs.normalized_response or ""
            match = citation_re.search(text)
            if match:
                return match.group(0)
        return ""

    @staticmethod
    def _fold(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value or "")
        return "".join(
            character for character in normalized
            if not unicodedata.combining(character)
        ).casefold().replace("'", " ").replace("'", " ")

    @classmethod
    def _federal_statute_target(cls, query: str) -> Optional[tuple[str, tuple[str, ...]]]:
        folded = cls._fold(query)
        for markers, english_name, aliases in FEDERAL_STATUTES:
            if any(marker in folded for marker in markers):
                return english_name, aliases
        return None

    @classmethod
    def _extract_search_results(cls, observation) -> list[dict]:
        """Parse results from a search_legal_documents observation."""
        for source in (observation.normalized_response, observation.raw_response):
            for payload in cls._candidate_payloads(source):
                results = payload.get("results", []) if isinstance(payload, dict) else []
                if results:
                    return [r for r in results if isinstance(r, dict)]
        return []

    @staticmethod
    def _candidate_payloads(source) -> list[dict]:
        """Yield dict payloads from a normalized string, raw dict, or MCP wrapper."""
        candidates: list[dict] = []
        if isinstance(source, str):
            try:
                parsed = json.loads(source)
                if isinstance(parsed, dict):
                    candidates.append(parsed)
            except (TypeError, ValueError):
                pass
        elif isinstance(source, dict):
            candidates.append(source)
            sc = source.get("structuredContent")
            if isinstance(sc, dict):
                candidates.append(sc)
            for item in (source.get("content") or []):
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    try:
                        inner = json.loads(item["text"])
                        if isinstance(inner, dict):
                            candidates.append(inner)
                    except (TypeError, ValueError):
                        pass
        return candidates

    @classmethod
    def _validated_federal_law_citation(cls, state: ResearchState,
                                        query: str) -> str:
        """Retient seulement une loi fédérale dont le titre correspond au besoin."""
        target = cls._federal_statute_target(query)
        aliases = target[1] if target else ()
        for observation in reversed(state.tool_history):
            if observation.tool_name != "search_legal_documents" or not observation.ok:
                continue
            for result in cls._extract_search_results(observation):
                if cls._fold(str(result.get("dataset", ""))) != "legislation-fed":
                    continue
                names = [
                    cls._fold(str(result.get(key, "")))
                    for key in ("name_en", "name_fr")
                    if result.get(key)
                ]
                if aliases and not any(
                    cls._fold(alias) == name
                    for alias in aliases for name in names
                ):
                    continue
                citation = result.get("citation_fr") or result.get("citation_en")
                if citation:
                    return str(citation)
        return ""

    @staticmethod
    def _no_result(text: str) -> bool:
        stripped = (text or "").strip()
        return stripped in ("", "[]", "{}") or bool(NO_RESULT_RE.search(stripped))

    @staticmethod
    def _keyword_candidates(tool: str, query: str,
                            thinking: str = "") -> list[str]:
        folded = query.casefold()
        if tool == "search_ccq_keywords":
            if any(marker in folded for marker in ("bail", "baux", "locataire", "loyer", "logement")):
                return ["bail", "louage d'un logement"]
            if any(marker in folded for marker in ("limite", "clôture", "empièt", "bornage", "voisin")):
                return ["bornage", "empiètement"]
            if "vice" in folded:
                return ["vice caché", "garantie de qualité"]
            if "responsab" in folded or "préjudice" in folded:
                return ["responsabilité civile", "réparation du préjudice"]
            if any(marker in folded for marker in ("travail", "salari", "employeur")):
                return ["contrat de travail", "salarié"]
            thinking_kw = PlannerAgent._compact_keyword(thinking) if thinking else ""
            primary = thinking_kw if thinking_kw and thinking_kw != "droit applicable" else PlannerAgent._compact_keyword(query)
            return [primary, "obligation"]
        if tool == "search_cpc_keywords":
            if "mise en état" in folded or "prépar" in folded:
                return ["protocole de l'instance", "gestion de l'instance"]
            if "citation à comparaître" in folded or "témoin" in folded:
                return ["assignation d'un témoin", "citation à comparaître"]
            if "signif" in folded or "notifi" in folded:
                return ["signification", "notification"]
            if "injonction" in folded:
                return ["injonction", "injonction interlocutoire"]
            if "appel" in folded:
                return ["appel", "permission d'appeler"]
            thinking_kw = PlannerAgent._compact_keyword(thinking) if thinking else ""
            primary = thinking_kw if thinking_kw and thinking_kw != "droit applicable" else PlannerAgent._compact_keyword(query)
            return [primary, "gestion de l'instance"]
        if any(marker in folded for marker in ("eau", "potable")):
            return ["qualité de l'eau potable", "eau potable"]
        if any(marker in folded for marker in ("environnement", "impact", "activité")):
            return ["encadrement d'activités environnementales", "impact environnemental"]
        thinking_kw = PlannerAgent._compact_keyword(thinking) if thinking else ""
        primary = thinking_kw if thinking_kw and thinking_kw != "droit applicable" else PlannerAgent._compact_keyword(query)
        return [primary, "règlement Québec"]

    @staticmethod
    def _compact_keyword(query: str) -> str:
        stopwords = {
            "avec", "avoir", "cela", "cette", "comme", "comment", "dans", "donner",
            "faire", "informations", "mais", "peux", "pour", "pouvez", "quelles",
            "quels", "quoi", "savoir", "suis", "tout", "trouver", "voudrais",
        }
        words = [
            word for word in re.findall(r"[A-Za-zÀ-ÿ]{4,}", query.casefold())
            if word not in stopwords
        ]
        return " ".join(words[:3]) or "droit applicable"

    @staticmethod
    def _extract_article_nums_from_history(state: ResearchState) -> list[str]:
        """Extract article numbers from prior get_*_articles calls and semantic searches."""
        nums: list[str] = []
        seen: set[str] = set()
        for obs in state.tool_history:
            if obs.tool_name in {"get_ccq_articles", "get_cpc_articles"} and obs.ok:
                for n in numeros_demandes(obs.tool_name, obs.arguments):
                    if n not in seen:
                        nums.append(n)
                        seen.add(n)
            elif obs.tool_name in {"semantic_search_ccq", "semantic_search_cpc"} and obs.ok:
                for m in ARTICLE_LABEL_RE.finditer(obs.normalized_response or ""):
                    n = m.group(1)
                    if n not in seen:
                        nums.append(n)
                        seen.add(n)
        return nums

    @staticmethod
    def _extract_search_intent(thinking: str) -> SearchIntent:
        if not thinking or not thinking.strip():
            return SearchIntent(keywords=[], target_type="auto", case_name="")

        case_name = ""
        match = _CASE_NAME_RE.search(thinking)
        if match:
            case_name = (match.group(1)
                         or f"{match.group(2)} c. {match.group(3)}")

        words_in_thinking = set(re.findall(
            r"[a-zà-ÿéèêëàâäùûüôöîïç]+", thinking.casefold()))
        case_score = len(words_in_thinking & _CASE_INDICATOR_WORDS)
        law_score = len(words_in_thinking & _LAW_INDICATOR_WORDS)
        if case_name or case_score > law_score:
            target_type = "cases"
        elif law_score > case_score:
            target_type = "laws"
        else:
            target_type = "auto"

        all_words = re.findall(r"[A-Za-zÀ-ÿ]{4,}", thinking.casefold())
        exclude = _THINKING_STOPWORDS | _CASE_INDICATOR_WORDS | _LAW_INDICATOR_WORDS
        seen: set[str] = set()
        keywords: list[str] = []
        for word in all_words:
            if word not in exclude and word not in seen:
                seen.add(word)
                keywords.append(word)
            if len(keywords) >= 5:
                break

        return SearchIntent(keywords=keywords, target_type=target_type,
                            case_name=case_name)

    @staticmethod
    def _infer_jurisdiction(query: str, request_type: str) -> str:
        if request_type == "comparative_law":
            return "Québec et Canada (fédéral)"
        if request_type == "dataset_coverage" or "banque" in query.casefold():
            return "Canada (fédéral)"
        if request_type == "non_legal":
            return "sans objet"
        return "Québec"

    @staticmethod
    def _final(request_type: str, jurisdiction: str, need: str) -> PlannerDecision:
        return PlannerDecision(request_type=request_type, jurisdiction=jurisdiction,
                               decision=Decision.final_answer,
                               decision_trace=DecisionTrace(request_type=request_type,
                                                            jurisdiction=jurisdiction,
                                                            need=need, next_action="final_answer"))
