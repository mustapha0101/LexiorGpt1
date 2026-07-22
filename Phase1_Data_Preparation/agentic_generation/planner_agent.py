# -*- coding: utf-8 -*-
"""Planner : une seule prochaine action, validée contre le catalogue."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .prompts import CHAT_PLANNER_SUPPLEMENT, planner_system_prompt
from .schemas import Decision, DecisionTrace, PlannerDecision, ResearchState
from .tool_catalog import ToolCatalog
from .validators import validate_next_action, validate_planner_decision

ARTICLE_RE = re.compile(r"\b(?:article\s+)?(\d{1,4}(?:\.\d+)?)\b", re.I)
ARTICLE_LABEL_RE = re.compile(r"\barticle\s+(\d{1,4}(?:\.\d+)?)\b", re.I)
NO_RESULT_RE = re.compile(
    r"(?:aucun(?:e)?\s+(?:article|résultat|document|décision)|"
    r"rien\s+trouvé|no\s+results?)",
    re.I,
)

# ── Garde-fou juridictionnel du mode chat ────────────────────────────────
# Outils dont les sources sont exclusivement québécoises.
QC_ONLY_TOOLS = {
    "semantic_search_ccq", "semantic_search_cpc",
    "get_ccq_articles", "get_cpc_articles",
    "search_ccq_keywords", "search_cpc_keywords",
    "search_quebec_regulations", "get_quebec_regulation",
    "get_quebec_legal_info", "search_quebec_jurisprudence",
}

_PROVINCE_RE = re.compile(
    r"\b(ontario|alberta|manitoba|saskatchewan|"
    r"colombie[- ]britannique|british columbia|"
    r"nouvelle[- ][ée]cosse|nova scotia|"
    r"nouveau[- ]brunswick|new brunswick|"
    r"terre[- ]neuve(?:[- ]et[- ]labrador)?|newfoundland|"
    r"[îi]le[- ]du[- ]prince[- ][ée]douard|prince edward island|"
    r"yukon|nunavut|territoires du nord[- ]ouest|northwest territories)\b",
    re.I,
)
_QC_MENTION_RE = re.compile(
    r"\b(qu[ée]bec|montr[ée]al|gatineau|laval|sherbrooke|trois[- ]rivi[èe]res)\b",
    re.I,
)
_YES_RE = re.compile(r"^\s*(oui|yes|ouais|exactement|c'est ça)\s*[.!]?\s*$", re.I)
_NO_RE = re.compile(r"^\s*(non|no|nope|pas au qu[ée]bec)\s*[.!]?\s*$", re.I)


class QuebecToolsBlocked(ValueError):
    """Outil québécois choisi alors que l'utilisateur n'est pas au Québec."""

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
                 chat_mode: bool = False):
        self.catalog = catalog
        self.client = client
        self.offline = offline
        # chat_mode : requête libre sans route scriptée — les gardes fondés
        # sur expected_route/request_type ne s'appliquent pas.
        self.chat_mode = chat_mode

    def decide(self, state: ResearchState) -> PlannerDecision:
        if self.offline:
            decision = self._offline_decide(state)
            self._raise_if_invalid(state, decision)
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
                decision = self._guard_duplicate_call(state, decision)
                if not self.chat_mode:
                    decision = self._guard_tool_compatibility(state, decision)
                    decision = self._guard_required_tools(state, decision)
                else:
                    decision = self._guard_chat_jurisdiction(
                        state, decision, retried=bool(feedback))
                decision = self._guard_budget(state, decision)
                self._raise_if_invalid(state, decision)
                return decision
            except ValueError as exc:
                # ValidationError pydantic incluse (sous-classe de ValueError).
                if feedback:
                    raise
                feedback = str(exc) or "décision invalide"

    def _raise_if_invalid(self, state: ResearchState,
                          decision: PlannerDecision) -> None:
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
            raise ValueError("décision Planner invalide : " + "; ".join(errors))

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
                        need=f"outil requis non encore appelé",
                        next_action=f"call_tool:{tool}"),
                )
        return decision

    @staticmethod
    def _chat_jurisdiction_hint(state: ResearchState) -> Optional[str]:
        """Juridiction déduite DÉTERMINISTIQUEMENT de la conversation.

        Parcourt les messages dans l'ordre; le signal le plus récent
        l'emporte. Retourne « Québec », un nom de province, ou
        « hors Québec (province non précisée) » — None si rien ne tranche.
        """
        hint: Optional[str] = None
        messages = state.messages
        for index, message in enumerate(messages):
            if message.role.value != "user":
                continue
            province = _PROVINCE_RE.search(message.content)
            if province:
                hint = province.group(0).title()
                continue
            if _QC_MENTION_RE.search(message.content):
                hint = "Québec"
                continue
            previous_is_quebec_question = (
                index > 0
                and messages[index - 1].role.value == "assistant"
                and "québec" in messages[index - 1].content.lower()
                and messages[index - 1].content.rstrip().endswith("?")
            )
            if previous_is_quebec_question:
                if _YES_RE.match(message.content):
                    hint = "Québec"
                elif _NO_RE.match(message.content):
                    hint = "hors Québec (province non précisée)"
        return hint

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

    def _guard_budget(self, state: ResearchState,
                      decision: PlannerDecision) -> PlannerDecision:
        """Budget d'outils épuisé : forcer la synthèse au lieu d'un appel."""
        if decision.decision != Decision.call_tool:
            return decision
        if len(state.tool_history) < state.max_tool_calls:
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
        if policy and policy.required_capabilities:
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
                        need=f"redirection vers outil compatible",
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
            "remaining_tool_calls": state.max_tool_calls - len(state.tool_history),
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
        if len(state.tool_history) >= state.max_tool_calls:
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
        if len(state.tool_history) >= len(route):
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
        tool = route[len(state.tool_history)]
        args = self._arguments(tool, state)
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
    def _generate_offline_thinking(tool: str, args: dict, state: ResearchState) -> str:
        """Génère un thinking en langue naturelle pour le mode offline."""
        query = state.scenario.user_query
        request_type = state.scenario.request_type
        step = len(state.tool_history)

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

    def _effective_route(self, state: ResearchState) -> list[str]:
        request_type = state.scenario.request_type
        steps = state.scenario.expected_route.steps
        jurisdiction = getattr(state.scenario, "jurisdiction_status", "")
        failure_mode = getattr(state.scenario, "planned_failure_mode", None) or getattr(state.scenario, "failure_mode", None)
        clarification_stage = getattr(state.scenario, "clarification_stage", "none")
        route: list[str] = []
        for step in steps:
            if not step.optional:
                route.append(step.tool)
                continue
            if (request_type == "case_analysis"
                    and step.tool == "semantic_search_ccq"):
                route.append(step.tool)
            elif (request_type == "procedure_guidance"
                  and step.tool == "semantic_search_cpc"):
                route.append(step.tool)
            elif (jurisdiction == "supported_federal"
                  and step.tool == "fetch_document"):
                route.append(step.tool)
        for search_tool in ("semantic_search_ccq", "semantic_search_cpc"):
            if search_tool not in route or route.count(search_tool) > 1:
                continue
            searches = [
                o for o in state.tool_history
                if o.tool_name == search_tool
            ]
            if searches and self._no_result(searches[0].normalized_response):
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

    def _arguments(self, tool: str, state: ResearchState,
                   thinking: str = "") -> Optional[dict]:
        query = "\n".join(
            message.content for message in state.messages
            if message.role.value == "user"
        ) or state.scenario.user_query
        if tool in {"get_ccq_articles", "get_cpc_articles"}:
            already_fetched: set[float] = set()
            for obs in state.tool_history:
                if obs.tool_name == tool and obs.ok:
                    sa = obs.arguments.get("start_article")
                    ea = obs.arguments.get("end_article", sa)
                    if sa is not None:
                        already_fetched.update(
                            float(v) for v in range(int(sa), int(ea or sa) + 1)
                        )
            if state.tool_history:
                search_tool = (
                    "semantic_search_ccq"
                    if tool == "get_ccq_articles"
                    else "semantic_search_cpc"
                )
                search_text = "\n".join(
                    observation.normalized_response
                    for observation in state.tool_history
                    if observation.ok and observation.tool_name == search_tool
                )
                candidates = ARTICLE_LABEL_RE.findall(search_text)
                if not candidates:
                    candidates = ARTICLE_RE.findall(query)
            else:
                candidates = ARTICLE_RE.findall(query)
            candidates = [
                c for c in candidates if float(c) not in already_fetched
            ]
            if not candidates:
                fallback = self._topic_article(tool, query, already_fetched)
                if fallback is not None:
                    return fallback
                return None
            values = [float(value) for value in candidates[:3]]
            primary = values[0]
            nearby = [value for value in values if abs(value - primary) <= 5]
            start, end = min(nearby), max(nearby)

            def json_number(value: float):
                return int(value) if value.is_integer() else value

            arguments = {"start_article": json_number(start)}
            if end != start:
                arguments["end_article"] = json_number(end)
            return arguments
        if tool in {"semantic_search_ccq", "semantic_search_cpc"}:
            previous = sum(
                observation.tool_name == tool for observation in state.tool_history
            )
            semantic_query = query.strip()
            if previous:
                semantic_query += (
                    "\nReformulation de recherche: identifier les règles, recours, "
                    "conditions et exceptions juridiquement équivalents."
                )
            return {"query": semantic_query}
        if tool in {"search_ccq_keywords", "search_cpc_keywords", "search_quebec_regulations"}:
            candidates = self._keyword_candidates(tool, query, thinking)
            previous = sum(
                observation.tool_name == tool for observation in state.tool_history
            )
            keyword = candidates[min(previous, len(candidates) - 1)]
            return {"keyword": keyword}
        if tool == "search_quebec_jurisprudence":
            article_nums = self._extract_article_nums_from_history(state)
            situation = self._compact_keyword(query)
            if article_nums:
                article_part = " ".join(
                    f"article {n}" for n in article_nums[:2]
                )
                jurisprudence_query = f"{article_part} {situation}"
            else:
                intent = self._extract_search_intent(thinking)
                if intent.keywords:
                    jurisprudence_query = " ".join(intent.keywords[:4])
                else:
                    jurisprudence_query = situation
            return {"query": jurisprudence_query[:200]}
        if tool == "get_quebec_regulation":
            urls = [u for o in state.tool_history for u in o.source_urls]
            if not urls:
                url_re = re.compile(r"https?://[^\s\"',\]\)]+")
                for obs in reversed(state.tool_history):
                    if obs.tool_name == "search_quebec_regulations" and obs.ok:
                        urls = url_re.findall(obs.normalized_response or "")
                        if urls:
                            break
            return {"url": urls[0]} if urls else None
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

    _CCQ_TOPIC_ARTICLES: dict[tuple[str, ...], int] = {
        ("vice caché", "vice", "garantie de qualité"): 1726,
        ("responsab", "préjudice", "dommage"): 1457,
        ("contrat", "obligation"): 1375,
        ("vente", "vendeur", "acheteur"): 1708,
        ("bail", "locataire", "loyer", "logement"): 1851,
        ("mandat", "mandataire"): 2130,
        ("succession", "héritier", "testament"): 613,
        ("hypothèque", "sûreté"): 2660,
        ("prescription", "délai"): 2875,
        ("mariage", "divorce", "séparation"): 392,
        ("propriété", "bien", "immeuble"): 947,
        ("tutelle", "mineur", "curatelle", "protection"): 177,
        ("société", "associé", "entreprise"): 2186,
        ("assurance",): 2389,
        ("donation", "don"): 1806,
        ("servitude",): 1177,
        ("usufruit",): 1120,
        ("copropriété", "condo"): 1038,
        ("travail", "salarié", "employeur"): 2085,
    }

    _CPC_TOPIC_ARTICLES: dict[tuple[str, ...], int] = {
        ("signif", "notifi"): 109,
        ("injonction",): 509,
        ("appel",): 351,
        ("exécution",): 681,
        ("médiation", "conférence"): 161,
        ("demande", "action", "recours"): 141,
        ("preuve", "témoin"): 251,
        ("saisie",): 696,
    }

    _CCQ_DEFAULT_ARTICLE = 1375
    _CPC_DEFAULT_ARTICLE = 1

    @classmethod
    def _topic_article(cls, tool: str, query: str,
                       already_fetched: set[float]) -> Optional[dict]:
        """Fallback article number from query topic when search failed."""
        folded = query.casefold()
        is_ccq = "ccq" in tool
        topics = cls._CCQ_TOPIC_ARTICLES if is_ccq else cls._CPC_TOPIC_ARTICLES
        for markers, article in topics.items():
            if float(article) in already_fetched:
                continue
            if any(m in folded for m in markers):
                return {"start_article": article}
        default = cls._CCQ_DEFAULT_ARTICLE if is_ccq else cls._CPC_DEFAULT_ARTICLE
        if float(default) not in already_fetched:
            return {"start_article": default}
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
                r"\b\d{4}\s+(?:SCC|CSC|FC|CF|FCA|CAF|QCCA|QCCS|QCCQ)\s+\d+\b"
                r"|(?:RSC|LRC|SC|LC),?\s+\d{4},?\s*c\.?\s*[A-Z]?-?\d+"
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
                sa = obs.arguments.get("start_article")
                if sa is not None:
                    n = str(int(sa)) if isinstance(sa, float) and sa == int(sa) else str(sa)
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
