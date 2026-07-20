# -*- coding: utf-8 -*-
"""Planner : une seule prochaine action, validée contre le catalogue."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Optional

from .prompts import planner_system_prompt
from .schemas import Decision, DecisionTrace, PlannerDecision, ResearchState
from .tool_catalog import ToolCatalog
from .validators import validate_planner_decision

ARTICLE_RE = re.compile(r"\b(?:article\s+)?(\d{1,4}(?:\.\d+)?)\b", re.I)
ARTICLE_LABEL_RE = re.compile(r"\barticle\s+(\d{1,4}(?:\.\d+)?)\b", re.I)
NO_RESULT_RE = re.compile(
    r"(?:aucun(?:e)?\s+(?:article|résultat|document|décision)|"
    r"rien\s+trouvé|no\s+results?)",
    re.I,
)

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
)


class PlannerAgent:
    def __init__(self, catalog: ToolCatalog, client=None, offline: bool = False):
        self.catalog = catalog
        self.client = client
        self.offline = offline

    def decide(self, state: ResearchState) -> PlannerDecision:
        decision = self._offline_decide(state) if self.offline else self._teacher_decide(state)
        if not self.offline and decision.decision == Decision.call_tool and decision.next_tool:
            decision = self._validate_arguments(state, decision)
        errors = validate_planner_decision(decision, self.catalog)
        if decision.decision == Decision.call_tool and decision.next_tool:
            allowed = set(state.scenario.expected_route.allowed_tools())
            if decision.next_tool not in allowed:
                errors.append(
                    f"outil {decision.next_tool} incompatible avec la catégorie "
                    f"{state.scenario.request_type}"
                )
        if state.scenario.expected_route.no_tool and decision.decision == Decision.call_tool:
            errors.append("appel d'outil interdit pour cette demande")
        if errors:
            raise ValueError("décision Planner invalide : " + "; ".join(errors))
        return decision

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
        reconstructed = self._arguments(decision.next_tool, state)
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

    def _teacher_decide(self, state: ResearchState) -> PlannerDecision:
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
        raw = self.client.complete_json("planner", [
            {"role": "system", "content": planner_system_prompt(self.catalog)},
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
        ], temperature=0.0)
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
        if scenario.expected_route.requires_clarification and not clarified:
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
            required_sources=scenario.expected_source_types,
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
        category = state.scenario.request_type
        steps = state.scenario.expected_route.steps
        route: list[str] = []
        for step in steps:
            if not step.optional:
                route.append(step.tool)
                continue
            if (category in {"cas_civil_quebecois", "clarification_puis_recherche"}
                    and step.tool == "semantic_search_ccq"):
                route.append(step.tool)
            elif (category == "cas_procedure_quebecoise" and
                  step.tool == "semantic_search_cpc"):
                route.append(step.tool)
            elif category == "jurisprudence_federale" and step.tool == "fetch_document":
                route.append(step.tool)
            elif category == "source_trop_longue" and step.tool == "fetch_document":
                route.append(step.tool)
            elif category == "resultat_vide" and step.tool == "semantic_search_ccq" and state.tool_history:
                route.append(step.tool)
        # Une recherche thématique sans résultat autorise exactement une
        # reformulation. La seconde recherche reste dans la route lors des
        # tours suivants, ce qui permet ensuite de récupérer l'article trouvé.
        if category != "resultat_vide":
            for search_tool in ("semantic_search_ccq", "semantic_search_cpc"):
                if search_tool not in route or route.count(search_tool) > 1:
                    continue
                searches = [
                    observation for observation in state.tool_history
                    if observation.tool_name == search_tool
                ]
                if searches and self._no_result(searches[0].normalized_response):
                    route.insert(route.index(search_tool) + 1, search_tool)
        return route

    def _arguments(self, tool: str, state: ResearchState) -> Optional[dict]:
        query = "\n".join(
            message.content for message in state.messages
            if message.role.value == "user"
        ) or state.scenario.user_query
        if tool in {"get_ccq_articles", "get_cpc_articles"}:
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
            else:
                candidates = ARTICLE_RE.findall(query)
            if not candidates:
                return None
            values = [float(value) for value in candidates[:3]]
            primary = values[0]
            # Les codes regroupent souvent une même règle sur quelques
            # dispositions consécutives (ex. CPC 269 à 274 pour les témoins).
            # On récupère ce petit bloc officiel, sans élargir aux candidats
            # thématiquement proches mais éloignés dans le Code.
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
            candidates = self._keyword_candidates(tool, query)
            previous = sum(
                observation.tool_name == tool for observation in state.tool_history
            )
            keyword = candidates[min(previous, len(candidates) - 1)]
            return {"keyword": keyword}
        if tool == "search_quebec_jurisprudence":
            return {"query": "vice caché application faits" if "vice" in query.casefold() else query[:180]}
        if tool == "get_quebec_regulation":
            urls = [u for o in state.tool_history for u in o.source_urls]
            return {"url": urls[0]} if urls else None
        if tool == "get_quebec_legal_info":
            return {"type": "eevlois"}
        if tool == "coverage":
            return {"doc_type": "cases"}
        if tool == "search_legal_documents":
            doc_type = "laws" if state.scenario.request_type in {"loi_federale", "cas_federal_concret", "source_trop_longue"} else "cases"
            target = self._federal_statute_target(query) if doc_type == "laws" else None
            if target:
                return {
                    "query": target[0], "search_type": "name",
                    "doc_type": "laws", "search_language": "en",
                    "dataset": "LEGISLATION-FED", "size": 5,
                }
            return {"query": query[:180], "doc_type": doc_type,
                    "search_language": "fr", "size": 5}
        if tool == "fetch_document":
            is_law = state.scenario.request_type in {
                "loi_federale", "cas_federal_concret", "source_trop_longue"
            }
            if is_law:
                citation = self._validated_federal_law_citation(state, query)
            else:
                citations = [c for o in state.tool_history for c in o.citations]
                citation = citations[0] if citations else ""
            if not citation:
                return None
            args = {"citation": citation, "output_language": "fr",
                    "doc_type": "laws" if is_law else "cases"}
            target = self._federal_statute_target(query) if is_law else None
            if (state.scenario.request_type == "cas_federal_concret" and target and
                    target[0] == "Bankruptcy and Insolvency Act"):
                # L'art. 49 décrit la cession volontaire, les documents, le
                # séquestre officiel et la nomination du syndic. Une section
                # ciblée évite de tronquer les quelque 400 articles de la loi.
                args["section"] = "49"
            previous_fetches = [o for o in state.tool_history if o.tool_name == "fetch_document"]
            if previous_fetches:
                args.update({"start_char": 6000, "end_char": 12000})
            elif state.scenario.request_type == "source_trop_longue":
                args.update({"start_char": 0, "end_char": 6000})
            return args
        return None

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
    def _validated_federal_law_citation(cls, state: ResearchState,
                                        query: str) -> str:
        """Retient seulement une loi fédérale dont le titre correspond au besoin."""
        target = cls._federal_statute_target(query)
        aliases = target[1] if target else ()
        for observation in reversed(state.tool_history):
            if observation.tool_name != "search_legal_documents" or not observation.ok:
                continue
            try:
                payload = json.loads(observation.normalized_response)
            except (TypeError, ValueError):
                continue
            results = payload.get("results", []) if isinstance(payload, dict) else []
            for result in results:
                if not isinstance(result, dict):
                    continue
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
    def _keyword_candidates(tool: str, query: str) -> list[str]:
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
            return [PlannerAgent._compact_keyword(query), "obligation"]
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
            return [PlannerAgent._compact_keyword(query), "gestion de l'instance"]
        if any(marker in folded for marker in ("eau", "potable")):
            return ["qualité de l'eau potable", "eau potable"]
        if any(marker in folded for marker in ("environnement", "impact", "activité")):
            return ["encadrement d'activités environnementales", "impact environnemental"]
        return [PlannerAgent._compact_keyword(query), "règlement Québec"]

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
    def _infer_jurisdiction(query: str, request_type: str) -> str:
        if request_type == "comparaison_quebec_federal":
            return "Québec et Canada (fédéral)"
        if request_type in {"loi_federale", "jurisprudence_federale",
                            "cas_federal_concret", "couverture_dataset"} or "banque" in query.casefold():
            return "Canada (fédéral)"
        if request_type in {"juridiction_ambigue", "question_incomplete"}:
            return "indéterminée"
        if request_type == "question_non_juridique":
            return "sans objet"
        return "Québec"

    @staticmethod
    def _final(request_type: str, jurisdiction: str, need: str) -> PlannerDecision:
        return PlannerDecision(request_type=request_type, jurisdiction=jurisdiction,
                               decision=Decision.final_answer,
                               decision_trace=DecisionTrace(request_type=request_type,
                                                            jurisdiction=jurisdiction,
                                                            need=need, next_action="final_answer"))
