# -*- coding: utf-8 -*-
"""Agent de trajectoire : poursuite bornée et réponse finale strictement fondée."""

from __future__ import annotations

import json
import re

from .prompts import CHAT_WRITER_SUPPLEMENT, TRAJECTORY_ANSWER_SYSTEM
from .schemas import ResearchState


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.S)
PRECISE_ARTICLE_TYPES = {"exact_text_retrieval"}
ARTICLE_FETCH_TOOLS = {"get_ccq_articles", "get_cpc_articles"}
RETRIEVAL_ONLY_TOOLS = {"semantic_search_ccq", "semantic_search_cpc"}


def normalize_final_answer(text: str) -> str:
    """Retire les wrappers JSON/markdown parfois ajoutés par un petit modèle."""
    cleaned = _JSON_FENCE_RE.sub("", (text or "").strip()).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            payload = json.loads(cleaned)
        except ValueError:
            return cleaned
        if set(payload) == {"answer"} and isinstance(payload["answer"], str):
            return payload["answer"].strip()
    return cleaned


class TrajectoryAgent:
    def __init__(self, client=None, offline: bool = False,
                 chat_mode: bool = False):
        self.client = client
        self.offline = offline
        self.chat_mode = chat_mode

    def _system_prompt(self) -> str:
        if self.chat_mode:
            return TRAJECTORY_ANSWER_SYSTEM + CHAT_WRITER_SUPPLEMENT
        return TRAJECTORY_ANSWER_SYSTEM

    def final_answer(self, state: ResearchState) -> tuple[str, str]:
        """Retourne (thinking, answer) pour le tour final."""
        official_text = self._official_article_text(state)
        if state.scenario.request_type in PRECISE_ARTICLE_TYPES and official_text:
            art_nums = ", ".join(
                str(a) for o in state.tool_history
                if o.ok and o.tool_name in ARTICLE_FETCH_TOOLS
                for a in [o.arguments.get("start_article", "")]
                if a
            ) or "demandé"
            used_tool = next(
                (o.tool_name for o in state.tool_history
                 if o.ok and o.tool_name in ARTICLE_FETCH_TOOLS), "")
            code = ("Code civil du Québec" if "ccq" in used_tool
                    else "Code de procédure civile")
            thinking = (
                f"L'utilisateur a demandé le texte officiel de l'article {art_nums} "
                f"du {code}. J'ai récupéré le texte via l'outil approprié. "
                "Je le reproduis intégralement et mot pour mot, sans paraphrase "
                "ni explication ajoutée."
            )
            return thinking, official_text
        if self.offline:
            return self._offline_answer(state)
        if self.client is None:
            raise RuntimeError("client Teacher requis hors mode offline")
        evidence = [{
            "tool": o.tool_name, "content": o.normalized_response,
            "urls": o.source_urls, "citations": o.citations,
            "truncated": o.truncated, "error": o.error,
        } for o in state.tool_history if o.tool_name not in RETRIEVAL_ONLY_TOOLS]
        prompt = {
            "question": state.scenario.user_query,
            "messages_utilisateur": [
                message.content for message in state.messages
                if message.role.value == "user"
            ],
            "type_de_demande": state.scenario.request_type,
            "informations_manquantes": state.missing_critical_facts,
            "preuves_officielles_uniquement": evidence,
            "outils_de_selection_non_citables": [
                o.tool_name for o in state.tool_history
                if o.tool_name in RETRIEVAL_ONLY_TOOLS
            ],
            "instruction": (
                "Produis le RAISONNEMENT puis ---ANSWER--- puis la RÉPONSE. "
                "N'utilise aucun fait absent des messages utilisateur. "
                + (
                    "Rédige UNIQUEMENT l'explication du texte officiel fourni. "
                    "Ne recopie pas l'article : le contrôleur l'ajoutera mot pour mot."
                    if state.scenario.request_type == "article_explanation"
                    else "Adapte la structure à la question réellement posée."
                )
            ),
        }
        result = self.client.complete("trajectory_writer", [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ], temperature=0.1)
        thinking, answer = self._split_thinking_answer(result)
        answer = normalize_final_answer(answer)
        if state.scenario.request_type == "article_explanation" and official_text:
            answer = self._package_explanation(official_text, answer)
        return thinking, answer

    def repair(self, state: ResearchState, answer: str,
               thinking: str, instructions: list[str]) -> tuple[str, str]:
        """Retourne (thinking, answer) après réparation."""
        official_text = self._official_article_text(state)
        if state.scenario.request_type in PRECISE_ARTICLE_TYPES and official_text:
            return thinking, official_text
        if self.offline or not self.client:
            return thinking, answer
        result = self.client.complete("repair", [
            {"role": "system", "content": (
                self._system_prompt() +
                "\nRépare sans ajouter de fait, source, URL, article ou décision."
            )},
            {"role": "user", "content": json.dumps({"answer": answer, "instructions": instructions,
                                                     "sources": [
                                                         o.normalized_response
                                                         for o in state.tool_history
                                                         if o.tool_name not in RETRIEVAL_ONLY_TOOLS
                                                     ]}, ensure_ascii=False)},
        ], temperature=0.0)
        repaired_thinking, repaired_answer = self._split_thinking_answer(result)
        repaired_answer = normalize_final_answer(repaired_answer)
        if state.scenario.request_type == "article_explanation" and official_text:
            repaired_answer = self._package_explanation(official_text, repaired_answer)
        return repaired_thinking or thinking, repaired_answer

    _THINKING_LABEL_RE = re.compile(
        r"^(?:\s*-{3,}\s*)*\s*RAISONNEMENT\s*:?\s*", re.I)
    _ANSWER_LABEL_RE = re.compile(
        r"^(?:\s*-{3,}\s*)*\s*R[ÉE]PONSE\s*:?\s*", re.I)
    # Libellé avec deux-points (contenu sur la même ligne) OU titre nu en
    # majuscules seul sur sa ligne (contenu à la ligne suivante).
    _ANSWER_MARKER_RE = re.compile(
        r"(?m)^(?:\s*-{3,}\s*\n)*[ \t]*R[ÉE]PONSE[ \t]*(?::[ \t]*|\n+)")

    @classmethod
    def _strip_labels(cls, thinking: str, answer: str) -> tuple[str, str]:
        thinking = cls._THINKING_LABEL_RE.sub("", (thinking or "").strip())
        thinking = re.sub(r"(?:\s*-{3,}\s*)+$", "", thinking).strip()
        answer = cls._ANSWER_LABEL_RE.sub("", (answer or "").strip()).strip()
        return thinking, answer

    @classmethod
    def _split_thinking_answer(cls, text: str) -> tuple[str, str]:
        """Sépare le raisonnement IRAC et la réponse finale.

        Contrat : ``---ANSWER---``.  Tolère aussi un libellé ``RÉPONSE :``
        en début de ligne, qu'un modèle substitue parfois au séparateur.
        """
        if "---ANSWER---" in text:
            thinking, answer = text.split("---ANSWER---", 1)
            return cls._strip_labels(thinking, answer)
        matches = list(cls._ANSWER_MARKER_RE.finditer(text))
        if matches:
            marker = matches[-1]
            return cls._strip_labels(text[:marker.start()], text[marker.end():])
        return "", normalize_final_answer(text)

    @staticmethod
    def _package_explanation(official_text: str, draft: str) -> str:
        explanation = (draft or "").strip()
        if official_text in explanation:
            explanation = explanation.replace(official_text, "", 1).strip()
        explanation = re.sub(r"^#{0,3}\s*explication\s*[:—-]?\s*", "", explanation,
                             flags=re.I).strip()
        if not explanation:
            explanation = "Aucune explication supplémentaire étayée n'a été produite."
        return f"{official_text}\n\nExplication\n{explanation}"

    @staticmethod
    def _official_article_text(state: ResearchState) -> str:
        if state.scenario.request_type not in (
            PRECISE_ARTICLE_TYPES | {"article_explanation"}
        ):
            return ""
        for observation in reversed(state.tool_history):
            if (observation.tool_name in ARTICLE_FETCH_TOOLS and observation.ok and
                    not observation.truncated and observation.normalized_response.strip()):
                return observation.normalized_response.strip()
        return ""

    @staticmethod
    def _offline_answer(state: ResearchState) -> tuple[str, str]:
        if state.scenario.request_type == "non_legal":
            return (
                "Cette demande n’est pas juridique. Je réponds directement sans "
                "recherche dans les sources législatives.",
                "Bonjour! Je peux vous aider à identifier et rechercher des sources "
                "juridiques canadiennes ou québécoises.",
            )
        if state.scenario.request_type == "article_explanation":
            official = TrajectoryAgent._official_article_text(state)
            if official:
                return (
                    "L’utilisateur demande l’explication d’un article. J’ai récupéré "
                    "le texte officiel et je l’explique uniquement à partir de cette source.",
                    f"{official}\n\nExplication\n"
                    "Cet article est expliqué uniquement à partir de son texte officiel "
                    "reproduit ci-dessus.",
                )
        if not state.tool_history:
            return (
                "Aucun outil n’a été appelé car des précisions essentielles manquent.",
                "Je ne peux pas conclure sans les précisions essentielles demandées "
                "sur les faits et la juridiction.",
            )
        failures = [o for o in state.tool_history if not o.ok]
        if failures:
            return (
                "L’outil MCP a retourné une erreur. Sans texte officiel, je ne "
                "peux pas fabriquer de conclusion juridique.",
                "Limite de recherche — l’outil MCP a retourné une erreur. Je ne "
                "dispose donc pas du texte officiel nécessaire et je ne vais pas "
                "fabriquer de règle ou de conclusion.",
            )
        usable = [
            o for o in state.tool_history
            if o.normalized_response.strip() and o.tool_name not in RETRIEVAL_ONLY_TOOLS
        ]
        if not usable:
            return (
                "Aucune source exploitable n’a été récupérée malgré les recherches.",
                "La recherche n’a retourné aucun résultat exploitable, même après "
                "une reformulation limitée. Je ne peux pas identifier une source "
                "fiable ni conclure sur le fond.",
            )
        excerpts = []
        for obs in usable[-2:]:
            excerpt = obs.normalized_response[:500].strip()
            excerpts.append(f"- {obs.tool_name}: {excerpt}")
        limits = []
        if any(o.truncated for o in usable):
            limits.append("Au moins une source a été tronquée; aucune conclusion ne "
                          "porte sur la partie non récupérée.")
        urls = list(dict.fromkeys(u for o in usable for u in o.source_urls))
        source_line = "\nSources récupérées: " + ", ".join(urls) if urls else ""
        tools_used = ", ".join(dict.fromkeys(o.tool_name for o in usable))
        thinking = (
            f"J’ai récupéré des sources via {tools_used}. "
            "Je structure ma réponse avec les faits retenus, les règles "
            "récupérées, leur application aux faits de l’utilisateur et les "
            "limites de l’analyse."
        )
        answer = (
            "Faits de l’utilisateur — la question est analysée telle qu’elle a "
            "été formulée.\n\n"
            "Règles et documents récupérés —\n" + "\n".join(excerpts) + "\n\n"
            "Application — ces extraits constituent les seules bases disponibles; "
            "une application définitive exigerait la vérification du texte complet "
            "et, selon les faits, un avis professionnel.\n\n"
            "Limites — " + (" ".join(limits) if limits else
                            "la réponse reste limitée aux sources récupérées.")
            + source_line
        )
        return thinking, answer
