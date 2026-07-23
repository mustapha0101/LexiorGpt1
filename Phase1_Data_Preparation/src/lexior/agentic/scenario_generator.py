# -*- coding: utf-8 -*-
"""Générateur de scénarios : LLM en production, fixtures déterministes offline."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Optional

from .prompts import SCENARIO_GENERATOR_SYSTEM, scenario_user_prompt
from .schemas import ScenarioSpec
from .taxonomy import (
    RequestTypeSpec, get_request_type, sample_request_type,
    sample_clarification_stage, sample_jurisdiction, sample_failure_mode,
)


ARTICLE_RE = re.compile(r"\b(?:article\s+)?(\d{1,4}(?:\.\d+)?)\b", re.I)


FIXTURE_QUERIES: dict[str, str] = {
    "case_analysis":
        "J'ai acheté une maison au Québec et découvert après la vente un "
        "vice caché important. Quels sont mes recours?",
    "procedure_guidance":
        "Au Québec, une demande m'a été signifiée tardivement. "
        "Quelles conséquences sont possibles?",
    "topic_research":
        "Quels articles du Code civil du Québec portent sur les vices cachés?",
    "exact_text_retrieval":
        "Quel est le texte officiel de l'article 1457 du Code civil du Québec?",
    "article_explanation":
        "Que signifie l'article 1457 du Code civil du Québec?",
    "case_law_research":
        "Pouvez-vous trouver des décisions québécoises récentes sur "
        "les vices cachés?",
    "law_or_regulation_identification":
        "Quel règlement québécois encadre les activités selon leur impact "
        "environnemental?",
    "legislative_status_verification":
        "Comment vérifier si une modification législative québécoise est "
        "entrée en vigueur?",
    "document_analysis":
        "J'ai reçu une mise en demeure pour un bail commercial. "
        "Quels sont les points à vérifier?",
    "comparative_law":
        "Comparez les sources québécoises et fédérales applicables à "
        "cette opération bancaire.",
    "dataset_coverage":
        "Le corpus contient-il des décisions de la Cour suprême du Canada?",
    "non_legal":
        "Bonjour! Comment allez-vous?",
}

VERIFIED_CCQ_ARTICLES = (
    1, 6, 7, 10, 11, 35, 1375, 1376, 1457, 1458, 1463, 1474,
    1607, 1611, 1708, 1713, 1726, 1739, 2085, 2098, 2130, 2803,
    2843, 2925, 2938, 3148,
)
VERIFIED_CPC_ARTICLES = (
    1, 2, 9, 18, 20, 49, 84, 100, 145, 168, 171, 202, 340, 341,
    346, 491, 500, 511, 529, 535, 570,
)

_FEDERAL_MARKERS = (
    "faill", "insolv", "banque", "bancaire", "brevet", "marque de commerce",
    "maritime", "fédéral", "canada",
)
_CIVIL_MARKERS = (
    "maison", "immeuble", "vendeur", "vente", "contrat", "logement",
    "locataire", "propriétaire", "voisin", "responsabil", "préjudice",
    "vice caché",
)
_PENAL_MARKERS = ("amende", "stationnement", "infraction", "criminel", "pénal")
_LEGAL_MARKERS = (
    "article", "code civil", "tribunal", "loi", "juridique", "recours",
    "contrat", "responsabilité", "règlement",
)


class ScenarioGenerator:
    def __init__(self, client=None, seed: int = 3407, offline: bool = False,
                 request_type_weights: Optional[dict[str, float]] = None,
                 jurisdiction_weights: Optional[dict[str, float]] = None,
                 clarification_stage_weights: Optional[dict[str, float]] = None,
                 failure_mode_weights: Optional[dict[str, float]] = None,
                 failure_injection_rate: float = 0.07,
                 anchor_bank=None,
                 # backward compat
                 taxonomy_proportions: Optional[dict[str, float]] = None):
        self.client = client
        self.rng = random.Random(seed)
        self.seed = seed
        self.offline = offline
        self.request_type_weights = request_type_weights or taxonomy_proportions or {}
        self.jurisdiction_weights = jurisdiction_weights or {}
        self.clarification_stage_weights = clarification_stage_weights or {}
        self.failure_mode_weights = failure_mode_weights or {}
        self.failure_injection_rate = failure_injection_rate
        self.anchor_bank = anchor_bank
        self.index = 0

    def generate(
        self,
        request_type_name: Optional[str] = None,
        clarification_stage: Optional[str] = None,
        jurisdiction_status: Optional[str] = None,
        failure_mode: Optional[str] = None,
    ) -> ScenarioSpec:
        rt = (get_request_type(request_type_name) if request_type_name
              else sample_request_type(self.rng, self.request_type_weights))
        self.index += 1

        if clarification_stage is None:
            clarification_stage = sample_clarification_stage(
                self.rng, rt, self.clarification_stage_weights)
        if jurisdiction_status is None:
            jurisdiction_status = sample_jurisdiction(
                self.rng, self.jurisdiction_weights)
        if failure_mode is None:
            failure_mode = sample_failure_mode(
                self.rng, self.failure_injection_rate,
                self.failure_mode_weights)

        if self.offline:
            return self._fixture(rt, clarification_stage,
                                 jurisdiction_status, failure_mode)
        if self.client is None:
            raise RuntimeError("client Teacher requis hors mode offline")

        article_anchor = self._article_anchor(rt.name)
        federal_anchor = self._federal_anchor(rt.name, jurisdiction_status)
        prompt = scenario_user_prompt(
            rt.name, rt.description,
            jurisdiction_status, str(self.seed + self.index),
            clarification_stage=clarification_stage,
            article_anchor=str(article_anchor) if article_anchor else None,
            federal_anchor=federal_anchor,
        )
        raw = self.client.complete_json("scenario_generator", [
            {"role": "system", "content": SCENARIO_GENERATOR_SYSTEM},
            {"role": "user", "content": prompt},
        ])

        raw["request_type"] = rt.name
        raw["expected_route"] = rt.expected_route.model_dump(mode="json")
        raw["legal_domain"] = rt.legal_domain
        raw["jurisdiction_status"] = jurisdiction_status
        raw["clarification_stage"] = clarification_stage
        raw["source_intent"] = rt.default_source_intents

        if failure_mode:
            raw["planned_failure_mode"] = failure_mode

        if article_anchor:
            raw["user_query"] = self._anchored_query(
                rt.name, str(raw.get("user_query", "")), article_anchor)
            facts = dict(raw.get("facts_provided") or {})
            facts["article_number"] = article_anchor
            raw["facts_provided"] = facts

        self._enforce_contract(raw, rt,
                               has_federal_anchor=bool(federal_anchor))
        self._enforce_jurisdiction(raw)
        self._enforce_clarification(raw)

        query = str(raw.get("user_query", ""))
        raw["scenario_id"] = hashlib.sha256(
            f"{self.seed}:{rt.name}:{self.index}:{query}".encode()
        ).hexdigest()[:16]
        facts_shape = sorted((raw.get("facts_provided") or {}).keys())
        raw["scenario_family_id"] = hashlib.sha256(
            f"{rt.name}:{facts_shape}".encode()).hexdigest()[:16]

        return ScenarioSpec.model_validate(raw)

    def _enforce_contract(self, raw: dict, rt: RequestTypeSpec,
                          has_federal_anchor: bool = False) -> None:
        query = str(raw.get("user_query", ""))
        folded = query.casefold()

        if rt.name == "case_analysis":
            clarification = raw.get("clarification_stage", "none")
            jurisdiction = raw.get("jurisdiction_status", "undetermined")
            if clarification == "before_search":
                if not any(m in folded for m in _CIVIL_MARKERS):
                    raw["user_query"] = (
                        "Quels recours ai-je contre le vendeur de ma maison?")
                    raw["synthetic_clarification_answer"] = (
                        "La maison est au Québec et j'ai découvert un vice "
                        "caché important après la vente.")
                    raw["facts_provided"] = {}
                    raw["facts_required_before_search"] = [
                        "lieu de l'immeuble", "nature précise du problème"]
            if (jurisdiction == "supported_federal" and not has_federal_anchor
                    and not any(m in folded for m in _FEDERAL_MARKERS)):
                raw["user_query"] = (
                    "Une banque située à Montréal refuse une opération. "
                    "Quelle règle fédérale pourrait s'appliquer?")
                raw["facts_provided"] = {}
        elif rt.name == "topic_research":
            ccq_bad = ("prestation", "ccq", "c.c.q",
                       "commission de la construction")
            if any(m in folded for m in ccq_bad):
                raw["user_query"] = FIXTURE_QUERIES["topic_research"]
                raw["facts_provided"] = {}
            elif any(m in folded for m in _PENAL_MARKERS):
                raw["user_query"] = FIXTURE_QUERIES["topic_research"]
                raw["facts_provided"] = {}
                raw["clarification_answer"] = None
        elif rt.name == "procedure_guidance":
            if any(m in folded for m in _PENAL_MARKERS):
                raw["user_query"] = FIXTURE_QUERIES["procedure_guidance"]
                raw["facts_provided"] = {}
        elif rt.name == "law_or_regulation_identification":
            jurisdiction = raw.get("jurisdiction_status", "undetermined")
            if (jurisdiction == "supported_federal" and not has_federal_anchor
                    and not re.search(r"\bloi\s+(?:sur|de|concernant)\b", folded)):
                raw["user_query"] = (
                    "Trouvez le texte officiel de la Loi sur les banques "
                    "du Canada.")
                raw["facts_provided"] = {}
        elif rt.name == "non_legal":
            if any(m in folded for m in _LEGAL_MARKERS):
                raw["user_query"] = FIXTURE_QUERIES["non_legal"]
            raw["facts_provided"] = {}

    @staticmethod
    def _enforce_jurisdiction(raw: dict) -> None:
        """Ensure jurisdiction_status matches the actual query content."""
        query = str(raw.get("user_query", "")).casefold()
        declared = raw.get("jurisdiction_status", "undetermined")

        ccq_markers = ("code civil", "ccq", "c.c.q", "article", "québ",
                       "légisquébec", "code de procédure civile", "cpc")
        federal_markers = ("fédéral", "canada", "lrc", "failli", "insolv",
                           "brevet", "marque de commerce", "maritime",
                           "banque", "bancaire", "code criminel")

        has_quebec = any(m in query for m in ccq_markers)
        has_federal = any(m in query for m in federal_markers)

        if has_quebec and not has_federal:
            if declared not in ("supported_quebec", "undetermined"):
                raw["jurisdiction_status"] = "supported_quebec"
                raw["expected_jurisdiction"] = "Québec"
        elif has_federal and not has_quebec:
            if declared == "supported_quebec":
                raw["jurisdiction_status"] = "supported_federal"
                raw["expected_jurisdiction"] = "Canada (fédéral)"

        status = raw.get("jurisdiction_status", "undetermined")
        if status == "supported_quebec":
            raw.setdefault("expected_jurisdiction", "Québec")
        elif status == "supported_federal":
            raw.setdefault("expected_jurisdiction", "Canada (fédéral)")

    @staticmethod
    def _enforce_clarification(raw: dict) -> None:
        """Ensure clarification metadata is self-consistent."""
        stage = raw.get("clarification_stage", "none")
        if stage == "none":
            raw["synthetic_clarification_answer"] = None
            raw["clarification_answer"] = None
        elif stage in ("before_search", "after_initial_research"):
            if not raw.get("synthetic_clarification_answer") and not raw.get("clarification_answer"):
                raw["clarification_stage"] = "none"

    def _article_anchor(self, request_type_name: str) -> Optional[int]:
        if request_type_name in {"exact_text_retrieval", "article_explanation"}:
            return self.rng.choice(VERIFIED_CCQ_ARTICLES)
        return None

    def _federal_anchor(self, request_type_name: str,
                        jurisdiction_status: str) -> Optional[str]:
        if not self.anchor_bank:
            return None
        if jurisdiction_status != "supported_federal":
            return None
        if request_type_name in {"case_analysis", "case_law_research"}:
            anchor = self.anchor_bank.sample_case(self.rng)
            if anchor:
                return (f"Décision fédérale imposée : {anchor.name_fr} "
                        f"({anchor.citation_fr}, {anchor.dataset}). "
                        f"Domaine : {anchor.topic_hint}.")
        if request_type_name == "law_or_regulation_identification":
            anchor = self.anchor_bank.sample_law(self.rng)
            if anchor:
                return (f"Loi fédérale imposée : {anchor.name_fr} "
                        f"({anchor.citation_fr}).")
        return None

    @staticmethod
    def _anchored_query(request_type_name: str, query: str,
                        article: int) -> str:
        numbers = ARTICLE_RE.findall(query)
        folded = query.casefold()
        expected_code_named = (
            ("code civil" in folded or "ccq" in folded)
        )
        if numbers == [str(article)] and expected_code_named:
            return query
        if request_type_name == "article_explanation":
            return (f"Que signifie l'article {article} du Code civil du "
                    "Québec dans son texte officiel?")
        return (f"Pouvez-vous me donner le texte officiel de l'article "
                f"{article} du Code civil du Québec?")

    def _fixture(self, rt: RequestTypeSpec,
                 clarification_stage: str,
                 jurisdiction_status: str,
                 failure_mode: Optional[str]) -> ScenarioSpec:
        family = hashlib.sha256(
            f"{rt.name}:{self.index // 3}".encode()).hexdigest()[:12]
        scenario_id = hashlib.sha256(
            f"{self.seed}:{rt.name}:{self.index}".encode()).hexdigest()[:16]

        facts_before_search: list[str] = []
        clarification_answer: Optional[str] = None

        if clarification_stage == "before_search":
            facts_before_search = ["juridiction", "faits essentiels"]
            clarification_answer = (
                "C'est au Québec, mon employeur a réduit mes heures "
                "sans préavis.")
        elif clarification_stage == "after_initial_research":
            clarification_answer = (
                "La situation est au Québec et concerne un contrat civil.")

        raw = {
            "user_query": FIXTURE_QUERIES[rt.name],
            "jurisdiction_status": jurisdiction_status,
            "clarification_stage": clarification_stage,
            "synthetic_clarification_answer": clarification_answer,
            "clarification_answer": clarification_answer,
        }
        self._enforce_jurisdiction(raw)
        self._enforce_clarification(raw)

        return ScenarioSpec(
            scenario_id=scenario_id,
            scenario_family_id=f"{rt.name}-{family}",
            request_type=rt.name,
            language="fr",
            user_query=FIXTURE_QUERIES[rt.name],
            legal_domain=rt.legal_domain,
            jurisdiction_status=raw["jurisdiction_status"],
            clarification_stage=raw["clarification_stage"],
            source_intent=rt.default_source_intents,
            facts_provided={"fixture": True},
            facts_required_before_search=facts_before_search,
            expected_route=rt.expected_route,
            synthetic_clarification_answer=raw.get("synthetic_clarification_answer"),
            clarification_answer=raw.get("clarification_answer"),
            planned_failure_mode=failure_mode,
        )
