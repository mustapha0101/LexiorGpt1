# -*- coding: utf-8 -*-
"""Service de juridiction — résolution déterministe, unique pour les deux modes.

Source canonique de :
  - la détection de province dans une conversation (regex, pas de LLM);
  - la liste des outils exclusivement québécois (``QC_ONLY_TOOLS``);
  - la sémantique de verrouillage (une juridiction établie par un signal
    explicite de l'utilisateur ne change jamais silencieusement).

``agentic_generation.planner_agent`` délègue ici; il n'existe qu'une seule
implémentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Sequence

# ── Outils dont les sources sont exclusivement québécoises ───────────────

QC_ONLY_TOOLS = {
    "semantic_search_ccq", "semantic_search_cpc",
    "get_ccq_articles", "get_cpc_articles",
    "search_ccq_keywords", "search_cpc_keywords",
    "search_quebec_regulations", "get_quebec_regulation",
    "get_quebec_legal_info", "search_quebec_jurisprudence",
}

OUTSIDE_QUEBEC = "hors Québec (province non précisée)"


class QuebecToolsBlocked(ValueError):
    """Outil québécois choisi alors que l'utilisateur n'est pas au Québec."""


# ── Signaux déterministes ────────────────────────────────────────────────

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


def detect_jurisdiction_hint(messages: Sequence) -> Optional[str]:
    """Juridiction déduite DÉTERMINISTIQUEMENT de la conversation.

    Parcourt les messages dans l'ordre; le signal le plus récent l'emporte.
    Retourne « Québec », un nom de province, ou
    « hors Québec (province non précisée) » — ``None`` si rien ne tranche.

    ``messages`` : séquence d'objets avec ``role`` (Role ou str) et
    ``content`` (str) — le schéma ``Message`` du pipeline convient.
    """
    hint: Optional[str] = None
    for index, message in enumerate(messages):
        role = getattr(message.role, "value", message.role)
        if role != "user":
            continue
        province = _PROVINCE_RE.search(message.content)
        if province:
            hint = province.group(0).title()
            continue
        if _QC_MENTION_RE.search(message.content):
            hint = "Québec"
            continue
        if index > 0:
            previous = messages[index - 1]
            prev_role = getattr(previous.role, "value", previous.role)
            previous_is_quebec_question = (
                prev_role == "assistant"
                and "québec" in previous.content.lower()
                and previous.content.rstrip().endswith("?")
            )
            if previous_is_quebec_question:
                if _YES_RE.match(message.content):
                    hint = "Québec"
                elif _NO_RE.match(message.content):
                    hint = OUTSIDE_QUEBEC
    return hint


def is_quebec(value: str) -> bool:
    return bool(value) and value.strip().casefold() in ("québec", "quebec")


# ── Couverture : quatre cas, quatre comportements ────────────────────────

CANADIAN_PROVINCES = frozenset({
    "ontario", "alberta", "manitoba", "saskatchewan",
    "colombie-britannique", "colombie britannique", "british columbia",
    "nouvelle-écosse", "nouvelle écosse", "nova scotia",
    "nouveau-brunswick", "nouveau brunswick", "new brunswick",
    "terre-neuve", "terre-neuve-et-labrador", "newfoundland",
    "île-du-prince-édouard", "ile-du-prince-edouard", "prince edward island",
    "yukon", "nunavut", "territoires du nord-ouest", "northwest territories",
})

_FEDERAL_VALUES = frozenset({"federal", "fédéral", "canada", "canadien"})


class JurisdictionCoverage(str, Enum):
    """Ce que le système peut traiter, et ce qu'il doit refuser."""

    supported_quebec = "supported_quebec"
    supported_federal = "supported_federal"
    unsupported_other_canadian = "unsupported_other_canadian"
    unsupported_foreign = "unsupported_foreign"
    unknown = "unknown"


# Comportement associé — c'est ce qui manquait : les catégories existaient
# dans le YAML, pas la conduite à tenir.
COVERAGE_ACTIONS: dict[JurisdictionCoverage, str] = {
    JurisdictionCoverage.supported_quebec: "proceed",
    JurisdictionCoverage.supported_federal: "proceed",
    JurisdictionCoverage.unsupported_other_canadian: "decline",
    JurisdictionCoverage.unsupported_foreign: "decline",
    # Inconnu : on demande, on ne devine pas.
    JurisdictionCoverage.unknown: "clarify",
}


def is_federal(value: str) -> bool:
    return (value or "").strip().casefold() in _FEDERAL_VALUES


def classify_coverage(value: str,
                      federal_matter: bool = False) -> JurisdictionCoverage:
    """Couverture d'une juridiction, en tenant compte du droit applicable.

    Exception explicite : une personne hors Québec dont le cas relève du
    droit fédéral EST couverte — la localisation ne décide pas seule.
    """
    folded = (value or "").strip().casefold()
    if not folded:
        return JurisdictionCoverage.unknown
    if is_quebec(folded):
        return JurisdictionCoverage.supported_quebec
    if folded in _FEDERAL_VALUES:
        return JurisdictionCoverage.supported_federal
    if federal_matter:
        return JurisdictionCoverage.supported_federal
    if folded in CANADIAN_PROVINCES or folded == OUTSIDE_QUEBEC.casefold():
        return JurisdictionCoverage.unsupported_other_canadian
    return JurisdictionCoverage.unsupported_foreign


def coverage_action(value: str, federal_matter: bool = False) -> str:
    """``proceed``, ``decline`` ou ``clarify``."""
    return COVERAGE_ACTIONS[classify_coverage(value, federal_matter)]


def allows_quebec_tools(value: str) -> bool:
    """Les outils QC sont permis quand la juridiction est Québec ou inconnue."""
    return not value or is_quebec(value)


# ── Résolution avec verrouillage ─────────────────────────────────────────


@dataclass(frozen=True)
class JurisdictionResolution:
    """Résultat de la résolution — un seul objet de vérité pour l'état."""

    value: str = ""
    basis: str = ""          # explicit_user_statement | scenario | planner
    locked: bool = False
    verified: bool = False

    @property
    def status(self) -> str:
        """Valeur pour ``jurisdiction_status`` / ``juridiction_etablie``."""
        return self.value or "unknown"


class JurisdictionService:
    """Résolution partagée par les deux modes.

    - live : détection déterministe sur la conversation complète; un signal
      explicite verrouille la valeur. Une valeur verrouillée ne change que
      sur un NOUVEAU signal explicite (jamais silencieusement).
    - dataset : la juridiction vient du scénario puis des décisions du
      planner (comportement historique de l'orchestrateur).
    """

    def resolve_live(
        self,
        messages: Sequence,
        previous: Optional[JurisdictionResolution] = None,
    ) -> JurisdictionResolution:
        previous = previous or JurisdictionResolution()
        hint = detect_jurisdiction_hint(messages)
        if hint:
            if previous.locked and hint == previous.value:
                return previous
            return JurisdictionResolution(
                value=hint, basis="explicit_user_statement",
                locked=True, verified=True,
            )
        if previous.locked:
            return previous
        return previous

    def resolve_dataset(
        self,
        scenario,
        planner_jurisdiction: str = "",
        previous: Optional[JurisdictionResolution] = None,
    ) -> JurisdictionResolution:
        previous = previous or JurisdictionResolution()
        if planner_jurisdiction:
            if previous.locked and planner_jurisdiction != previous.value:
                # Une juridiction verrouillée ne change pas sur simple
                # inférence du planner.
                return previous
            return JurisdictionResolution(
                value=planner_jurisdiction, basis="planner",
                locked=previous.locked, verified=previous.verified,
            )
        if previous.value:
            return previous
        seed = (getattr(scenario, "jurisdiction", "")
                or getattr(scenario, "expected_jurisdiction", ""))
        if seed:
            return JurisdictionResolution(value=seed, basis="scenario")
        return previous

    @staticmethod
    def enforce_locked(
        resolution: JurisdictionResolution, proposed: str,
    ) -> str:
        """Valeur autoritaire pour une proposition du planner.

        Si la juridiction est verrouillée, la proposition est écrasée —
        c'est le mécanisme qui empêche tout changement silencieux.
        """
        if resolution.locked and resolution.value:
            return resolution.value
        return proposed or resolution.value

    @staticmethod
    def lock(resolution: JurisdictionResolution) -> JurisdictionResolution:
        return replace(resolution, locked=True)
