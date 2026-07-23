# -*- coding: utf-8 -*-

"""
Client Teacher provider-agnostic.

Le SDK Python OpenAI n'est utilisé que comme client d'une API HTTP compatible
OpenAI : lorsque TEACHER_BASE_URL pointe vers un vLLM RunPod, AUCUNE requête
ne part vers OpenAI et aucune vraie clé OpenAI n'est nécessaire.

Sécurité :
  - allow_remote_calls=False (défaut) : tout appel réseau qui n'est pas servi
    par le cache lève RemoteCallsDisabled. Aucun appel payant pendant les
    tests ou le dry-run.
  - aucun secret n'apparaît dans les erreurs ni dans les journaux.

Coûts : un compteur par rôle logique (scenario_generator, planner,
trajectory_writer, legal_critic, agentic_critic, repair). Modèle auto-hébergé
=> coût nul, mais les jetons sont toujours comptés (api_cost.CostTracker).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Callable, Optional

from .config import EndpointConfig

from lexior.observability.costs import CostTracker

COST_ROLES = ("scenario_generator", "planner", "trajectory_writer",
              "legal_critic", "agentic_critic", "retrieval_reranker", "repair")


class TeacherError(Exception):
    """Échec définitif d'un appel Teacher (après épuisement des retries)."""


class RemoteCallsDisabled(TeacherError):
    """Appel réseau refusé : --allow-remote-calls n'a pas été fourni."""


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
_SENSITIVE_RE = re.compile(
    r"(?i)(?:https?://[^\s]+|bearer\s+[^\s,;]+|api[_-]?key[=:]\s*[^\s,;]+)")


def _safe_error(error: Optional[Exception]) -> str:
    return _SENSITIVE_RE.sub("[redacted]", str(error))[:500]


def parse_json_object(text: str) -> dict[str, Any]:
    """Extrait le premier objet JSON d'une sortie de modèle.

    Tolère les clôtures markdown et le texte parasite autour, mais n'invente
    rien : si aucun objet valide n'est présent, l'appel échoue.
    """
    cleaned = _FENCE_RE.sub("", text or "").strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"aucun objet JSON dans la sortie : {cleaned[:200]!r}")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:i + 1])
    raise ValueError(f"objet JSON non fermé dans la sortie : {cleaned[:200]!r}")


class TeacherClient:
    """Client d'une API compatible OpenAI, avec cache, retries et coûts."""

    def __init__(self,
                 endpoint: EndpointConfig,
                 allow_remote_calls: bool = False,
                 cache=None,
                 cache_extra_key: str = "",
                 sleeper: Callable[[float], None] = time.sleep,
                 client_factory: Optional[Callable[[], Any]] = None):
        self.endpoint = endpoint
        self.allow_remote_calls = allow_remote_calls
        self.cache = cache
        # Clé de cache : hash(model + prompt_version + messages +
        # tool_catalog_hash). prompt_version et catalog_hash arrivent ici.
        self.cache_extra_key = cache_extra_key
        self.sleeper = sleeper
        self._client = None
        self._client_factory = client_factory
        self.cost: dict[str, CostTracker] = {
            role: CostTracker(endpoint.model) for role in COST_ROLES}

    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import OpenAI
                if not self.endpoint.base_url:
                    raise TeacherError(
                        "TEACHER_BASE_URL manquant (ou OPENAI_BASE_URL en "
                        "rétrocompatibilité).")
                self._client = OpenAI(
                    base_url=self.endpoint.base_url,
                    api_key=self.endpoint.api_key or "not-needed",
                    timeout=self.endpoint.timeout,
                    max_retries=0,  # les retries sont gérés ici, avec comptage
                )
        return self._client

    def _cache_key(self, role: str, messages: list[dict], temperature: float) -> str:
        import hashlib
        blob = json.dumps({
            "model": self.endpoint.model,
            "extra": self.cache_extra_key,
            "messages": messages,
            "temperature": temperature,
        }, ensure_ascii=False, sort_keys=True)
        return "teacher:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------

    def complete(self, role: str, messages: list[dict[str, str]],
                 temperature: Optional[float] = None) -> str:
        """Un appel chat.completions, avec cache et retries.

        `role` est le rôle LOGIQUE de l'appel (comptabilité), pas un rôle de
        message.
        """
        if role not in self.cost:
            raise ValueError(f"rôle de coût inconnu : {role} (valides : {COST_ROLES})")
        temp = self.endpoint.temperature if temperature is None else temperature

        key = None
        if self.cache is not None:
            key = self._cache_key(role, messages, temp)
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        if not self.allow_remote_calls:
            raise RemoteCallsDisabled(
                "Appel Teacher refusé : relancer avec --allow-remote-calls "
                "pour autoriser les appels réseau (aucun appel n'est fait par "
                "défaut).")

        last_error: Optional[Exception] = None
        for attempt in range(self.endpoint.max_retries + 1):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.endpoint.model,
                    messages=messages,
                    temperature=temp,
                )
                self.cost[role].record(response)
                content = response.choices[0].message.content or ""
                if self.cache is not None and key is not None:
                    self.cache.put(key, content)
                return content
            except Exception as e:  # réseau, HTTP, parsing SDK...
                last_error = e
                self.cost[role].record_failure()
                if attempt < self.endpoint.max_retries:
                    self.sleeper(min(2.0 ** attempt, 30.0))
        raise TeacherError(
            f"Appel Teacher en échec après {self.endpoint.max_retries + 1} "
            f"tentatives (rôle {role}) : {type(last_error).__name__}: "
            f"{_safe_error(last_error)}") from last_error

    def complete_json(self, role: str, messages: list[dict[str, str]],
                      temperature: Optional[float] = None,
                      json_retries: int = 1) -> dict[str, Any]:
        """complete() + extraction JSON, avec au plus `json_retries` relances
        explicitement motivées par l'erreur de format."""
        attempt_messages = list(messages)
        last_err: Optional[Exception] = None
        for _ in range(json_retries + 1):
            text = self.complete(role, attempt_messages, temperature)
            try:
                return parse_json_object(text)
            except ValueError as e:
                last_err = e
                attempt_messages = list(messages) + [
                    {"role": "assistant", "content": text[:2000]},
                    {"role": "user", "content":
                        "Sortie invalide : réponds UNIQUEMENT par l'objet JSON "
                        "demandé, sans texte autour."},
                ]
        raise TeacherError(f"Sortie JSON invalide (rôle {role}) : {last_err}")

    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Pour le doctor : GET /v1/models. Appel réseau — soumis au drapeau."""
        if not self.allow_remote_calls:
            raise RemoteCallsDisabled(
                "Vérification du Teacher refusée sans --allow-remote-calls.")
        models = self._get_client().models.list()
        return [m.id for m in models.data]

    def cost_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        total_cost = 0.0
        total_in = total_cached_in = total_out = 0
        total_calls = total_failed_calls = 0
        for role, tracker in self.cost.items():
            snap = tracker.snapshot()
            if snap["calls"] or snap["failed_calls"]:
                report[role] = snap
            total_cost += snap["cost_usd"]
            total_in += snap["tokens_in"]
            total_cached_in += snap.get("tokens_cached_in", 0)
            total_out += snap["tokens_out"]
            total_calls += snap["calls"]
            total_failed_calls += snap["failed_calls"]
        report["total"] = {
            "calls": total_calls,
            "failed_calls": total_failed_calls,
            "tokens_in": total_in,
            "tokens_cached_in": total_cached_in,
            "tokens_out": total_out,
            "cost_usd": round(total_cost, 6),
        }
        return report
