# -*- coding: utf-8 -*-

"""Enregistrement des sessions de chat — jeu de test, jamais du corpus.

``return_live_answer`` construisait déjà la trajectoire complète puis la
jetait (``export_result: None``). On l'écrit ici, en JSONL, une ligne par
question.

Trois garanties :

* **désactivé par défaut** — ``chat_sessions_dir`` vide, donc rien ne peut
  s'activer tout seul en production ;
* **hors du corpus** — répertoire distinct de ``data/agentic/{accepted,
  rejected}``. Ces sessions ne sont pas des données d'entraînement ;
* **les rejets aussi** — ``route_after_acceptance`` envoie tout rejet vers le
  nœud ``reject``, y compris en live. Brancher le seul ``return_live_answer``
  aurait perdu exactement les échecs qu'on veut lire.

Le texte est intégral, jamais tronqué : découvrir après coup qu'une réponse
était coupée au moment de l'analyser serait pire que le poids du fichier.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional


def _horodatage() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _statuts_par_index(state: Any) -> dict[int, dict[str, str]]:
    """Classification de chaque résultat d'outil, par rang d'appel.

    ``search_evaluations`` porte déjà ``tool_call_index``, ``result_status``
    et ``result_reason`` — un élément par observation. Rien à recalculer :
    on lit le verdict qui a réellement gouverné le routage.
    """
    statuts: dict[int, dict[str, str]] = {}
    for evaluation in (state.get("search_evaluations") or []):
        index = getattr(evaluation, "tool_call_index", None)
        if index is None and isinstance(evaluation, dict):
            index = evaluation.get("tool_call_index")
        if index is None:
            continue
        lire = (evaluation.get if isinstance(evaluation, dict)
                else lambda k, d="": getattr(evaluation, k, d))
        statuts[int(index)] = {
            "classification": str(lire("result_status", "") or ""),
            "motif": str(lire("result_reason", "") or ""),
        }
    return statuts


def _preuves(state: Any) -> list[dict[str, Any]]:
    entrees: list[dict[str, Any]] = []
    for niveau in ("usable_evidence_entries", "citable_sources",
                   "candidate_sources", "alternative_sources"):
        for entree in (state.get(niveau) or []):
            if not isinstance(entree, dict):
                continue
            entrees.append({
                "niveau": niveau.replace("_entries", "").replace("_sources", ""),
                "outil": entree.get("tool_name", ""),
                "index_appel": entree.get("tool_history_index"),
                "statut_detaille": entree.get("detailed_status", ""),
                "articles": entree.get("articles", []),
                "citations": entree.get("citations", []),
                "urls": entree.get("source_urls", []),
                "content_hash": entree.get("content_hash", ""),
            })
    return entrees


def construire_entree(state: Any, statut: str) -> dict[str, Any]:
    """Une entrée JSONL depuis l'état du graphe. Aucun texte tronqué."""
    statuts = _statuts_par_index(state)
    outils = []
    for rang, obs in enumerate(state.get("tool_history") or []):
        classification = statuts.get(rang, {})
        outils.append({
            "rang": rang + 1,
            "nom": getattr(obs, "tool_name", ""),
            "serveur": getattr(obs, "server", ""),
            "arguments": getattr(obs, "arguments", {}) or {},
            "ok": bool(getattr(obs, "ok", False)),
            "erreur": getattr(obs, "error", None),
            "classification": classification.get("classification", ""),
            "motif_classification": classification.get("motif", ""),
            "latence_ms": getattr(obs, "latency_ms", None),
            "content_hash": getattr(obs, "content_hash", "") or "",
            "urls": list(getattr(obs, "source_urls", []) or []),
            "citations": list(getattr(obs, "citations", []) or []),
            "reponse": getattr(obs, "normalized_response", "") or "",
        })

    acceptation = state.get("acceptance_result")
    def _champ(nom, defaut):
        if acceptation is None:
            return defaut
        if isinstance(acceptation, dict):
            return acceptation.get(nom, defaut)
        return getattr(acceptation, nom, defaut)

    decision = state.get("latest_decision") or {}
    return {
        "enregistre_le": _horodatage(),
        "session_id": state.get("thread_id", ""),
        "statut": statut,
        # active_issue et non latest_user_message : après une clarification,
        # ce dernier vaut la réponse (« Au Québec. ») et la question écrite
        # serait perdue. Les deux sont conservés.
        "question": (state.get("active_issue")
                     or state.get("latest_user_message", "")),
        "dernier_message": state.get("latest_user_message", ""),
        "reponse_de_clarification": state.get("clarification_answer", ""),
        "request_type": state.get("request_type", ""),
        "juridiction": {
            "retenue": state.get("resolved_jurisdiction", ""),
            "statut": state.get("jurisdiction_status", ""),
            "fondement": state.get("jurisdiction_basis", ""),
        },
        "raisonnement_final": (decision.get("thinking_text", "")
                               if isinstance(decision, dict) else ""),
        "outils": outils,
        "preuves": _preuves(state),
        "reponse_finale": state.get("final_answer", ""),
        "acceptation": {
            "accepte": bool(_champ("accepted", False)),
            "erreurs": list(_champ("blocking_errors", []) or []),
            "avertissements": list(_champ("warnings", []) or []),
            "motifs": list(_champ("reasons", []) or []),
            "controles_echoues": list(_champ("failed_checks", []) or []),
            "raison_arret": state.get("stop_reason", ""),
            "problemes_validation": list(state.get("validation_issues") or []),
        },
    }


def enregistrer_tour(state: Any, config: Any, statut: str) -> Optional[str]:
    """Écrit une ligne JSONL. Retourne le fichier, ou ``None`` si désactivé."""
    dossier = getattr(config, "chat_sessions_dir", "") or ""
    if not dossier:
        return None                      # désactivé : le défaut
    os.makedirs(dossier, exist_ok=True)
    session = str(state.get("thread_id", "") or "sans-session")
    sur = "".join(c if c.isalnum() or c in "-_" else "-" for c in session)[:64]
    fichier = os.path.join(dossier, f"session-{sur}.jsonl")
    entree = construire_entree(state, statut)
    with open(fichier, "a", encoding="utf-8") as flux:
        flux.write(json.dumps(entree, ensure_ascii=False) + "\n")
    return fichier
