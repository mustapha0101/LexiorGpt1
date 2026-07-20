"""Projection bornée des preuves envoyées aux critiques LLM."""

from __future__ import annotations

from typing import Any

from .schemas import ResearchState


MAX_CRITIC_EVIDENCE_CHARS = 8_000
MAX_CRITIC_REFERENCES = 20
RETRIEVAL_ONLY_TOOLS = {"semantic_search_ccq", "semantic_search_cpc"}


def bounded_tool_history(state: ResearchState) -> list[dict[str, Any]]:
    """Exclut toujours ``raw_response`` des prompts de critique.

    La réponse brute reste dans le JSONL d'audit, mais elle peut contenir des
    centaines de milliers de tokens. Les critiques n'ont besoin que du texte
    normalisé réellement montré au Trajectory Agent.
    """
    projected = []
    for observation in state.tool_history:
        retrieval_only = observation.tool_name in RETRIEVAL_ONLY_TOOLS
        projected.append({
            "tool_name": observation.tool_name,
            "server": observation.server,
            "arguments": observation.arguments,
            "normalized_response": (
                observation.normalized_response[:MAX_CRITIC_EVIDENCE_CHARS]
                if not retrieval_only else
                "Liste de candidats pour sélection seulement; non citable."
            ),
            "retrieval_only": retrieval_only,
            "content_hash": observation.content_hash,
            "source_urls": ([] if retrieval_only else
                            observation.source_urls[:MAX_CRITIC_REFERENCES]),
            "citations": ([] if retrieval_only else
                          observation.citations[:MAX_CRITIC_REFERENCES]),
            "truncated": observation.truncated,
            "error": observation.error,
            "ok": observation.ok,
            "latency_ms": observation.latency_ms,
        })
    return projected
