# -*- coding: utf-8 -*-
"""verify_tool_result — succès technique et normalisation.

Vérifie la DERNIÈRE observation (stubs d'articles abrogés, réponses
vides déguisées, texte tronqué), remplace l'observation par sa version
vérifiée et ajoute le message outil à la conversation — le contenu du
message est donc toujours la réponse vérifiée.
"""

from __future__ import annotations

from typing import Any

from lexior.agentic.schemas import Message, Role

from ..context import GraphContext
from ..state import LexiorState

NAME = "verify_tool_result"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    tool_history = list(state.get("tool_history", []))
    if not tool_history:
        return {}

    observation, issues = ctx.services.verification.verify(tool_history[-1])
    tool_history[-1] = observation

    messages = list(state.get("messages", []))
    messages.append(Message(
        role=Role.tool, name=observation.tool_name,
        content=observation.normalized_response,
    ))

    return {
        "tool_history": tool_history,
        "messages": messages,
        "sources": list(state.get("sources", []))
        + list(observation.source_urls),
        "last_tool_assessment": {
            "tool_name": observation.tool_name,
            "tool_call_succeeded": observation.ok,
            "verifier_issues": list(issues),
        },
    }
