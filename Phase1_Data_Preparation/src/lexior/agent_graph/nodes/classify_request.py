# -*- coding: utf-8 -*-
"""classify_request — classification sémantique avant toute politique."""

from __future__ import annotations

from typing import Any

from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState
from ._common import requested_output_type

NAME = "classify_request"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    scenario = state["scenario"]
    latest = state.get("latest_user_message", "")

    if is_live(state.get("mode", "")):
        classification = ctx.services.request_classifier.classify(
            latest,
            state.get("messages", []),
            state.get("active_issue", ""),
        )
    else:
        classification = ctx.services.request_classifier.from_scenario(
            scenario.request_type, scenario.legal_domain)

    return {
        "request_classification": classification.model_dump(mode="json"),
        "request_intent": classification.intent.value,
        "request_type": classification.request_type,
        "legal_domain": classification.legal_domain,
        "jurisdiction_material": classification.jurisdiction_material,
        "employment_regime_material": classification.employment_regime_material,
        "classification_confidence": classification.confidence,
        "requested_output_type": requested_output_type(latest),
    }
