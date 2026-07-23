# -*- coding: utf-8 -*-
"""initialize — normalisation du tour et métadonnées de génération."""

from __future__ import annotations

from typing import Any

from agentic_generation.schemas import GenerationMetadata, StateStatus
from lexior.services.modes import normalize_mode

from ..context import GraphContext
from ..state import LexiorState
from ._common import last_user_content

NAME = "initialize"


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    config = ctx.config
    return {
        "mode": normalize_mode(state.get("mode", "dataset")),
        "status": StateStatus.planning.value,
        "stop_reason": "",
        "latest_user_message": last_user_content(state.get("messages", [])),
        "generation_metadata": GenerationMetadata(
            teacher_model=config.teacher.model,
            teacher_base_url_hash=config.teacher.base_url_hash,
            critic_model=config.critic.model,
            seed=config.seed,
            prompt_version=config.prompt_version,
            tool_catalog_hash=ctx.catalog.catalog_hash,
        ),
    }
