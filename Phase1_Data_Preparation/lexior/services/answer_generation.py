# -*- coding: utf-8 -*-
"""Service de rédaction — réponse finale strictement fondée.

Enveloppe ``TrajectoryAgent.final_answer`` (une implémentation, deux
configurations : dataset sans supplément chat, live avec).
"""

from __future__ import annotations

from typing import Optional

from agentic_generation.schemas import ResearchState
from agentic_generation.trajectory_agent import (
    PRECISE_ARTICLE_TYPES,
    TrajectoryAgent,
)

from .modes import LIVE, normalize_mode


def _accepts_contract(fn) -> bool:
    import inspect
    try:
        return "contract" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class AnswerGenerationService:
    def __init__(self, client=None, offline: bool = False):
        self.client = client
        self.offline = offline
        self._writers = {
            "dataset": TrajectoryAgent(client=client, offline=offline,
                                       chat_mode=False),
            LIVE: TrajectoryAgent(client=client, offline=offline,
                                  chat_mode=True),
        }

    @classmethod
    def from_writer(cls, writer) -> "AnswerGenerationService":
        """Service au-dessus d'un rédacteur déjà construit (ou injecté)."""
        service = cls.__new__(cls)
        service.client = getattr(writer, "client", None)
        service.offline = bool(getattr(writer, "offline", False))
        service._writers = {"dataset": writer, LIVE: writer}
        return service

    def writer_for(self, mode: str) -> TrajectoryAgent:
        return self._writers[normalize_mode(mode)]

    def generate(self, state: ResearchState, mode: str,
                 contract: Optional[dict] = None) -> tuple[str, str]:
        """(raisonnement, réponse) pour le tour final."""
        writer = self.writer_for(mode)
        if _accepts_contract(writer.final_answer):
            return writer.final_answer(state, contract=contract)
        # Rédacteur injecté d'une génération antérieure, sans ``contract``.
        return writer.final_answer(state)


PRECISE_TYPES = PRECISE_ARTICLE_TYPES
