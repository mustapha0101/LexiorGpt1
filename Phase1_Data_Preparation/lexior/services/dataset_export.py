# -*- coding: utf-8 -*-
"""Service d'export dataset — persistance JSONL intermédiaire.

L'export ChatML reste un pas déterministe SÉPARÉ
(``agentic_generation.training_formatter``) : ce service n'écrit que le
format intermédiaire « agentic-2.0 » (accepted/rejected), inchangé.
"""

from __future__ import annotations

from typing import Optional

from agentic_generation.schemas import RejectionRecord, TrainingTrajectory
from agentic_generation.storage import RunStorage


class DatasetExportService:
    def __init__(self, storage: Optional[RunStorage] = None):
        self.storage = storage

    def export_accepted(self, trajectory: TrainingTrajectory) -> bool:
        if self.storage is None:
            return False
        self.storage.append_accepted(trajectory)
        return True

    def export_rejected(self, rejection: RejectionRecord) -> bool:
        if self.storage is None:
            return False
        self.storage.append_rejected(rejection)
        return True
