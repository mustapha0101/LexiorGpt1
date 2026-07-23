# -*- coding: utf-8 -*-
"""Le système d'agents central de Lexior, fondé sur LangGraph.

UN graphe (``build_graph``), UN état (``LexiorState``), UN exécuteur
(``GraphRunner``) — deux modes : ``dataset`` et ``live``.
"""

from .state import (
    LexiorState,
    initial_state,
    to_quality_report,
    to_research_state,
    to_trajectory,
)
from .step_verifier import ProposalVerdict, StepVerifier, VerifiedProposal
from .result_classifier import ResultClassifier
from .context import GraphContext, build_context
from .graph import build_graph, build_default_graph
from .runner import DatasetRunResult, GraphRunner, LiveTurnResult
from .events import NODE_LABELS, StreamTranslator
from .checkpointing import (
    create_memory_checkpointer,
    create_sqlite_checkpointer,
)
from .nodes import NODE_NAMES

__all__ = [
    "LexiorState",
    "initial_state",
    "to_quality_report",
    "to_research_state",
    "to_trajectory",
    "StepVerifier",
    "VerifiedProposal",
    "ProposalVerdict",
    "ResultClassifier",
    "GraphContext",
    "build_context",
    "build_graph",
    "build_default_graph",
    "GraphRunner",
    "DatasetRunResult",
    "LiveTurnResult",
    "NODE_LABELS",
    "NODE_NAMES",
    "StreamTranslator",
    "create_memory_checkpointer",
    "create_sqlite_checkpointer",
]
