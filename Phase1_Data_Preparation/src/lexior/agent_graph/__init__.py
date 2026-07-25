# -*- coding: utf-8 -*-
"""Le système d'agents central de Lexior, fondé sur LangGraph.

UN graphe (``build_graph``), UN état (``LexiorState``), UN exécuteur
(``GraphRunner``) — deux modes : ``dataset`` et ``live``.

Les symboles qui dépendent de ``langgraph`` sont résolus à la demande :
importer un module de logique pure (``result_classifier``, ``state``,
``routing``) ne doit pas exiger que langgraph soit installé, sinon la
collecte des tests échoue dans un environnement minimal.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .state import (
    LexiorState,
    initial_state,
    to_quality_report,
    to_research_state,
    to_trajectory,
)
from .step_verifier import ProposalVerdict, StepVerifier, VerifiedProposal
from .result_classifier import ResultClassifier

# Symbole public -> module qui le porte. Chargé au premier accès.
_LAZY: dict[str, str] = {
    "GraphContext": ".context",
    "build_context": ".context",
    "build_graph": ".graph",
    "build_default_graph": ".graph",
    "GraphRunner": ".runner",
    "DatasetRunResult": ".runner",
    "LiveTurnResult": ".runner",
    "NODE_LABELS": ".events",
    "StreamTranslator": ".events",
    "create_memory_checkpointer": ".checkpointing",
    "create_sqlite_checkpointer": ".checkpointing",
    "NODE_NAMES": ".nodes",
}

if TYPE_CHECKING:  # pragma: no cover - aide les analyseurs statiques
    from .checkpointing import (
        create_memory_checkpointer,
        create_sqlite_checkpointer,
    )
    from .context import GraphContext, build_context
    from .events import NODE_LABELS, StreamTranslator
    from .graph import build_default_graph, build_graph
    from .nodes import NODE_NAMES
    from .runner import DatasetRunResult, GraphRunner, LiveTurnResult


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} n'expose pas {name!r}")
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


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
