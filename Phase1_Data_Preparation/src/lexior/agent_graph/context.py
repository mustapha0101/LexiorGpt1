# -*- coding: utf-8 -*-
"""GraphContext — dépendances injectées dans les nœuds du graphe.

Le contexte est lié une seule fois à la construction du graphe; les deux
modes partagent le MÊME contexte (donc les mêmes services). Aucune
dépendance dans l'état : l'état reste sérialisable pour le checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentic_generation.config import AgenticConfig
from agentic_generation.tool_catalog import ToolCatalog
from lexior.services import LexiorServices


@dataclass
class GraphContext:
    config: AgenticConfig
    catalog: ToolCatalog
    services: LexiorServices

    @property
    def max_tool_calls(self) -> int:
        return self.config.max_tool_calls

    @property
    def max_repairs(self) -> int:
        return self.config.max_repairs

    @property
    def max_reformulations(self) -> int:
        return getattr(self.config, "max_search_reformulations", 1)


def build_context(config: AgenticConfig, catalog: ToolCatalog,
                  services: Optional[LexiorServices] = None,
                  **service_kwargs) -> GraphContext:
    if services is None:
        from lexior.services import build_services
        services = build_services(config, catalog, **service_kwargs)
    return GraphContext(config=config, catalog=catalog, services=services)
