# -*- coding: utf-8 -*-
"""Service planner — unique point d'entrée de planification pour le graphe.

Un seul service pour les deux modes. La différence dataset/live n'est pas
un moteur différent : c'est la présence (dataset) ou l'absence (live)
d'une route scriptée dans le scénario, qui active ou non les gardes
scriptés du ``PlannerAgent`` sous-jacent.
"""

from __future__ import annotations

from typing import Optional

from lexior.agentic.planner_agent import PlannerAgent
from lexior.agentic.schemas import PlannerDecision, ResearchState
from lexior.agentic.tool_catalog import ToolCatalog

from .modes import LIVE, normalize_mode


class PlannerService:
    """Propose la prochaine action. Ne décide JAMAIS du routage global.

    Le nœud ``plan`` appelle :meth:`propose`; le nœud ``validate_plan``
    (déterministe) autorise, modifie ou rejette; les arêtes
    conditionnelles de LangGraph choisissent le nœud suivant.
    """

    def __init__(self, catalog: ToolCatalog, client=None,
                 offline: bool = False):
        self.catalog = catalog
        self.client = client
        self.offline = offline
        # Même classe, même client, même catalogue — seule la présence
        # d'une route scriptée change (chat_mode).
        self._agents = {
            "dataset": PlannerAgent(catalog, client=client, offline=offline,
                                    chat_mode=False),
            LIVE: PlannerAgent(catalog, client=client, offline=offline,
                               chat_mode=True),
        }

    @classmethod
    def from_agent(cls, agent, catalog: ToolCatalog = None) -> "PlannerService":
        """Service au-dessus d'un agent DÉJÀ construit (ou injecté).

        Utilisé par la façade de compatibilité ``AgenticOrchestrator``
        et par les tests qui injectent un planner factice — l'objet n'a
        besoin que d'une méthode ``decide(state)``.
        """
        service = cls.__new__(cls)
        service.catalog = catalog or getattr(agent, "catalog", None)
        service.client = getattr(agent, "client", None)
        service.offline = bool(getattr(agent, "offline", False))
        service._agents = {"dataset": agent, LIVE: agent}
        return service

    def agent_for(self, mode: str) -> PlannerAgent:
        return self._agents[normalize_mode(mode)]

    def propose(self, state: ResearchState, mode: str,
                feedback: Optional[str] = None) -> PlannerDecision:
        """Une proposition d'action pour l'état courant.

        ``feedback`` : message correctif (reformulation, résultat
        inutilisable, réparation de trajectoire) injecté dans la
        conversation du planner via ``state.stop_reason`` transitoire —
        le PlannerAgent en ligne le reçoit tel quel dans son prompt.
        """
        agent = self.agent_for(mode)
        if feedback and not self.offline:
            # Le PlannerAgent en ligne relit tout l'historique; le
            # feedback est ajouté comme message utilisateur transitoire.
            from lexior.agentic.schemas import Message, Role
            state = state.model_copy(deep=True)
            state.messages.append(Message(
                role=Role.user,
                content=f"[contrôle qualité] {feedback}",
            ))
        return agent.decide(state)
