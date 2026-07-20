# -*- coding: utf-8 -*-

"""
Pipeline de génération de données agentiques multi-agent pour LexiorGPT.

Remplace la génération juridique one-shot par une machine à états contrôlée :

    Scenario Generator -> Planner -> MCP Executor -> Trajectory Agent
    -> Legal Critic -> Agentic Critic -> Validateur déterministe -> Stockage

Le dataset produit apprend au modèle une POLITIQUE (reconnaître la demande,
choisir la juridiction, décider d'une clarification, appeler le bon outil MCP
avec des arguments valides, s'arrêter au bon moment, répondre à partir des
sources réellement récupérées) — pas la récitation des lois depuis ses poids.

Point d'entrée : python -m agentic_generation.cli --help
"""

SCHEMA_VERSION = "agentic-1.0"
DATASET_TYPE = "agentic_legal"
