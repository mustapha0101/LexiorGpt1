# -*- coding: utf-8 -*-
"""Lexior — multi-agent legal assistant framework.

Le ``.env`` du dépôt est chargé ici, à l'import du paquet : tout module
qui lit ``os.environ`` en bénéficie sans avoir à s'en occuper. Les
variables déjà présentes dans l'environnement restent prioritaires.
"""

from .env import load_project_env

load_project_env()

__all__ = ["load_project_env"]
