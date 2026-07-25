# -*- coding: utf-8 -*-
"""Le catalogue du chat suit la couverture déclarée, sans liste parallèle.

``api/app.py`` retirait ``search_quebec_jurisprudence`` par une exclusion
codée en dur. Elle a survécu à la réactivation de l'outil : la couverture
l'annonçait disponible, l'app de chat ne l'exposait pas. Deux sources de
vérité, une seule mise à jour — le même défaut que les cinq listes de
tribunaux.
"""

from __future__ import annotations

import pytest

from lexior.services.tool_coverage import TOOL_COVERAGE

app = pytest.importorskip(
    "lexior.api.app", reason="dépendances de l'API absentes")


def test_the_api_module_imports_at_all():
    """Le module lisait ses chemins depuis `src/` depuis la migration.

    Conséquences : catalogue introuvable, et load_config retombant en
    silence sur ses défauts — donc une app de chat servant une
    configuration qui n'était pas la sienne.
    """
    assert app._PHASE1.name == "Phase1_Data_Preparation"
    assert (app._PHASE1 / "configs" / "agentic_generation.yaml").is_file()
    assert app._CATALOG.tools, "le catalogue doit être chargé"


def test_the_chat_config_is_the_real_one():
    """Un chargement silencieusement vide donnerait les valeurs par défaut."""
    assert app._CFG.rag.min_dense_score == pytest.approx(0.40), (
        "la configuration du chat ne vient pas du YAML de production")


def test_the_chat_exposes_every_available_tool():
    exposed = set(app._CHAT_CATALOG.tools)
    expected = {
        name for name in app._CATALOG.tools
        if (entry := TOOL_COVERAGE.get(name)) is None
        or entry.is_available("live")
    }

    assert exposed == expected


def test_the_quebec_jurisprudence_tool_is_reachable_from_the_chat():
    """Le reliquat : réactivé dans tool_coverage, absent du chat."""
    assert TOOL_COVERAGE["search_quebec_jurisprudence"].is_available("live")
    assert "search_quebec_jurisprudence" in app._CHAT_CATALOG.tools


def test_no_hardcoded_exclusion_remains():
    """Aucun nom d'outil ne doit être filtré nominativement."""
    source = (app.__file__ and open(app.__file__, encoding="utf-8").read())

    for name in TOOL_COVERAGE:
        assert f'!= "{name}"' not in source, (
            f"{name} est exclu nominativement — dériver de tool_coverage")
