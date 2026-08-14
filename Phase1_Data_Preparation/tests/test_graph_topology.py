# -*- coding: utf-8 -*-
"""Topologie du graphe compilé — aucun appel réseau ni modèle.

Le graphe est construit pour de vrai (services injectés, exécuteur d'outils
inerte). Ces contrôles auraient attrapé le nœud « reject » qui bouclait sur
lui-même et le plafond de récursion inférieur aux budgets configurés.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lexior.agent_graph import build_context
from lexior.agent_graph.checkpointing import create_memory_checkpointer
from lexior.agent_graph.graph import build_graph
from lexior.agent_graph.runner import GraphRunner
from lexior.agentic.config import load_config
from lexior.agentic.tool_catalog import load_catalog
from lexior.services import build_services
from lexior.services.container import LexiorServices

PHASE1 = Path(__file__).resolve().parent.parent
REPO = PHASE1.parent


class _ExecuteurInerte:
    def execute(self, call):  # pragma: no cover - ne doit jamais servir
        raise AssertionError("aucun outil ne doit être appelé ici")


@pytest.fixture(scope="module")
def contexte():
    cfg = load_config(str(PHASE1 / "configs" / "agentic_generation.yaml"))
    catalog = load_catalog(str(REPO / "docs" / "mcp_tools_catalog.json"))
    services = build_services(cfg, catalog, executor=_ExecuteurInerte(),
                              teacher=None, critic_client=None)
    return build_context(cfg, catalog, services)


@pytest.fixture(scope="module")
def aretes(contexte):
    graphe = build_graph(contexte, checkpointer=create_memory_checkpointer())
    rendu = graphe.get_graph()
    return ({n for n in rendu.nodes},
            [(e.source, e.target) for e in rendu.edges])


def _successeurs(aretes):
    adjacence: dict[str, set[str]] = {}
    for source, cible in aretes:
        adjacence.setdefault(source, set()).add(cible)
    return adjacence


def test_tous_les_noeuds_atteignables(aretes):
    noeuds, liens = aretes
    adjacence = _successeurs(liens)
    vus, pile = set(), ["__start__"]
    while pile:
        courant = pile.pop()
        if courant in vus:
            continue
        vus.add(courant)
        pile.extend(adjacence.get(courant, ()))
    assert not noeuds - vus, f"inatteignables depuis START : {sorted(noeuds - vus)}"


def test_tous_les_noeuds_atteignent_la_fin(aretes):
    noeuds, liens = aretes
    inverse: dict[str, set[str]] = {}
    for source, cible in liens:
        inverse.setdefault(cible, set()).add(source)
    vus, pile = set(), ["__end__"]
    while pile:
        courant = pile.pop()
        if courant in vus:
            continue
        vus.add(courant)
        pile.extend(inverse.get(courant, ()))
    assert not noeuds - vus, f"n'atteignent jamais END : {sorted(noeuds - vus)}"


def test_reject_ne_boucle_pas_sur_lui_meme(aretes):
    """« reject » est la sortie de secours : s'y renvoyer épuise le budget."""
    _, liens = aretes
    assert ("reject", "reject") not in liens


@pytest.mark.parametrize("noeud", [
    "initialize", "classify_request", "classify_follow_up",
    "update_active_task", "resolve_jurisdiction", "analyze_facts",
])
def test_prefixe_lineaire_garde_par_un_routeur(aretes, noeud):
    """Une arête statique concurrencerait un Command(goto="reject")."""
    _, liens = aretes
    assert "reject" in _successeurs(liens).get(noeud, set())


def test_plafond_de_recursion_couvre_les_budgets(contexte):
    """Un plafond fixe sous les budgets rend ceux-ci inatteignables."""
    runner = GraphRunner(contexte, checkpointer=create_memory_checkpointer())
    config = contexte.config
    cycles = (config.max_tool_calls_live + config.max_repairs
              + config.max_clarifications_live
              + config.max_search_reformulations_live)
    assert runner.recursion_limit > 15 + 9 * cycles


def test_tous_les_services_references_existent():
    """« reject » appelait ctx.services.export, qui n'existe pas."""
    import ast
    import dataclasses

    disponibles = {f.name for f in dataclasses.fields(LexiorServices)}
    references: set[str] = set()
    for chemin in (PHASE1 / "src" / "lexior" / "agent_graph" / "nodes").glob("*.py"):
        arbre = ast.parse(chemin.read_text(encoding="utf-8"))
        for noeud in ast.walk(arbre):
            if (isinstance(noeud, ast.Attribute)
                    and isinstance(noeud.value, ast.Attribute)
                    and noeud.value.attr == "services"):
                references.add(noeud.attr)
    manquants = sorted(references - disponibles)
    assert not manquants, f"services référencés mais absents : {manquants}"
