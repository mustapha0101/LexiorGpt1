# -*- coding: utf-8 -*-

"""Le champ ``condition`` doit être lu — et rester lu.

Onze étapes facultatives portaient une condition écrite que personne ne
consommait, pendant que trois d'entre elles étaient réencodées en dur dans
``_effective_route``. Le test de complétude ci-dessous est ce qui empêche
la situation de se reformer : écrire une condition sans garde correspondante
fait échouer la suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.agentic.taxonomy import REQUEST_TYPES              # noqa: E402
from lexior.agentic.taxonomy_conditions import (               # noqa: E402
    GARDES, GardeContexte, etape_facultative_retenue,
    juridiction_compatible,
)


def _etapes_facultatives():
    for nom, spec in sorted(REQUEST_TYPES.items()):
        for step in spec.expected_route.steps:
            if step.optional:
                yield nom, step


def _ctx(**kw):
    base = dict(request_type="case_analysis",
                jurisdiction_status="supported_quebec",
                user_query="", tool_history=())
    base.update(kw)
    return GardeContexte(**base)


class _Obs:
    def __init__(self, tool_name, ok=True, reponse="du contenu"):
        self.tool_name = tool_name
        self.ok = ok
        self.normalized_response = reponse


# ── Complétude : le garde-fou du mécanisme ───────────────────────────────


@pytest.mark.parametrize(
    "request_type,step",
    list(_etapes_facultatives()),
    ids=[f"{n}:{s.tool}" for n, s in _etapes_facultatives()])
def test_toute_condition_ecrite_est_lue(request_type, step):
    """Une condition écrite sans garde est une consigne que personne n'applique."""
    if not step.condition:
        pytest.skip(f"{request_type}:{step.tool} n'a aucune condition écrite")
    assert step.condition in GARDES, (
        f"{request_type}:{step.tool} porte la condition « {step.condition} » "
        f"qui n'est dans aucune entrée de GARDES. Ajoutez-y une garde, ou "
        f"JUGEMENT si elle n'est pas décidable depuis l'état.")


def test_le_registre_ne_contient_pas_de_condition_morte():
    """L'inverse : une garde qui ne correspond à aucune étape déclarée."""
    ecrites = {s.condition for _, s in _etapes_facultatives() if s.condition}
    orphelines = set(GARDES) - ecrites
    assert not orphelines, f"gardes sans étape correspondante : {orphelines}"


# ── Couche 1 : la juridiction, dérivée de tool_coverage ──────────────────


def test_un_outil_quebecois_est_ecarte_dun_scenario_federal():
    assert not juridiction_compatible("semantic_search_ccq", "supported_federal")
    assert not juridiction_compatible("get_ccq_articles", "supported_federal")


def test_un_outil_federal_est_ecarte_dun_scenario_quebecois():
    assert not juridiction_compatible("fetch_document", "supported_quebec")
    assert not juridiction_compatible("search_legal_documents", "supported_quebec")


def test_les_statuts_inconnus_ne_bloquent_rien():
    """On n'écarte que ce dont on est sûr."""
    for statut in ("municipal_coverage_uncertain", "supported_other_canadian",
                   "unsupported_foreign", ""):
        assert juridiction_compatible("semantic_search_ccq", statut), statut


def test_document_analysis_recoit_la_garde_qui_lui_manquait():
    """C'est le cas qui cassait quand on honorait les étapes sans garde."""
    spec = REQUEST_TYPES["document_analysis"]
    for step in spec.expected_route.steps:
        assert step.optional
        assert not etape_facultative_retenue(
            step.tool, step.condition,
            _ctx(request_type="document_analysis",
                 jurisdiction_status="supported_federal",
                 user_query="l'article 1457 est cité dans mon document")), step.tool


# ── Couche 2 : les prédicats ─────────────────────────────────────────────


def test_les_trois_paires_codees_en_dur_se_comportent_comme_avant():
    """Non-régression : ce que la liste blanche autorisait, les gardes aussi."""
    assert etape_facultative_retenue(
        "semantic_search_ccq", "if Quebec civil law and article unknown",
        _ctx(user_query="mon voisin passe sur mon terrain"))
    assert etape_facultative_retenue(
        "semantic_search_cpc", "if procedural provision unknown",
        _ctx(request_type="procedure_guidance",
             user_query="comment contester un jugement"))
    assert etape_facultative_retenue(
        "fetch_document", "federal document identified",
        _ctx(request_type="comparative_law",
             jurisdiction_status="supported_federal",
             tool_history=(_Obs("search_legal_documents"),)))


def test_la_recherche_semantique_est_inutile_quand_larticle_est_nomme():
    assert not etape_facultative_retenue(
        "semantic_search_ccq", "if Quebec civil law and article unknown",
        _ctx(user_query="explique-moi l'article 1726"))


def test_federal_document_identified_exige_une_recherche_prealable():
    ctx = _ctx(request_type="comparative_law",
               jurisdiction_status="supported_federal")
    assert not etape_facultative_retenue(
        "fetch_document", "federal document identified", ctx)


def test_une_condition_de_jugement_reste_hors_route():
    """JUGEMENT laisse l'étape permise, mais hors de la route déterministe."""
    assert not etape_facultative_retenue(
        "search_quebec_jurisprudence",
        "article contains open-ended notion or facts warrant it",
        _ctx(user_query="explique-moi l'article 1726"))


def test_une_condition_inconnue_ne_leve_pas_dexception():
    assert not etape_facultative_retenue(
        "semantic_search_ccq", "condition jamais vue", _ctx())
