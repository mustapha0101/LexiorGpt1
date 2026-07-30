# -*- coding: utf-8 -*-

"""Évaluation des conditions d'étape facultative déclarées dans la taxonomie.

``ExpectedRouteStep`` porte un champ ``condition`` depuis l'origine. Onze des
douze étapes facultatives le remplissent — « article number known or
discoverable », « federal document identified »… — et **aucune ligne du dépôt
ne le lisait**. ``_effective_route`` réencodait trois de ces conditions en
Python, sous forme de paires codées en dur, et écartait les huit autres. Deux
variables y étaient même calculées sans jamais servir.

Ce module consomme le champ. Deux couches :

* **Couche 1, la juridiction** — dérivée, écrite nulle part. ``tool_coverage``
  déclare déjà la juridiction de chaque outil. Un outil québécois n'entre
  jamais dans la route déterministe d'un scénario fédéral, et inversement.
  C'est la garde qui manquait à ``document_analysis`` : ses deux étapes
  facultatives sont des outils québécois, appliqués sans égard à la
  juridiction, et honorer les étapes sans cette garde le faisait échouer en
  ``jurisdiction_mismatch``.

* **Couche 2, les prédicats** — un registre indexé par la prose exacte de la
  condition. Une condition absente du registre, ou marquée ``JUGEMENT``,
  laisse l'étape PERMISE mais hors de la route déterministe : c'est le
  comportement actuel, donc rien ne régresse.

``tests/test_taxonomy_conditions.py`` échoue si une étape facultative porte
une condition absente du registre. Sans ce test, on recréerait exactement le
défaut corrigé ici : un mécanisme déclaratif que personne ne lit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional

# Sentinelle : la condition dépend de ce que l'usager demande, pas de l'état.
# Le planner tranche ; la route déterministe ne présume rien.
JUGEMENT = None

# Familles de juridiction. DEUX vocabulaires, comme pour les noms de type de
# demande : le dataset écrit un STATUT (« supported_quebec »), le live écrit
# un NOM de juridiction (« Québec »), parce que ResearchState.jurisdiction_
# status reçoit resolved_jurisdiction. Ne connaître que le premier rendait la
# garde inerte en live — exactement là où le planner choisit librement.
#
# Les valeurs absentes de cette table (municipal_coverage_uncertain,
# supported_other_canadian, unsupported_foreign, unknown, «») ne bloquent
# RIEN : on n'écarte que ce dont on est sûr.
_QUEBEC = frozenset({"Québec"})
_FEDERAL = frozenset({"Federal", "Canada"})

_FAMILLES: dict[str, frozenset[str]] = {
    "supported_quebec": _QUEBEC,
    "quebec": _QUEBEC,
    "qc": _QUEBEC,
    "supported_federal": _FEDERAL,
    "federal": _FEDERAL,
    "canada": _FEDERAL,
}

_RE_ARTICLE = re.compile(r"\barticles?\s+\d{1,4}", re.IGNORECASE)


@dataclass(frozen=True)
class GardeContexte:
    """Tout ce qu'une garde a le droit de lire — et rien de plus.

    Volontairement plus étroit que ``LexiorState`` : une garde doit rester
    pure et testable sans monter un graphe.
    """

    request_type: str
    jurisdiction_status: str
    user_query: str
    tool_history: tuple[Any, ...] = ()


def _article_cite(ctx: GardeContexte) -> bool:
    """Un numéro d'article est-il nommé dans la demande ?

    Pour ``document_analysis``, la condition écrite parle du document fourni.
    Celui-ci arrive collé dans la question, donc la même lecture s'applique ;
    si un jour le document transite par un champ séparé, cette garde devra
    le lire aussi.
    """
    return bool(_RE_ARTICLE.search(ctx.user_query or ""))


def _a_repondu(ctx: GardeContexte, outil: str) -> bool:
    """Cet outil a-t-il déjà renvoyé un résultat exploitable ?"""
    for obs in ctx.tool_history:
        if (getattr(obs, "tool_name", "") == outil
                and getattr(obs, "ok", False)
                and (getattr(obs, "normalized_response", "") or "").strip()):
            return True
    return False


# Registre indexé par la prose EXACTE de la condition. Les trois premières
# entrées remplacent les paires qui étaient codées en dur dans
# ``_effective_route`` ; les quatre suivantes étaient décidables et se
# trouvaient écartées faute d'implémentation.
GARDES: dict[str, Optional[Callable[[GardeContexte], bool]]] = {
    "if Quebec civil law and article unknown":
        lambda c: not _article_cite(c),
    "if procedural provision unknown":
        lambda c: not _article_cite(c),
    "if relevant legislation unknown":
        lambda c: not _article_cite(c),
    "if article identified in document":
        _article_cite,
    "article number known or discoverable":
        lambda c: (_article_cite(c)
                   or _a_repondu(c, "semantic_search_ccq")
                   or _a_repondu(c, "semantic_search_cpc")),
    "federal document identified":
        lambda c: _a_repondu(c, "search_legal_documents"),
    "coverage reveals gap worth investigating":
        lambda c: _a_repondu(c, "coverage"),

    # Indécidables depuis l'état : elles portent sur ce que l'usager veut.
    "article contains open-ended notion or facts warrant it": JUGEMENT,
    "only if user requests application examples or jurisprudence needed": JUGEMENT,
    "only if procedural test needs jurisprudential clarification": JUGEMENT,
    "only if facts involve application to specific situation": JUGEMENT,
}


def _famille(valeur: str) -> Optional[frozenset[str]]:
    """Famille de juridiction, quel que soit le vocabulaire employé."""
    brut = (valeur or "").strip().lower()
    if not brut:
        return None
    sans_accent = unicodedata.normalize("NFD", brut)
    sans_accent = "".join(c for c in sans_accent
                          if unicodedata.category(c) != "Mn")
    return _FAMILLES.get(brut) or _FAMILLES.get(sans_accent)


def juridiction_compatible(tool: str, jurisdiction_status: str) -> bool:
    """Couche 1 : l'outil est-il de la bonne juridiction pour ce scénario ?

    Permissif par défaut : outil inconnu de ``tool_coverage``, outil sans
    juridiction déclarée, ou valeur hors des deux familles connues → on
    n'écarte pas. On ne bloque que ce dont on est sûr.
    """
    attendue = _famille(jurisdiction_status)
    if attendue is None:
        return True
    from lexior.services.tool_coverage import TOOL_COVERAGE
    entree = TOOL_COVERAGE.get(tool)
    declarees = set(getattr(entree, "legal_jurisdictions", ()) or ())
    if not declarees:
        return True
    return bool(declarees & attendue)


def etape_facultative_retenue(
    tool: str, condition: str, ctx: GardeContexte,
) -> bool:
    """Une étape FACULTATIVE entre-t-elle dans la route déterministe ?

    Les étapes obligatoires ne passent pas par ici : les écarter casserait
    la route (``comparative_law`` exige un outil fédéral même au Québec).
    """
    if not juridiction_compatible(tool, ctx.jurisdiction_status):
        return False
    garde = GARDES.get(condition or "")
    if garde is None:                       # absente ou JUGEMENT
        return False
    return bool(garde(ctx))
