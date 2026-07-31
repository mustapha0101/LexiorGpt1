# -*- coding: utf-8 -*-

"""D'où vient un numéro d'article ? — source unique pour deux contrôles.

Sur une question de morsure de chien, après deux recherches vides, le planner
a appelé ``get_ccq_articles(1465)`` en écrivant « je sais que l'article 1465
stipule que le propriétaire d'un animal est responsable ». C'est faux — 1465
vise les biens, 1466 les animaux — et ça contredit la première ligne du prompt
système : jamais de loi de mémoire.

Deux contrôles s'appuient sur la même question, et doivent donc partager la
même réponse :

* le planner REFUSE de récupérer un numéro sans provenance ;
* le contrôle de pertinence n'applique PAS son veto lexical à un numéro qui
  en a une.

Le second dépend du premier. Le veto lexical déclare hors sujet tout résultat
sans racine commune avec la question : « le chien de ma voisine m'a mordue »
ne partage aucun mot avec « le propriétaire d'un animal est tenu de réparer
le préjudice », et l'article 1466 — le bon — ressortait ``irrelevant``. C'est
le même écart de vocabulaire qui a fermé les sept pistes d'optimisation.
Exempter la récupération par numéro n'est légitime QUE parce que le numéro
est désormais garanti provenir d'une recherche ou de la question.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

# Outils qui récupèrent un texte officiel par son numéro, et le champ qui le
# porte.
RECUPERATION_PAR_NUMERO: dict[str, tuple[str, ...]] = {
    # ``start_article`` est le format historique. Les serveurs MCP actuels
    # acceptent aussi ``articles: [..]`` pour récupérer plusieurs textes.
    "get_ccq_articles": ("start_article", "articles"),
    "get_cpc_articles": ("start_article", "articles"),
}

_RE_NUMERO = re.compile(r"\b(\d{1,4}(?:\.\d+)?)\b")


def numeros_dans(texte: str) -> set[str]:
    return set(_RE_NUMERO.findall(texte or ""))


def numeros_demandes(tool_name: str, arguments: Optional[dict]) -> tuple[str, ...]:
    """Numéros réellement passés à un outil de récupération officiel.

    Un appel groupé doit conserver la provenance de *chaque* article. Le
    réduire au premier, ou ne reconnaître que le schéma historique
    ``start_article``, permettrait à un résultat officiel pertinent d'être
    rejeté par le veto lexical après une récupération valide.
    """
    champs = RECUPERATION_PAR_NUMERO.get(tool_name or "")
    if not champs:
        return ()
    numeros: list[str] = []
    for champ in champs:
        brut = (arguments or {}).get(champ)
        valeurs = brut if isinstance(brut, (list, tuple, set)) else (brut,)
        for valeur in valeurs:
            numero = str(valeur).strip() if valeur is not None else ""
            if numero and numero not in numeros:
                numeros.append(numero)
    return tuple(numeros)


def numero_demande(tool_name: str, arguments: Optional[dict]) -> Optional[str]:
    """Compatibilité : premier numéro demandé, ou ``None``.

    Les nouveaux appelants qui évaluent la provenance doivent utiliser
    :func:`numeros_demandes` afin de ne pas perdre les articles groupés.
    """
    return next(iter(numeros_demandes(tool_name, arguments)), None)


def a_une_provenance(
    numero: str, question: str, reponses_precedentes: Iterable[str],
) -> bool:
    """Ce numéro a-t-il été produit par une recherche, ou écrit par l'usager ?"""
    if not numero:
        return True
    if numero in numeros_dans(question):
        return True
    return any(numero in numeros_dans(reponse)
               for reponse in reponses_precedentes)


def reponses_reussies(observations: Iterable[Any],
                      avant: Optional[Any] = None) -> list[str]:
    """Textes des observations réussies, jusqu'à ``avant`` exclu s'il est donné."""
    textes: list[str] = []
    for obs in observations:
        if avant is not None and obs is avant:
            break
        if getattr(obs, "ok", False):
            textes.append(getattr(obs, "normalized_response", "") or "")
    return textes
