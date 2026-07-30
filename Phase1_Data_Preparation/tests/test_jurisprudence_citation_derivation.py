# -*- coding: utf-8 -*-

"""La citation dérivée par le serveur doit être reconnue par nos regex.

Les titres renvoyés par CanLII via Exa ne portent PAS la citation : ce sont
« D É C I S I O N » ou « Décision sans titre ». Tant que la sortie n'en
contenait aucune, ``_classify_jurisprudence`` renvoyait ``irrelevant`` et le
lot entier était jeté — l'outil rendait les bons documents et le pipeline les
rejetait tous.

``mcp-server/src/scraper.ts`` dérive donc la citation de l'URL. Ces tests
vérifient le contrat entre les deux côtés : le format produit là-bas doit
être reconnu ici, pour chaque tribunal de ``citations.py``.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lexior.agentic.citations import (  # noqa: E402
    CASE_CITATION_RE, QUEBEC_COURT_SCOPES,
)
from lexior.agent_graph.result_classifier import ResultClassifier  # noqa: E402

# Miroir fidèle de citationDepuisUrl() — mcp-server/src/scraper.ts.
# Le tribunal vient de l'identifiant du document, jamais du segment de chemin :
# /qc/qcrdl/doc/2010/2010canlii140580/ se cite « 2010 CanLII 140580 », et le
# réécrire « 2010 QCRDL 140580 » fabriquerait une référence inexistante.
_RE_DOCID_CANLII = re.compile(r"/doc/\d{4}/(\d{4})([a-z]+)(\d+)", re.I)


def citation_depuis_url(url: str) -> str | None:
    match = _RE_DOCID_CANLII.search(url or "")
    if not match:
        return None
    return f"{match.group(1)} {match.group(2).upper()} {match.group(3)}"


# Sortie réellement renvoyée par le serveur local le 2026-07-29, requête
# « résiliation de bail logement insalubre ». Non retouchée.
SORTIE_REELLE = (
    "\n### 2022 QCTAL 35727\n"
    "https://www.canlii.org/fr/qc/qctal/doc/2022/2022qctal35727/2022qctal35727.html\n"
    "  \n\n\n### 2010 CANLII 140580\n"
    "https://www.canlii.org/fr/qc/qcrdl/doc/2010/2010canlii140580/"
    "2010canlii140580.html?searchUrlHash=AAAAAQAoQ29vcC4&resultIndex=2\n"
    "  \n\n\n### 2019 QCRDL 27403\n"
    "https://www.canlii.org/fr/qc/qcrdl/doc/2019/2019qcrdl27403/"
    "2019qcrdl27403.html?resultIndex=1\n"
    "  \n\n\n### Décision\n"
    "https://citoyens.soquij.qc.ca/php/decision.php?ID=5B90AAD41A54AEB822A1\n"
    "  "
)

# La même, telle qu'elle était AVANT la dérivation : titres CanLII bruts.
SORTIE_SANS_CITATION = (
    "\n### [D É C I S I O N](https://www.canlii.org/fr/qc/qctal/doc/2022/"
    "2022qctal35727/2022qctal35727.html)\n"
    "  \n\n\n### [Décision sans titre](https://www.canlii.org/fr/qc/qcrdl/doc/"
    "2019/2019qcrdl27403/2019qcrdl27403.html)\n  "
)


@pytest.mark.parametrize("scope", QUEBEC_COURT_SCOPES)
def test_chaque_tribunal_de_citations_py_est_reconnu(scope):
    """Ajouter un tribunal dans citations.py ne doit pas casser la chaîne."""
    code = scope.lower()
    url = (f"https://www.canlii.org/fr/qc/{code}/doc/2022/"
           f"2022{code}12345/2022{code}12345.html")
    citation = citation_depuis_url(url)
    assert citation == f"2022 {scope} 12345"
    assert CASE_CITATION_RE.search(citation), (
        f"{scope} est dans QUEBEC_COURT_SCOPES mais la citation dérivée "
        f"« {citation} » n'est pas reconnue par CASE_CITATION_RE")


def test_soquij_na_pas_cette_structure_et_garde_son_titre():
    assert citation_depuis_url(
        "https://citoyens.soquij.qc.ca/php/decision.php?ID=5B90AAD4") is None


def test_le_tribunal_vient_de_lidentifiant_pas_du_chemin():
    """Une décision d'avant les citations neutres reste « CanLII »."""
    assert citation_depuis_url(
        "https://www.canlii.org/fr/qc/qcrdl/doc/2010/2010canlii140580/"
        "2010canlii140580.html") == "2010 CANLII 140580"


def test_sortie_reelle_classee_usable():
    """Le test qui compte : la vraie sortie traverse le classificateur."""
    statut = ResultClassifier().classify(
        "search_quebec_jurisprudence", SORTIE_REELLE, True)
    assert getattr(statut, "value", statut) == "usable"


def test_sortie_sans_citation_etait_rejetee():
    """Témoin : sans citation dérivée, tout le lot partait à la poubelle."""
    statut = ResultClassifier().classify(
        "search_quebec_jurisprudence", SORTIE_SANS_CITATION, True)
    assert getattr(statut, "value", statut) == "irrelevant"
