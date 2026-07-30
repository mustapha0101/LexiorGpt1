# -*- coding: utf-8 -*-

"""L'affirmation portée par une citation est-elle soutenue par le TEXTE ?

``validate_final`` vérifiait que le NUMÉRO d'article cité figure dans les
preuves. Jamais que ce qui est affirmé à son sujet corresponde à ce que
l'article dit.

Le trou est démontré. Sur une question de branches d'arbre, le modèle a
inventé le contenu des articles 984, 985 et 986 AVANT de les lire, a récupéré
leur vrai texte, et a gardé son invention :

    « les fruits qui tombent d'un arbre appartiennent au propriétaire de
      l'arbre, ce qui souligne la responsabilité du propriétaire en cas de
      dommages »

La première moitié est exacte, la seconde ne découle pas du texte. Le numéro
étant bien dans les preuves, la trajectoire a été acceptée.

Ce module pose au modèle la seule question qui compte : *ce texte-ci
soutient-il cette affirmation-là*. Pas si le numéro est le bon, pas si
l'affirmation est juridiquement correcte en général.

Un échec technique n'est jamais un succès : il produit un verdict marqué
``echec_technique``, que ``validate_final`` transforme en erreur bloquante.
Laisser passer faute d'avoir pu vérifier reviendrait à ne pas vérifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# « article 1457 », « articles 984 et 985 », « art. 596.1 »
_RE_CITATION = re.compile(
    r"\b(?:articles?|art\.)\s*(\d{1,4}(?:\.\d+)?)"
    r"((?:\s*(?:,|;|et|ou|à)\s*\d{1,4}(?:\.\d+)?)*)",
    re.IGNORECASE)
_RE_AUTRES_NUMEROS = re.compile(r"\d{1,4}(?:\.\d+)?")
# Fin de phrase : point, point-virgule ou saut de ligne.
_RE_PHRASE = re.compile(r"[^.;\n]+[.;\n]?")

_SYSTEME = (
    "Tu cherches UNE seule chose : l'affirmation ajoute-t-elle un contenu "
    "que le texte de loi fourni ne permet pas ?\n"
    "\n"
    "Réponds soutenue=false UNIQUEMENT si l'affirmation contredit le texte, "
    "ou lui prête une portée, une conséquence ou une condition qui n'y "
    "figure pas.\n"
    "\n"
    "Réponds soutenue=true dans tous les autres cas, en particulier :\n"
    "- l'affirmation est INCOMPLÈTE, elle omet une partie du texte — une "
    "omission n'est pas une invention ;\n"
    "- elle reformule, résume ou vulgarise fidèlement ;\n"
    "- elle applique le texte à la situation de l'usager sans lui ajouter de "
    "règle ;\n"
    "- elle renvoie à un article dont le texte ne t'est pas fourni : ignore "
    "cette partie, juge seulement ce que les textes fournis permettent de "
    "trancher.\n"
    "\n"
    "On ne te demande NI si le numéro d'article est le bon, NI si "
    "l'affirmation est juridiquement correcte en général, NI si elle est "
    "complète.\n"
    "Réponds uniquement par "
    '{"soutenue": true|false, "motif": "une phrase"}.')


@dataclass
class VerdictAffirmation:
    article: str
    affirmation: str
    soutenue: bool
    motif: str = ""
    echec_technique: bool = False

    def probleme(self) -> str:
        if self.echec_technique:
            return (f"article {self.article} : vérification impossible "
                    f"({self.motif})")
        return (f"article {self.article} : « {self.affirmation[:120]} » "
                f"n'est pas soutenu par le texte — {self.motif}")


def _numeros_cites(reponse: str) -> list[tuple[str, int, int]]:
    """(numéro, début, fin) de chaque article cité, énumérations comprises."""
    trouves: list[tuple[str, int, int]] = []
    for match in _RE_CITATION.finditer(reponse or ""):
        trouves.append((match.group(1), match.start(), match.end()))
        for suite in _RE_AUTRES_NUMEROS.finditer(match.group(2) or ""):
            trouves.append((suite.group(0), match.start(), match.end()))
    return trouves


def affirmation_autour(reponse: str, debut: int, fin: int) -> str:
    """La phrase qui porte la citation, plus la suivante.

    L'invention observée tient dans la proposition qui suit la citation
    exacte — « …, ce qui souligne la responsabilité du propriétaire ». Une
    fenêtre d'une seule phrase la manquerait quand elle déborde.
    """
    phrases = [(m.start(), m.end()) for m in _RE_PHRASE.finditer(reponse or "")]
    if not phrases:
        return (reponse or "").strip()
    index = next((i for i, (d, f) in enumerate(phrases)
                  if d <= debut < f), None)
    if index is None:
        index = next((i for i, (d, f) in enumerate(phrases) if f > debut), 0)
    morceaux = phrases[index:index + 2]
    return " ".join(reponse[d:f].strip() for d, f in morceaux).strip()


class AssertionGroundingService:
    """Vérifie chaque affirmation contre le texte réellement récupéré."""

    def __init__(self, client: Any = None, offline: bool = False,
                 role: str = "legal_critic"):
        self.client = client
        self.offline = offline
        self.role = role

    def disponible(self) -> bool:
        return bool(self.client) and not self.offline

    def verifier(self, reponse: str,
                 textes: dict[str, str]) -> list[VerdictAffirmation]:
        """Un verdict par couple (article cité, affirmation qui l'entoure).

        Les articles cités sans texte récupéré ne sont pas jugés ici : c'est
        le contrôle d'ancrage par numéro qui les couvre déjà.
        """
        if not self.disponible() or not (reponse or "").strip():
            return []
        verdicts: list[VerdictAffirmation] = []
        deja: set[tuple[str, str]] = set()
        for numero, debut, fin in _numeros_cites(reponse):
            if numero not in textes:
                continue
            affirmation = affirmation_autour(reponse, debut, fin)
            cle = (numero, affirmation)
            if cle in deja:
                continue
            deja.add(cle)
            # Une affirmation cite souvent PLUSIEURS articles — « selon les
            # articles 1889 et 1963 ». Ne montrer au juge que le texte de
            # l'un d'eux le fait conclure « non soutenue » parce que l'autre
            # manque : c'était un artefact du contrôle, pas une invention du
            # modèle. On fournit donc tous les textes disponibles cités dans
            # la fenêtre.
            portee = {n for n, _, _ in _numeros_cites(affirmation)
                      if n in textes} or {numero}
            verdicts.append(self._juger(
                numero, affirmation, {n: textes[n] for n in sorted(portee)}))
        return verdicts

    def _juger(self, numero: str, affirmation: str,
               textes: dict[str, str]) -> VerdictAffirmation:
        corpus = "\n\n".join(
            f"TEXTE OFFICIEL DE L'ARTICLE {n} :\n{t}"
            for n, t in textes.items())
        try:
            reponse = self.client.complete_json(
                self.role,
                [{"role": "system", "content": _SYSTEME},
                 {"role": "user",
                  "content": (f"{corpus}\n\nAFFIRMATION À VÉRIFIER "
                              f"(porte notamment sur l'article {numero}) :\n"
                              f"{affirmation}")}],
                temperature=0.0)
        except Exception as exc:                       # noqa: BLE001
            # Ne pas pouvoir vérifier n'est pas une vérification réussie.
            return VerdictAffirmation(
                article=numero, affirmation=affirmation, soutenue=False,
                motif=f"{type(exc).__name__}: {exc}"[:160],
                echec_technique=True)
        if not isinstance(reponse, dict) or "soutenue" not in reponse:
            return VerdictAffirmation(
                article=numero, affirmation=affirmation, soutenue=False,
                motif="réponse du vérificateur illisible",
                echec_technique=True)
        return VerdictAffirmation(
            article=numero, affirmation=affirmation,
            soutenue=bool(reponse.get("soutenue")),
            motif=str(reponse.get("motif") or "")[:200])


def textes_recuperes(tool_history: Any) -> dict[str, str]:
    """Numéro d'article -> texte officiel, depuis les réponses d'outils."""
    textes: dict[str, str] = {}
    for observation in (tool_history or []):
        if not getattr(observation, "ok", False):
            continue
        if getattr(observation, "tool_name", "") not in (
                "get_ccq_articles", "get_cpc_articles"):
            continue
        contenu = getattr(observation, "normalized_response", "") or ""
        blocs = re.split(r"(?m)^\s*Article\s+(\d{1,4}(?:\.\d+)?)\s*$",
                         contenu)
        if len(blocs) > 1:
            for i in range(1, len(blocs) - 1, 2):
                textes[blocs[i]] = blocs[i + 1].strip()
            continue
        # Un seul article, en-tête sur la même ligne que le texte.
        entete = re.match(r"\s*Article\s+(\d{1,4}(?:\.\d+)?)\b", contenu)
        if entete:
            textes[entete.group(1)] = contenu[entete.end():].strip()
    return textes
