# -*- coding: utf-8 -*-
"""Utilitaires partagés par les nœuds du graphe."""

from __future__ import annotations

import re
from typing import Optional

from lexior.agentic.schemas import Message
from lexior.agentic.citations import find_case_citation

_GREETING_RE = re.compile(
    r"^\s*(bonjour|salut|allo|hello|hi|merci)\b[\s!.,]*$", re.I)

_ARTICLE_REQUEST_RE = re.compile(
    r"(?:"
    r"\b(?:cite|donne|montre|reproduis|fournis)\b.*\barticles?\b"
    r"|\btexte\s+(?:exact|int[ée]gral|officiel)\b.*\barticles?\b"
    r"|\bque\s+di(?:t|sent)\b.*\barticles?\b"
    r"|\barticles?\b.*\b(?:exactement|mot\s+[àa]\s+mot|"
    r"texte\s+(?:exact|int[ée]gral|officiel))\b"
    r")",
    re.I | re.S,
)

_SOURCE_REQUEST_RE = re.compile(
    r"\b(le\s+site|le\s+lien|l'url|la\s+source|le\s+jugement\s+complet|"
    r"o[ùu]\s+(?:consulter|trouver|lire))\b",
    re.I,
)

_FOLLOW_UP_VERB_RE = re.compile(
    r"^\s*(donne|cite|montre|envoie|fournis|r[ée]sume|explique|"
    r"traduis|d[ée]taille|reformule|continue|et\s+si|pourquoi|"
    r"quel(?:le)?s?\b|combien)\b",
    re.I,
)

_ANAPHORA_RE = re.compile(
    r"\b(cette\s+d[ée]cision|ce\s+jugement|cet\s+arr[êe]t|cette\s+loi|"
    r"cet\s+article|ce\s+cas|celle-l[àa]|celui-l[àa]|le\s+premier|"
    r"la\s+derni[èe]re)\b",
    re.I,
)

# Tout signal juridique — même faible — désarme la détection « non
# juridique ». La liste couvre le vocabulaire du droit ET celui des
# situations qui l'appellent (préjudice, congédiement, bail…).
_LEGAL_MARKER_RE = re.compile(
    r"\b(droits?|loi|lois|l[ée]gal|l[ée]gale|juridique|juridiction|"
    r"articles?|c\.?c\.?q|c\.?p\.?c|r[èe]glement|jurisprudence|"
    # « code » seul désigne aussi du code informatique : il ne compte comme
    # marqueur juridique qu'accompagné de son intitulé.
    r"code\s+(?:civil|criminel|de\s+proc[ée]dure|du\s+travail|"
    r"de\s+la\s+route|de\s+la\s+s[ée]curit[ée])|"
    r"tribunal|cour|juge|avocat|notaire|proc[èe]s|poursuite|litige|"
    r"recours|plainte|mise\s+en\s+demeure|assign|contrat|clause|bail|"
    r"locataire|locateur|propri[ée]taire|loyer|[ée]viction|hypoth[èe]que|"
    r"succession|testament|h[ée]ritage|divorce|s[ée]paration|garde|"
    r"pension\s+alimentaire|cong[ée]di|licenci|employeur|employ[ée]|"
    r"salaire|responsab|pr[ée]judice|dommages?|indemnit|faute|"
    r"n[ée]glig|infraction|amende|contravention|police|arrest|criminel|"
    r"assurance|garantie|vice\s+cach[ée]|consommateur|rembours|"
    r"obligation|cr[ée]anc|dette|faillite|permis|licence|"
    r"discrimination|harc[èe]l|vie\s+priv[ée]e|renseignements\s+personnels|"
    r"refuse\s+de|a\s+droit|ai-?je\s+le\s+droit|annul)",
    re.I,
)

# Sujets explicitement hors du droit. Ne déclenche RIEN à lui seul :
# il faut aussi l'absence de tout marqueur juridique ci-dessus.
_NON_LEGAL_TOPIC_RE = re.compile(
    r"\b(recette|cuisin|ingr[ée]dient|dessert|"
    r"m[ée]t[ée]o|quel\s+temps\s+fait|temps\s+qu'?il\s+fait|"
    r"pluie|neige|temp[ée]rature|ensoleill|"
    r"capitale\s+de|population\s+de|traduis|traduction|"
    r"calcule?[- ]|combien\s+font|racine\s+carr[ée]e|"
    r"blague|histoire\s+dr[ôo]le|chanson|film|s[ée]rie\s+t[ée]l|musique|"
    r"hockey|soccer|football|match\s+de|[ée]quipe\s+de\s+(?:hockey|soccer)|"
    r"code\s+(?:python|javascript|java|c\+\+|html|sql)|"
    r"[ée]cris[- ]moi|programme?r\b|script\s+python|"
    r"po[èe]me|voyage|itin[ée]raire)",
    re.I,
)


def last_user_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if getattr(message.role, "value", message.role) == "user":
            return message.content
    return ""


def last_assistant_content(messages: list[Message]) -> str:
    for message in reversed(messages):
        if getattr(message.role, "value", message.role) == "assistant":
            return message.content
    return ""


def first_user_content(messages: list[Message]) -> str:
    for message in messages:
        if getattr(message.role, "value", message.role) == "user":
            return message.content
    return ""


def user_turn_count(messages: list[Message]) -> int:
    return sum(
        1 for m in messages
        if getattr(m.role, "value", m.role) == "user")


def is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text or ""))


def is_clearly_non_legal(text: str) -> bool:
    """Demande manifestement hors du droit, détectée sans appel de modèle.

    Volontairement CONJONCTIVE : il faut un sujet explicitement non
    juridique ET aucun marqueur juridique. Une question de droit exprimée
    en langage courant — « mon fils a cassé la vitrine du dépanneur » —
    ne porte aucun marqueur non juridique et reste donc traitée comme
    juridique. Le faux négatif (on cherche du droit pour rien) coûte une
    recherche; le faux positif (on refuse une vraie question) coûte la
    réponse.
    """
    value = text or ""
    if not value.strip():
        return False
    if _LEGAL_MARKER_RE.search(value):
        return False
    return bool(_NON_LEGAL_TOPIC_RE.search(value))


def requested_output_type(text: str) -> str:
    """Type de sortie demandé, détecté déterministiquement."""
    if _ARTICLE_REQUEST_RE.search(text or ""):
        return "article_text"
    if _SOURCE_REQUEST_RE.search(text or ""):
        return "source_url"
    return "answer"


def looks_like_follow_up(text: str, has_previous_answer: bool) -> bool:
    """Message court qui prolonge l'échange plutôt qu'il ne l'ouvre."""
    if not has_previous_answer:
        return False
    stripped = (text or "").strip()
    if not stripped:
        return False
    # Une demande explicite de texte ou de source prolonge naturellement la
    # réponse précédente, même si elle commence par « que disent... » plutôt
    # que par l'un des verbes courts ci-dessous.
    if _ARTICLE_REQUEST_RE.search(stripped) or _SOURCE_REQUEST_RE.search(stripped):
        return True
    if _ANAPHORA_RE.search(stripped):
        return True
    return (len(stripped) <= 80
            and bool(_FOLLOW_UP_VERB_RE.match(stripped)))


def detect_case_reference(text: str) -> Optional[str]:
    return find_case_citation(text) or None
