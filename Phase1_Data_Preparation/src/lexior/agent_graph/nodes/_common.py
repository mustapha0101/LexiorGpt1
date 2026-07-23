# -*- coding: utf-8 -*-
"""Utilitaires partagés par les nœuds du graphe."""

from __future__ import annotations

import re
from typing import Optional

from agentic_generation.schemas import Message

_GREETING_RE = re.compile(
    r"^\s*(bonjour|salut|allo|hello|hi|merci)\b[\s!.,]*$", re.I)

_ARTICLE_REQUEST_RE = re.compile(
    r"\b(cite|donne|montre|texte\s+(?:exact|int[ée]gral|officiel))\b.*"
    r"\barticle\b|\barticle\s+exact\b",
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
    if _ANAPHORA_RE.search(stripped):
        return True
    return (len(stripped) <= 80
            and bool(_FOLLOW_UP_VERB_RE.match(stripped)))


def detect_case_reference(text: str) -> Optional[str]:
    match = re.search(
        r"\b\d{4}\s+(?:QCCA|QCCS|QCCQ|QCTDP|SCC|CSC|FC|CF|FCA|CAF)\s+\d+\b",
        text or "")
    return match.group(0) if match else None
