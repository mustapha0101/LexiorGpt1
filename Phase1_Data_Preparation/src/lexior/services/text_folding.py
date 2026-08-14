# -*- coding: utf-8 -*-
"""Repliement de texte partagé par les vérifications de grounding.

Le corpus officiel est intégralement typographique : ``documents.jsonl``
contient 21 932 apostrophes U+2019 pour UNE seule apostrophe ASCII. Or
``unicodedata.normalize("NFKD", …)`` ne replie PAS U+2019 vers U+0027 —
c'est une ponctuation, pas un caractère décomposable.

Conséquence avant ce module : un relecteur qui recopiait « l'obligation »
avec une apostrophe droite voyait son extrait déclaré absent de l'article,
et le fait requis était abandonné silencieusement. La réponse perdait sa
formulation conditionnelle sans qu'aucun signal ne soit émis.

Toutes les comparaisons « cet extrait figure-t-il dans la source ? »
DOIVENT passer par ``fold_text``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Ponctuation que NFKD laisse intacte alors qu'elle est interchangeable
# à l'écrit. Les guillemets et tirets suivent la même logique que les
# apostrophes : la source est typographique, la copie ne l'est pas.
_PUNCTUATION_EQUIVALENTS = {
    0x2018: "'",   # ‘ guillemet-apostrophe culbuté
    0x2019: "'",   # ’ apostrophe typographique — le cas dominant
    0x201A: "'",   # ‚
    0x201B: "'",   # ‛
    0x02BC: "'",   # ʼ lettre apostrophe
    0x00B4: "'",   # ´ accent aigu employé comme apostrophe
    0x0060: "'",   # ` accent grave employé comme apostrophe
    0x201C: '"',   # “
    0x201D: '"',   # ”
    0x201E: '"',   # „
    0x00AB: '"',   # «
    0x00BB: '"',   # »
    0x2010: "-",   # ‐
    0x2011: "-",   # ‑ tiret insécable
    0x2012: "-",   # ‒
    0x2013: "-",   # –
    0x2014: "-",   # —
    0x2212: "-",   # −
    0x00A0: " ",   # espace insécable
    0x202F: " ",   # espace fine insécable
    0x2009: " ",   # espace fine
    0x2007: " ",   # espace chiffre
    0x2026: "...",  # …
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_punctuation(value: Any) -> str:
    """Uniformise apostrophes, guillemets, tirets et espaces."""
    return str(value or "").translate(_PUNCTUATION_EQUIVALENTS)


def fold_text(value: Any, *, collapse_whitespace: bool = True) -> str:
    """Forme repliée : ponctuation uniformisée, accents ôtés, minuscules.

    ``collapse_whitespace`` réduit toute suite d'espaces à un seul et
    supprime les bords — c'est ce qu'il faut pour un test d'inclusion
    d'extrait, où le découpage des lignes de la source ne doit pas
    compter.
    """
    text = unicodedata.normalize("NFKD", normalize_punctuation(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold()
    if collapse_whitespace:
        return _WHITESPACE_RE.sub(" ", text).strip()
    return text
