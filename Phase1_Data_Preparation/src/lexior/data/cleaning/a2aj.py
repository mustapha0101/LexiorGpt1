#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Couche de nettoyage des données sources A2AJ (a2aj/canadian-laws).

À appliquer AVANT tout appel au modèle Teacher. Le rôle de ce module est de ne
laisser passer que du droit fédéral, en vigueur, et rédigé en français.

Trois niveaux de filtrage :
  1. LOI      : juridiction fédérale uniquement, texte français réel.
  2. LOI      : rejet des lois entièrement abrogées.
  3. ARTICLE  : classification de chaque article ; seuls LIVE et PARTIAL sont
                transmis au Teacher.

Utilisation depuis generator_a2aj.py :

    from a2aj_cleaner import clean_law
    cleaned = clean_law(item)
    if cleaned is None:
        return False              # loi rejetée
    texte = cleaned["context_text"]
    meta  = cleaned["meta"]
"""

import re

# --- Juridictions retenues (§1a) -------------------------------------------
FEDERAL_JURISDICTIONS = {"LEGISLATION-FED", "REGULATIONS-FED"}

# --- Détection des articles-souches ----------------------------------------
# Un article "souche" est un article dont le texte ENTIER tient dans un seul
# crochet, p. ex. "[Abrogé, 2017, ch. 33, art. 228]" ou "[blank]".
_STUB_RE = re.compile(r"^\s*\[(?P<body>[^\]]{0,160})\]\s*$")

# Marqueur d'abrogation, où qu'il soit dans le texte.
_REPEAL_RE = re.compile(r"\[\s*(?:Abrog|Repealed)", re.IGNORECASE)

# Marqueur d'abrogation dans l'en-tête (bloc titre/citation) = loi
# entièrement abrogée. Autoritatif : c'est la mention du législateur.
# N.B. on ne se fie JAMAIS au NOM de la loi ("Loi d'abrogation des lois" est
# une loi ACTIVE dont l'objet est d'abroger d'autres lois).
_HEADER_WINDOW = 400

# Longueur minimale d'un article pour être exploitable.
MIN_SECTION_CHARS = 40
MIN_LAW_CHARS = 150

# Classes d'articles jugées inexploitables (aucun contenu normatif).
UNUSABLE_CLASSES = {"EMPTY", "BLANK", "DEAD", "SPENT", "STUB_OTHER", "TOO_SHORT"}
USABLE_CLASSES = {"LIVE", "PARTIAL"}


def classify_section(text):
    """Classe un article. Retourne l'une des étiquettes suivantes :

    LIVE       : droit en vigueur, aucun marqueur.               -> utilisable
    PARTIAL    : article en vigueur contenant un sous-alinéa
                 abrogé (p. ex. une définition biffée).          -> utilisable
    DEAD       : l'article LUI-MÊME est abrogé — le texte EST la
                 souche, p. ex. "[Abrogé, 2017, ch. 33, art. 228]". -> rejet
    BLANK      : "[blank]" — article vide.                       -> rejet
    SPENT      : disposition épuisée, dont le contenu a été
                 consolidé ailleurs : "[Abrogation]" (l'article
                 servait à abroger UNE AUTRE loi) ou
                 "[Modifications]". L'article n'est PAS abrogé,
                 mais il ne reste rien à interpréter.            -> rejet
    STUB_OTHER : autre souche entre crochets.                    -> rejet
    EMPTY      : chaîne vide.                                    -> rejet
    TOO_SHORT  : moins de MIN_SECTION_CHARS caractères.          -> rejet
    """
    t = (text or "").strip()
    if not t:
        return "EMPTY"

    m = _STUB_RE.match(t)
    if m:
        body = m.group("body").strip().lower()
        if body.startswith("blank"):
            return "BLANK"
        # "abrogé/abrogée/abrogés/abrogées/repealed" = l'article est abrogé.
        # À distinguer de "abrogation(s)", qui désigne une disposition
        # abrogeant une autre loi — elle figure dans des lois EN VIGUEUR
        # (p. ex. art. 89 de la Loi sur la radiodiffusion).
        if re.match(r"(?:abrog[ée]e?s?|repealed)\b", body):
            return "DEAD"
        if re.match(r"(?:abrogations?|modifications?)\b", body):
            return "SPENT"
        return "STUB_OTHER"

    if len(t) < MIN_SECTION_CHARS:
        return "TOO_SHORT"

    # Article vivant, mais contenant un sous-alinéa biffé : exploitable, à
    # condition de ne jamais s'appuyer sur le sous-alinéa lui-même.
    if _REPEAL_RE.search(t):
        return "PARTIAL"

    return "LIVE"


def is_wholly_repealed(raw_text, section_classes):
    """Une loi est entièrement abrogée si :
      (a) l'en-tête porte un marqueur d'abrogation  — OU —
      (b) il ne reste aucun article exploitable.

    L'union des deux est nécessaire : (b) seul laisse passer les lois abrogées
    d'un bloc dont les articles n'ont pas été individuellement biffés (p. ex.
    la Loi sur l'énergie nucléaire, abrogée en 1997, conserve 9 articles
    d'apparence vivante).
    """
    if _REPEAL_RE.search((raw_text or "")[:_HEADER_WINDOW]):
        return True
    return not any(c in USABLE_CLASSES for c in section_classes.values())


def clean_law(item, max_context_chars=4000, whole_only=False):
    """Filtre et nettoie une ligne brute d'a2aj/canadian-laws.

    max_context_chars vaut 4000 pour reproduire exactement le budget déjà
    utilisé par generator_a2aj.py ; la coupure se fait toutefois sur une
    frontière d'article plutôt qu'au milieu d'une phrase.

    whole_only=True n'accepte que les lois tenant entièrement dans le budget
    (2 413 lois sur 4 727, soit 51 %). Les lois écartées ne sont pas perdues :
    elles attendent un découpage par article, la troncature ne pouvant les
    représenter honnêtement (elle en garde surtout les définitions).

    Retourne None si la loi est rejetée, sinon un dict :
        {
          "context_text" : texte des articles vivants, prêt pour le Teacher,
          "meta"         : {title, url, jurisdiction, citation,
                            section_ids_used},
          "stats"        : compteur par classe d'article,
        }

    Les clés "title" et "url" conservent les noms attendus par le code
    existant (generator_a2aj.py:160 et :175) afin de ne rien casser en aval.
    """
    jurisdiction = item.get("dataset", "")
    if jurisdiction not in FEDERAL_JURISDICTIONS:
        return None

    # Français uniquement : le fédéral est bilingue, donc pas de repli vers
    # l'anglais — un texte fédéral sans version française est une anomalie.
    raw_text = item.get("unofficial_text_fr") or ""
    if len(raw_text.strip()) < MIN_LAW_CHARS:
        return None

    raw_sections = item.get("unofficial_sections_fr") or ""
    try:
        import json
        sections = json.loads(raw_sections) if raw_sections else {}
    except (ValueError, TypeError):
        sections = {}
    if not isinstance(sections, dict) or not sections:
        return None  # rien à donner au Teacher

    classes = {sid: classify_section(txt) for sid, txt in sections.items()}

    if is_wholly_repealed(raw_text, classes):
        return None

    usable = [(sid, sections[sid]) for sid, c in classes.items() if c in USABLE_CLASSES]
    if not usable:
        return None

    # Ordre naturel des articles (1, 2, 5.1, 10 …) plutôt qu'alphabétique.
    def _key(pair):
        return [int(p) if p.isdigit() else 0 for p in re.split(r"[.\-]", pair[0])[:4]]
    try:
        usable.sort(key=_key)
    except (ValueError, TypeError):
        usable.sort(key=lambda p: p[0])

    # Les blocs sont joints par "\n\n" : le séparateur compte dans le budget,
    # sinon context_text dépasse max_context_chars de 2 x (nb_articles - 1).
    parts, used, total = [], [], 0
    truncated = False
    for sid, txt in usable:
        block = f"**{sid}** {txt.strip()}"
        sep = 2 if parts else 0
        if total + sep + len(block) > max_context_chars:
            # Un article isolé plus long que le budget ne doit pas faire
            # rejeter la loi : on le tronque et on s'arrête là.
            if not parts:
                parts.append(block[:max_context_chars])
                used.append(sid)
            truncated = True
            break
        parts.append(block)
        used.append(sid)
        total += sep + len(block)
    if not used:
        return None

    # whole_only : ne conserver que les lois qui tiennent ENTIÈREMENT dans le
    # budget. Une loi tronquée n'est pas une loi plus courte — c'est presque
    # toujours ses définitions d'ouverture sans ses dispositions de fond. Tant
    # que les longues lois ne sont pas découpées par article, mieux vaut les
    # écarter que d'en tirer une question sur leur seul lexique.
    if whole_only and truncated:
        return None

    from collections import Counter
    return {
        "context_text": "\n\n".join(parts),
        "meta": {
            # noms de clés conservés depuis generator_a2aj.py
            "title": item.get("name_fr") or item.get("name_en") or "Loi fédérale",
            "url": item.get("source_url_fr") or item.get("source_url_en") or "",
            # remplace l'ancien champ "section": "N/A", qui était toujours vide
            # ("section" n'existe pas dans le schéma A2AJ)
            "section_ids_used": used,
            "jurisdiction": jurisdiction,
            "citation": item.get("citation_fr") or item.get("citation_en") or "",
        },
        "stats": dict(Counter(classes.values())),
    }
