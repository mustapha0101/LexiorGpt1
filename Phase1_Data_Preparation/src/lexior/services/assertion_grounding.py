# -*- coding: utf-8 -*-

"""L'affirmation portée par une citation est-elle soutenue par le TEXTE ?

``validate_final`` vérifiait que le NUMÉRO d'article cité figure dans les
preuves. Jamais que ce qui est affirmé à son sujet corresponde à ce que
l'article dit.

Le trou est démontré lorsqu'un modèle conserve une conséquence juridique
inventée après avoir récupéré le texte officiel. Vérifier uniquement que le
numéro de disposition est présent dans les preuves ne suffit pas : il faut
vérifier le lien entre l'affirmation et les conditions du texte.

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

from .text_folding import fold_text

# « article 1457 », « articles 984 et 985 », « art. 596.1 »
_RE_CITATION = re.compile(
    r"\b(?:articles?|art\.)\s*(\d{1,4}(?:\.\d+)?)"
    r"((?:\s*(?:,|;|et|ou|à)\s*\d{1,4}(?:\.\d+)?)*)",
    re.IGNORECASE)
_RE_AUTRES_NUMEROS = re.compile(r"\d{1,4}(?:\.\d+)?")
# Fin de phrase : point, point-virgule ou saut de ligne.
_RE_PHRASE = re.compile(r"[^.;\n]+[.;\n]?")
_FAIT_DEJA_REALISE_RE = re.compile(
    r"\b(?:deja|a\s+(?:cause|subi|detruit|endommage)|est\s+"
    r"(?:survenu|tombe|arrive)|s['']est\s+(?:produit|effondre)|"
    r"has\s+(?:occurred|caused)|was\s+(?:damaged|destroyed))\b", re.I)
_MESURE_PREVENTIVE_RE = re.compile(
    r"\b(?:menace|risque|susceptible|prevenir|prevention|eviter|empecher|"
    r"avant\s+que|may\s+fall|threatens?|prevent)\b", re.I)
_REPARATION_RE = re.compile(
    r"\b(?:repar(?:er|ation)|indemnis(?:er|ation)|dedommag|rembours|"
    r"prejudice|dommages?|compensat(?:ion|e)|repair|compensat)\b", re.I)
_ARTICLE_SELECTION_BATCH_SIZE = 8

_SYSTEME = (
    "On te donne le TEXTE OFFICIEL d'un ou plusieurs articles, les FAITS "
    "décrits par la personne, et une RÉPONSE qui lui a été donnée.\n"
    "\n"
    "Relève toute AFFIRMATION DE DROIT de la réponse — ce à quoi la personne "
    "a droit, ce qu'elle peut exiger, une condition, un délai, une "
    "conséquence — que les textes fournis ne permettent PAS. Trois cas :\n"
    "\n"
    "1. elle contredit le texte ;\n"
    "2. elle énonce une conséquence, une condition ou une portée absente du "
    "texte ;\n"
    "3. elle applique un article dont les CONDITIONS ne correspondent pas aux "
    "faits — par exemple une disposition conditionnée à un événement futur "
    "alors que les faits indiquent que cet événement est déjà survenu.\n"
    "\n"
    "Une règle générale peut soutenir une application CONDITIONNELLE même si "
    "elle ne répète pas littéralement l'objet ou l'événement décrit par la "
    "personne. Ne l'écarte pas pour cette seule absence de mots identiques si "
    "la réponse expose les conditions du texte comme restant à établir.\n"
    "\n"
    "Ne relève PAS :\n"
    "- une omission : on ne juge pas si la réponse est complète ;\n"
    "- une reformulation, un résumé ou une vulgarisation fidèle ;\n"
    "- l'application du texte aux faits quand les conditions SONT réunies ;\n"
    "- un conseil pratique non juridique — documenter, consulter un avocat, "
    "contacter la partie adverse ;\n"
    "- ce qui est attribué à un article dont le texte ne t'est pas fourni.\n"
    "\n"
    "On ne te demande NI si le numéro d'article est le bon, NI si "
    "l'affirmation est juridiquement correcte en général, NI si elle est "
    "complète.\n"
    "\n"
    'Réponds uniquement par {"non_soutenues": [{"affirmation": "citation '
    'exacte de la réponse", "motif": "une phrase"}]}. Liste vide si tout est '
    "soutenu.")

_SELECTION_SYSTEME = (
    "On te donne les FAITS décrits par une personne et le TEXTE OFFICIEL de "
    "plusieurs articles. Évalue chaque article seulement à partir de son "
    "texte, sans règle mémorisée et sans choisir l'article qui serait le plus "
    "complet en général.\n\n"
    "Pour chaque article, donne un statut :\n"
    "- applicable : une règle opérante du texte vise le même événement et "
    "le même objet que la demande, et ses conditions exprimées sont "
    "compatibles avec les faits;\n"
    "- incertain : un fait nécessaire manque ou demeure ambigu, mais une "
    "règle opérante vise déjà le même événement et le même objet;\n"
    "- incompatible : aucune règle opérante du texte ne vise l'événement ou "
    "l'objet de la demande, ou une condition exprimée contredit les faits.\n\n"
    "N'utilise pas « incertain » pour conserver un texte qui porte sur une "
    "situation différente. Une simple proximité de vocabulaire, de personnes "
    "ou de lieux ne rend pas un article applicable. Une mesure de prévention "
    "d'un dommage éventuel ne répond pas, à elle seule, à une demande de "
    "réparation d'un dommage déjà réalisé. Compare aussi la chaîne causale : "
    "le bien ou le fait que le texte désigne comme CAUSE doit correspondre à "
    "la cause décrite dans les faits; le seul fait que le bien endommagé soit "
    "semblable ne suffit pas. Pour chaque article, indique aussi la "
    "cause_du_texte et la cause_des_faits. Mets cause_compatible à false si "
    "le texte désigne une cause concrète différente; mets-la à true si elle "
    "correspond ou si la règle ne vise qu'une cause juridique générale. Un "
    "statut applicable est interdit lorsque cause_compatible vaut false. "
    "Compare enfin l'OPÉRATION juridique : le texte doit répondre à la "
    "question effectivement posée. N'utilise pas « incertain » pour une "
    "exception, une défense ou une limitation qui ne devient pertinente que "
    "si un nouveau fait ou une nouvelle prétention, absents de la demande, "
    "survenait plus tard.\n\n"
    "Un article applicable ou incertain ne permet pas d'affirmer une issue "
    "certaine : les conditions non établies doivent rester conditionnelles. "
    "Pour un article incertain, remplis faits_requis uniquement avec les faits "
    "qui sont à la fois absents des FAITS et exigés explicitement par une phrase "
    "du texte. Pour chaque fait, copie un court passage_source exactement présent "
    "dans l'article et formule une question compréhensible. N'ajoute aucun fait "
    "tiré de ta mémoire juridique. Pour applicable ou incompatible, faits_requis "
    "est une liste vide. "
    "Réponds uniquement par {\"articles\":[{\"article\":\"numéro\","
    "\"cause_du_texte\":\"...\",\"cause_des_faits\":\"...\","
    "\"cause_compatible\":true,\"statut\":\"applicable|incertain|incompatible\","
    "\"motif\":\"une phrase\",\"faits_requis\":[{\"id\":\"identifiant_court\","
    "\"description\":\"fait à établir\",\"question\":\"question à poser\","
    "\"passage_source\":\"extrait exact\"}]}]}. Chaque article fourni doit apparaître "
    "exactement une fois.")


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


@dataclass(frozen=True)
class ApplicabiliteArticle:
    """Compatibilité factuelle d'un texte officiel avant sa rédaction."""

    statut: str
    motif: str = ""
    faits_requis: tuple[dict[str, str], ...] = ()


def _normaliser_source(value: str) -> str:
    # Repliement partagé : l'apostrophe typographique du corpus officiel
    # DOIT être repliée, sinon un extrait recopié avec une apostrophe
    # droite est déclaré absent de sa propre source.
    return fold_text(value)


def _faits_requis_source_bornes(
        raw: Any, article_text: str,
        rejets: Optional[list[dict[str, str]]] = None,
) -> tuple[dict[str, str], ...]:
    """Conserve uniquement les faits reliés à un extrait exact de la source.

    Un extrait non retrouvé dans la source est écarté — mais l'abandon est
    CONSIGNÉ dans ``rejets`` : c'est le symptôme d'un relecteur qui
    paraphrase au lieu de citer, et il faut pouvoir le voir.
    """
    if not isinstance(raw, list):
        return ()
    source = _normaliser_source(article_text)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()[:240]
        passage = str(item.get("passage_source") or "").strip()[:500]
        if not description or not passage:
            continue
        if _normaliser_source(passage) not in source:
            if rejets is not None:
                rejets.append({
                    "motif": "passage_source_absent_de_la_source",
                    "description": description,
                    "passage_source": passage[:200],
                })
            continue
        fact_id = re.sub(
            r"[^a-z0-9]+", "_",
            _normaliser_source(str(item.get("id") or description)),
        ).strip("_")[:80]
        if not fact_id or fact_id in seen:
            continue
        question = str(item.get("question") or "").strip()[:300]
        if not question:
            question = f"Pouvez-vous préciser si {description}?"
        result.append({
            "id": fact_id,
            "description": description,
            "question": question,
            "passage_source": passage,
        })
        seen.add(fact_id)
    return tuple(result)


def _prevention_incompatible_avec_fait_realise(texte: str, faits: str) -> bool:
    """Écarte une règle seulement préventive après un dommage accompli.

    C'est un contrôle de temporalité commun à toute disposition : il ne
    dépend ni d'un article ni d'un type de bien. Un texte qui prévoit aussi
    la réparation demeure disponible pour le juge de pertinence.
    """
    texte_normalise = fold_text(texte, collapse_whitespace=False)
    faits_normalises = fold_text(faits, collapse_whitespace=False)
    return bool(
        _FAIT_DEJA_REALISE_RE.search(faits_normalises)
        and _MESURE_PREVENTIVE_RE.search(texte_normalise)
        and not _REPARATION_RE.search(texte_normalise)
    )


def articles_incompatibles_deterministes(
        textes: dict[str, str], faits: str = "") -> dict[str, ApplicabiliteArticle]:
    """Exclut les incompatibilitÃ©s temporelles lisibles sans modÃ¨le.

    Cette garantie reste active si le relecteur LLM est indisponible ou ne
    parvient pas Ã  classifier toute une grande liste de textes.
    """
    return {
        numero: ApplicabiliteArticle(
            statut="incompatible",
            motif=("le texte ne pr\u00e9voit qu'une mesure pr\u00e9ventive alors que "
                   "les faits d\u00e9crivent un dommage d\u00e9j\u00e0 r\u00e9alis\u00e9"))
        for numero, texte in textes.items()
        if _prevention_incompatible_avec_fait_realise(texte, faits)
    }


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
        # Extraits écartés faute d'être retrouvés dans leur source, au
        # dernier appel de sélection. Diagnostic, jamais une décision.
        self.derniers_rejets_passage: list[dict[str, str]] = []

    def disponible(self) -> bool:
        return bool(self.client) and not self.offline

    def verifier(self, reponse: str, textes: dict[str, str],
                 faits: str = "") -> list[VerdictAffirmation]:
        """Les affirmations de droit de la réponse tiennent-elles ?

        UN seul appel, sur la réponse ENTIÈRE. Une fenêtre autour de chaque
        citation manquait la phrase fautive dès qu'elle en débordait : une
        affirmation peut se trouver dans la phrase suivante et n'entrer dans
        aucune fenêtre.

        Les articles cités sans texte récupéré ne sont pas jugés ici : c'est
        le contrôle d'ancrage par numéro qui les couvre déjà.
        """
        if not self.disponible() or not (reponse or "").strip() or not textes:
            return []
        cites = {n for n, _, _ in _numeros_cites(reponse)} & set(textes)
        if not cites:
            return []
        temporal = [
            VerdictAffirmation(
                article=numero,
                affirmation=affirmation_autour(reponse, debut, fin),
                soutenue=False,
                motif=("le texte ne pr\u00e9voit qu'une mesure pr\u00e9ventive alors "
                       "que les faits d\u00e9crivent un dommage d\u00e9j\u00e0 r\u00e9alis\u00e9"),
            )
            for numero, debut, fin in _numeros_cites(reponse)
            if (numero in cites
                and _prevention_incompatible_avec_fait_realise(
                    textes[numero], faits))
        ]
        if temporal:
            return temporal
        return self._juger(reponse, {n: textes[n] for n in sorted(cites)},
                           faits)

    def selectionner_articles(self, textes: dict[str, str],
                              faits: str = "") -> dict[str, ApplicabiliteArticle]:
        """Classe les textes par lots afin de conserver une rÃ©ponse complÃ¨te."""
        if not self.disponible() or not textes:
            return {}
        items = list(textes.items())
        resultat: dict[str, ApplicabiliteArticle] = {}
        for start in range(0, len(items), _ARTICLE_SELECTION_BATCH_SIZE):
            lot = dict(items[start:start + _ARTICLE_SELECTION_BATCH_SIZE])
            selection = self._selectionner_lot(lot, faits)
            # Un lot incomplet ne permet aucune conclusion sur ses articles,
            # mais ne doit jamais annuler les incompatibilites deja etablies
            # dans les lots complets precedents.
            if set(selection) == set(lot):
                resultat.update(selection)
        return resultat

    def _selectionner_lot(self, textes: dict[str, str],
                          faits: str = "") -> dict[str, ApplicabiliteArticle]:
        """Classe les textes avant rédaction, sans introduire de disposition.

        Une sélection vide signifie que le service est indisponible ou que sa
        sortie est illisible : l'appelant conserve alors les preuves plutôt
        que d'écarter silencieusement un texte officiel.
        """
        if not self.disponible() or not textes:
            return {}
        corpus = "\n\n".join(
            f"TEXTE OFFICIEL DE L'ARTICLE {numero} :\n{texte}"
            for numero, texte in sorted(textes.items()))
        contenu = f"FAITS DÉCRITS PAR LA PERSONNE :\n{(faits or '').strip()}\n\n{corpus}"
        try:
            brut = self.client.complete_json(
                self.role,
                [{"role": "system", "content": _SELECTION_SYSTEME},
                 {"role": "user", "content": contenu}],
                temperature=0.0)
        except Exception:  # noqa: BLE001 -- la conservation est le repli sûr
            return {}
        entrees = brut.get("articles") if isinstance(brut, dict) else None
        if not isinstance(entrees, list):
            return {}
        statuts = {"applicable", "incertain", "incompatible"}
        resultat: dict[str, ApplicabiliteArticle] = {}
        rejets: list[dict[str, str]] = []
        for entree in entrees:
            if not isinstance(entree, dict):
                continue
            numero = str(entree.get("article") or "").strip()
            statut = str(entree.get("statut") or "").strip().lower()
            if numero not in textes or statut not in statuts:
                continue
            if entree.get("cause_compatible") is False:
                statut = "incompatible"
            faits_requis = (
                _faits_requis_source_bornes(
                    entree.get("faits_requis"), textes[numero], rejets)
                if statut == "incertain" else ()
            )
            resultat[numero] = ApplicabiliteArticle(
                statut=statut,
                motif=str(entree.get("motif") or "").strip()[:300],
                faits_requis=faits_requis)
        for numero, texte in textes.items():
            if (numero in resultat
                    and _prevention_incompatible_avec_fait_realise(texte, faits)):
                resultat[numero] = ApplicabiliteArticle(
                    statut="incompatible",
                    motif=("le texte ne prévoit qu'une mesure préventive alors "
                           "que les faits décrivent un dommage déjà réalisé"))
        if rejets:
            # Visible plutôt que muet : un extrait introuvable dans sa propre
            # source signale un relecteur qui paraphrase, et le fait requis
            # perdu retire sa formulation conditionnelle à la réponse.
            self.derniers_rejets_passage = list(rejets)
            print(
                f"[assertion_grounding] {len(rejets)} extrait(s) écarté(s) : "
                "passage_source introuvable dans le texte officiel",
                flush=True)
        else:
            self.derniers_rejets_passage = []
        # Une sélection partielle ne doit jamais faire disparaître un texte
        # officiel par accident : on ne filtre que si tous les articles ont
        # été classés exactement une fois.
        return resultat if set(resultat) == set(textes) else {}

    def _juger(self, reponse_finale: str, textes: dict[str, str],
               faits: str) -> list[VerdictAffirmation]:
        articles = ", ".join(sorted(textes))
        corpus = "\n\n".join(
            f"TEXTE OFFICIEL DE L'ARTICLE {n} :\n{t}"
            for n, t in textes.items())
        contenu = corpus
        if (faits or "").strip():
            contenu += f"\n\nFAITS DÉCRITS PAR LA PERSONNE :\n{faits.strip()}"
        contenu += f"\n\nRÉPONSE À VÉRIFIER :\n{reponse_finale}"
        try:
            brut = self.client.complete_json(
                self.role,
                [{"role": "system", "content": _SYSTEME},
                 {"role": "user", "content": contenu}],
                temperature=0.0)
        except Exception as exc:                       # noqa: BLE001
            # Ne pas pouvoir vérifier n'est pas une vérification réussie.
            return [VerdictAffirmation(
                article=articles, affirmation="(réponse entière)",
                soutenue=False, motif=f"{type(exc).__name__}: {exc}"[:160],
                echec_technique=True)]
        if not isinstance(brut, dict) or "non_soutenues" not in brut:
            return [VerdictAffirmation(
                article=articles, affirmation="(réponse entière)",
                soutenue=False, motif="réponse du vérificateur illisible",
                echec_technique=True)]
        entrees = brut.get("non_soutenues")
        if not isinstance(entrees, list):
            return [VerdictAffirmation(
                article=articles, affirmation="(réponse entière)",
                soutenue=False, motif="liste de verdicts illisible",
                echec_technique=True)]
        verdicts = [VerdictAffirmation(
            article=articles, affirmation="(réponse entière)", soutenue=True)]
        for entree in entrees:
            if not isinstance(entree, dict):
                continue
            verdicts.append(VerdictAffirmation(
                article=articles,
                affirmation=str(entree.get("affirmation") or "")[:300],
                soutenue=False,
                motif=str(entree.get("motif") or "")[:200]))
        return verdicts[1:] or verdicts


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
