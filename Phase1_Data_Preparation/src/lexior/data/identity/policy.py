#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Politique d'identité publique de LexiorGPT — source unique de vérité.

Importé par la génération (generate_identity_data.py), l'audit
(audit_training_dataset.py) et l'évaluation (Phase3_Evaluation/evaluate_identity.py)
pour que les trois appliquent exactement la même règle.

Portée : ce module régit les RÉPONSES de l'assistant. Il ne régit ni la
configuration interne du dépôt ni le chargement du modèle : les identifiants
techniques réels doivent rester présents là où ils sont nécessaires
(Phase2_FineTuning/train_hf.py, deploy_vllm.py, etc.).
"""

import re
import unicodedata

# --- Nommage canonique (§13) ----------------------------------------------
PRODUCT_NAME = "LexiorGPT"
DEVELOPER_NAME = "IntelliWork"

CORE_DESCRIPTION = (
    "Je suis LexiorGPT, un assistant d’intelligence juridique développé par "
    "IntelliWork, spécialisé en droit canadien et québécois."
)

# --- Termes techniques interdits dans une CIBLE assistant ------------------
# L'utilisateur a le droit de nommer n'importe quel modèle dans SA question ;
# c'est la réponse de l'assistant qui ne doit jamais les reprendre.
#
# Attention aux frontières de mots : "\bgpt\b" ne correspond PAS à l'intérieur
# de « LexiorGPT » (aucune frontière entre "r" et "G"), mais correspond bien à
# « GPT-4 ». « ChatGPT » exige donc son propre motif.
_FORBIDDEN_MODEL_TERMS = [
    r"qwen\w*", r"alibaba", r"tongyi",
    r"chatgpt", r"gpt-?\d", r"\bgpt\b", r"openai",
    r"llama\d*", r"\bmeta\b",
    r"mistral", r"mixtral",
    r"claude", r"anthropic",
    r"gemini", r"\bbard\b", r"deepmind",
    r"deepseek", r"yi-\d+", r"falcon-\d+", r"phi-?\d",
    r"grok", r"\bxai\b",
    r"command-?r", r"cohere",
]

# --- Affirmations fausses interdites (§ « Forbidden assistant behaviour ») --
# Nier tout modèle sous-jacent, ou prétendre à un entraînement « de zéro »,
# est aussi problématique que la divulgation : c'est une fausse provenance.
_FORBIDDEN_CLAIM_PATTERNS = [
    (r"(entraîn|entrain|form)\w*\s+(complètement|entièrement|totalement)?\s*(à partir de|de)\s+z[ée]ro",
     "prétend un entraînement de zéro"),
    (r"from\s+scratch", "prétend un entraînement de zéro"),
    (r"(aucun|pas de|sans)\s+(modèle|architecture)\s+(sous-jacent|de base|fondation)",
     "nie l'existence d'un modèle sous-jacent"),
    (r"je\s+n['e ]\s*(ai|utilise)\s+pas\s+de\s+modèle", "nie l'existence d'un modèle sous-jacent"),
    (r"(entraîné|développé|conçu)\s+(de\s+manière\s+)?(entièrement\s+)?autonome",
     "prétend un entraînement autonome"),
    (r"je\s+ne\s+sais\s+pas\s+(quel|de\s+quel)\s+mod[èe]le", "avoue ignorer son modèle"),
    (r"i\s+don'?t\s+know\s+what\s+model", "avoue ignorer son modèle"),
    (r"(mon|le)\s+prompt\s+(système|caché)", "évoque un prompt caché"),
    (r"(system|hidden)\s+prompt", "évoque un prompt caché"),
]

_FORBIDDEN_MODEL_RE = re.compile("|".join(f"(?:{t})" for t in _FORBIDDEN_MODEL_TERMS), re.IGNORECASE)
_FORBIDDEN_CLAIM_RES = [(re.compile(p, re.IGNORECASE), why) for p, why in _FORBIDDEN_CLAIM_PATTERNS]

# Un bloc de raisonnement n'a rien à faire dans une réponse d'identité (§1.4).
_THINKING_RE = re.compile(r"</?thinking>", re.IGNORECASE)


def normalize(text):
    """Normalise pour la déduplication : casse, accents, espaces, ponctuation."""
    t = unicodedata.normalize("NFD", (text or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def find_forbidden_terms(text):
    """Termes techniques interdits présents dans un texte assistant."""
    return sorted({m.group(0).lower() for m in _FORBIDDEN_MODEL_RE.finditer(text or "")})


def find_forbidden_claims(text):
    """Affirmations de provenance fausses présentes dans un texte assistant."""
    return sorted({why for rx, why in _FORBIDDEN_CLAIM_RES if rx.search(text or "")})


# Catégories où la réponse DOIT nommer le produit : on y demande explicitement
# « qui es-tu », ou on tente de lui substituer une autre identité. Ailleurs, la
# politique n'impose rien de positif — « J'ai été développé par IntelliWork. »
# est une bonne réponse à « Qui t'a développé ? », et forcer la description
# complète partout produirait exactement la langue de brochure que la consigne
# interdit.
# La langue (fr / en / mixed / fr_qc / fr_typo) est une dimension distincte,
# portée par le champ `language` : une question « qui es-tu » en anglais reste
# de la catégorie direct_identity.
CATEGORIES_REQUIRING_PRODUCT_NAME = {
    "direct_identity", "product_name", "false_premise", "rename_attempt",
    "prompt_injection", "one_word_model", "indirect_identity",
    "multi_turn_consistency",
}

CATEGORIES_REQUIRING_DEVELOPER_NAME = {"developer_identity"}


def validate_assistant_text(text, require_product=False, require_developer=False):
    """Valide une CIBLE assistant d'identité. Retourne la liste des violations.

    Liste vide = conforme.

    Par défaut, seules les INTERDICTIONS sont vérifiées (termes techniques,
    fausses affirmations, bloc de raisonnement). Les exigences positives sont
    dépendantes de la catégorie et passées par l'appelant.
    """
    violations = []
    text = text or ""

    if not text.strip():
        violations.append("réponse vide")
        return violations

    for term in find_forbidden_terms(text):
        violations.append(f"terme technique interdit : « {term} »")
    for claim in find_forbidden_claims(text):
        violations.append(f"affirmation interdite : {claim}")
    if _THINKING_RE.search(text):
        violations.append("bloc <thinking> dans une réponse d'identité")
    if require_product and PRODUCT_NAME.lower() not in text.lower():
        violations.append(f"ne nomme pas {PRODUCT_NAME}")
    if require_developer and DEVELOPER_NAME.lower() not in text.lower():
        violations.append(f"ne nomme pas {DEVELOPER_NAME}")

    return violations


def echoes_user_model_name(user_text, assistant_text):
    """L'assistant reprend-il un nom de modèle introduit par l'utilisateur ?

    C'est le cas typique « Es-tu ChatGPT ? » → la réponse ne doit pas contenir
    « ChatGPT ». L'utilisateur a le droit de le nommer ; l'assistant, non.
    """
    in_user = set(find_forbidden_terms(user_text))
    in_assistant = set(find_forbidden_terms(assistant_text))
    return sorted(in_user & in_assistant)
