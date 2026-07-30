# -*- coding: utf-8 -*-
"""validate_final — validation déterministe de la trajectoire complète.

Dataset : validation stricte (grounding, doublons, budget, séquence).
Live : les mêmes validateurs tournent en OBSERVATION (les problèmes
sont consignés dans ``validation_issues`` mais ne bloquent pas la
livraison — les bloqueurs live sont gérés par le contrat de réponse).
"""

from __future__ import annotations

import re

from typing import Any

from lexior.agentic.error_codes import ErrorCode, tag
from lexior.services.assertion_grounding import textes_recuperes
from lexior.services.modes import is_live

from ..context import GraphContext
from ..state import LexiorState, to_trajectory

NAME = "validate_final"


_RE_REGLE_ENONCEE = re.compile(
    r"\barticles?\s+\d{1,4}"                 # « l'article 1465 »
    r"|\ble\s+Code\s+civil\s+(?:du\s+Québec\s+)?(?:stipule|prévoit|dispose)"
    r"|\best\s+tenu\s+de\s+réparer"
    r"|\bsauf\s+s'?il\s+prouve", re.IGNORECASE)


def _regle_sans_source(state: LexiorState) -> list[str]:
    """Une règle énoncée alors qu'aucune source n'a été retenue.

    Le contrat porte déjà la directive « n'affirme aucune règle de fond »
    quand answer_mode vaut no_evidence. Le modèle l'a ignorée : sur une
    question de morsure de chien, il a énoncé la règle de l'article 1465
    (« sauf s'il prouve n'avoir commis aucune faute ») ET écrit « je n'ai pas
    trouvé d'articles spécifiques » dans la même réponse. Les deux ne peuvent
    pas être vraies. Une consigne de plus ne suffira pas — il en existait
    déjà une, comme « jamais de loi de mémoire » — donc on le constate.
    """
    contrat = state.get("answer_contract") or {}
    if contrat.get("answer_mode") != "no_evidence":
        return []
    reponse = state.get("final_answer") or ""
    if not _RE_REGLE_ENONCEE.search(reponse):
        return []
    return [tag(ErrorCode.ANSWER_FROM_MEMORY,
                "règle de fond énoncée alors qu'aucune preuve utilisable "
                "n'a été retenue")]


def run(state: LexiorState, ctx: GraphContext) -> dict[str, Any]:
    live = is_live(state.get("mode", ""))
    trajectory = to_trajectory(state)
    exempt = state.get("exempt_tools") or (
        ctx.services.validation.compute_exempt_tools(
            state.get("tool_history", [])))

    validation = ctx.services.validation.validate_trajectory(
        trajectory,
        allow_mock=(ctx.config.offline or ctx.config.dry_run or live),
        max_tool_calls=state.get("max_tool_calls", 4),
        exempt_tools=exempt,
    )
    validation.warnings.extend(ctx.services.validation.sequence_warnings(
        state["scenario"].request_type,
        [o.tool_name for o in state.get("tool_history", [])],
    ))
    validation.warnings.extend(_regle_sans_source(state))

    # Le NUMÉRO cité est vérifié ailleurs ; ici c'est l'AFFIRMATION qui est
    # confrontée au texte réellement récupéré. Un échec technique devient une
    # erreur : ne pas avoir pu vérifier n'est pas avoir vérifié.
    verdicts = ctx.services.assertion_grounding.verifier(
        state.get("final_answer") or "",
        textes_recuperes(state.get("tool_history", [])))
    for verdict in verdicts:
        if verdict.echec_technique:
            validation.errors.append(
                tag(ErrorCode.UNGROUNDED_ARTICLE, verdict.probleme()))
        elif not verdict.soutenue:
            validation.warnings.append(
                tag(ErrorCode.UNGROUNDED_ARTICLE, verdict.probleme()))

    return {
        "validation_result": validation,
        "validation_issues": list(validation.errors)
        + list(validation.warnings),
        "deterministic_blockers": ([] if live else list(validation.errors)),
        "deterministic_validation": bool(validation.valid),
        "exempt_tools": exempt,
    }
