# -*- coding: utf-8 -*-
"""Nœuds du graphe central Lexior — un module par nœud.

``NODE_MODULES`` est l'ordre topologique de référence; ``graph.py``
construit le graphe à partir de ce registre. Chaque module expose
``NAME`` et ``run(state, ctx) -> dict``.
"""

from . import (
    analyze_facts,
    build_answer_contract,
    classify_failures,
    classify_follow_up,
    classify_request,
    classify_tool_result,
    compute_acceptance,
    execute_tool,
    export_dataset,
    generate_answer,
    handle_clarification,
    initialize,
    plan,
    reformulate_search,
    reject,
    repair_answer,
    repair_trajectory,
    resolve_jurisdiction,
    return_live_answer,
    run_critics,
    update_active_task,
    update_research_state,
    validate_final,
    validate_plan,
    verify_tool_result,
)

NODE_MODULES = (
    initialize,
    classify_request,
    classify_follow_up,
    update_active_task,
    resolve_jurisdiction,
    analyze_facts,
    plan,
    validate_plan,
    handle_clarification,
    execute_tool,
    verify_tool_result,
    classify_tool_result,
    update_research_state,
    reformulate_search,
    build_answer_contract,
    generate_answer,
    run_critics,
    classify_failures,
    repair_answer,
    repair_trajectory,
    validate_final,
    compute_acceptance,
    export_dataset,
    return_live_answer,
    reject,
)

NODE_NAMES = tuple(module.NAME for module in NODE_MODULES)

__all__ = ["NODE_MODULES", "NODE_NAMES"]
