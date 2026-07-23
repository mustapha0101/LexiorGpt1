# -*- coding: utf-8 -*-

"""
Export des trajectoires pour le fine-tuning.

Le dataset maître conserve ``thinking`` comme champ séparé sur les objets
Message.  L'export fine-tuning fusionne ce champ dans un bloc
``<thinking>...</thinking>`` au début du contenu assistant.
"""

from __future__ import annotations

from .schemas import Message, Role, TrainingTrajectory


# ---------------------------------------------------------------------------
# Fusion thinking → contenu
# ---------------------------------------------------------------------------

def merge_thinking_into_content(message: Message) -> Message:
    """Fusionne le champ *thinking* dans le contenu pour un message assistant.

    Retourne un **nouveau** Message (pas de mutation de l'original).
    Les messages non-assistant ou sans thinking sont retournés tels quels.
    """
    if message.role is not Role.assistant:
        return message
    if not message.thinking:
        return message
    merged_content = f"<thinking>\n{message.thinking}\n</thinking>\n{message.content}"
    return Message(
        role=message.role,
        content=merged_content,
        name=message.name,
        thinking=None,
    )


# ---------------------------------------------------------------------------
# Conversion ChatML
# ---------------------------------------------------------------------------

def format_for_finetuning(trajectory: TrainingTrajectory) -> list[dict]:
    """Convertit les messages d'une trajectoire en format ChatML (list[dict])."""
    result: list[dict] = []
    for msg in trajectory.messages:
        if msg.role is Role.system:
            result.append({"role": "system", "content": msg.content})
        elif msg.role is Role.user:
            result.append({"role": "user", "content": msg.content})
        elif msg.role is Role.tool:
            result.append({"role": "tool", "name": msg.name, "content": msg.content})
        elif msg.role is Role.assistant:
            merged = merge_thinking_into_content(msg)
            result.append({"role": "assistant", "content": merged.content})
    return result


# ---------------------------------------------------------------------------
# Masque de perte
# ---------------------------------------------------------------------------

def build_loss_mask(messages: list[dict]) -> list[int]:
    """Construit un masque de perte pour chaque message ChatML.

    - system / user / tool → ``-100`` (masqué, pas entraîné)
    - assistant → ``1`` (inclus dans la perte)
    """
    mask: list[int] = []
    for msg in messages:
        if msg["role"] == "assistant":
            mask.append(1)
        else:
            mask.append(-100)
    return mask


# ---------------------------------------------------------------------------
# Export complet
# ---------------------------------------------------------------------------

def export_trajectory_for_training(trajectory: TrainingTrajectory) -> dict:
    """Exporte une trajectoire sous forme de dict prêt pour le fine-tuning."""
    formatted = format_for_finetuning(trajectory)
    return {
        "messages": formatted,
        "loss_mask": build_loss_mask(formatted),
        "scenario_id": trajectory.scenario_id,
        "request_type": trajectory.request_type,
    }
