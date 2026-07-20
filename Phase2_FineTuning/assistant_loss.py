# -*- coding: utf-8 -*-
"""ChatML strict et labels assistant-only, indépendants de TRL/Unsloth."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

STRICT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

CHATML_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>\n?",
    re.DOTALL,
)


def render_strict_chatml(messages: list[dict[str, Any]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message.get('content', '')}<|im_end|>\n"
        for message in messages
    )


def parse_strict_chatml(text: str) -> list[dict[str, str]]:
    messages = [{"role": match.group(1), "content": match.group(2)}
                for match in CHATML_RE.finditer(text)]
    if not messages or render_strict_chatml(messages) != text:
        raise ValueError("texte non conforme au ChatML strict Lexior")
    return messages


def _encode(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(ids)


def encode_messages_assistant_only(tokenizer, messages: list[dict[str, Any]],
                                   max_length: int | None = None) -> dict[str, list[int]]:
    """Encode par frontières ChatML explicites.

    L'en-tête de rôle est toujours masqué. Pour un tour assistant, son contenu
    et `<|im_end|>\n` portent la loss; toutes les parties system/user/tool sont
    étiquetées -100. Le découpage par segments rend la règle testable token par
    token et ne dépend d'aucun comportement implicite de SFTTrainer.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"rôle ChatML inconnu : {role}")
        header_ids = _encode(tokenizer, f"<|im_start|>{role}\n")
        body_ids = _encode(tokenizer, str(message.get("content", "")) + "<|im_end|>\n")
        input_ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))
        input_ids.extend(body_ids)
        labels.extend(body_ids if role == "assistant" else [-100] * len(body_ids))
    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def encode_example_assistant_only(example: dict[str, Any], tokenizer,
                                  max_length: int) -> dict[str, list[int]]:
    messages = example.get("messages")
    if not messages:
        messages = parse_strict_chatml(example["text"])
    return encode_messages_assistant_only(tokenizer, messages, max_length)


@dataclass
class AssistantOnlyDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            size = len(feature["input_ids"])
            padding = max_len - size
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
