# -*- coding: utf-8 -*-
"""Reference metadata for the human 40-situation evaluation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


CATEGORY_RANGES = (
    (1, 12, 1, "article_should_suffice"),
    (13, 20, 2, "regulation_only"),
    (21, 28, 3, "article_and_regulation"),
    (29, 40, 4, "jurisprudence_necessary"),
)


def _category(scenario_id: int) -> tuple[int, str]:
    for first, last, number, label in CATEGORY_RANGES:
        if first <= scenario_id <= last:
            return number, label
    raise ValueError(f"scenario_id out of range: {scenario_id}")


def _extract_markdown(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^##\s+(\d+)\s*$", text))
    descriptions: dict[int, str] = {}
    for index, match in enumerate(matches):
        scenario_id = int(match.group(1))
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():block_end]
        description = block.split("**Question :**", 1)[0].strip()
        description = re.sub(r"\s+", " ", description)
        if description:
            descriptions[scenario_id] = description
    return descriptions


def _extract_pdf(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    try:
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    except Exception:
        return {}
    # Reuse the same parser without writing a sidecar file.
    matches = list(re.finditer(r"(?m)^\s*(?:Situation\s*)?(\d{1,2})\s*$", text))
    descriptions: dict[int, str] = {}
    for index, match in enumerate(matches):
        scenario_id = int(match.group(1))
        if not 1 <= scenario_id <= 40:
            continue
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():block_end]
        # The PDF repeats the two human-entry labels after every situation.
        # They are instructions, not part of the situation shown in the UI.
        block = re.split(r"\bta question\s*:\s*", block, maxsplit=1, flags=re.IGNORECASE)[0]
        # Category headings and the closing instructions can occur between the
        # last item of one category and the next numeric marker.
        block = re.split(r"\bCatégorie\s+\d+\s*[—-]", block, maxsplit=1, flags=re.IGNORECASE)[0]
        block = re.sub(r"\s+", " ", block).strip(" :\n\r")
        if block:
            descriptions[scenario_id] = block
    return descriptions


def load_reference_scenarios(phase1_root: Path | None = None) -> list[dict[str, Any]]:
    root = phase1_root or Path(__file__).resolve().parents[3]
    descriptions: dict[int, str] = {}
    for candidate in (
        root / "40_situations.md.pdf",
        root / "questions_reelles" / "40_situations.md.pdf",
    ):
        descriptions = _extract_pdf(candidate)
        if len(descriptions) >= 40:
            break
    if len(descriptions) < 40:
        descriptions = _extract_markdown(
            root / "questions_reelles" / "situations_a_completer.md")
    scenarios: list[dict[str, Any]] = []
    for scenario_id in range(1, 41):
        category, label = _category(scenario_id)
        scenarios.append({
            "scenario_id": scenario_id,
            "category": category,
            "category_label": label,
            "scenario_description": descriptions.get(
                scenario_id, f"Situation {scenario_id} de la référence des 40 situations."),
        })
    return scenarios
