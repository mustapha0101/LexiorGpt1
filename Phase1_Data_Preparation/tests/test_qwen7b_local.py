# -*- coding: utf-8 -*-
"""Optional smoke test for the configured local Qwen reference."""

import os

import pytest

from lexior.agentic.config import load_config


@pytest.mark.local_model
def test_qwen7b_reference_is_configured_when_local_run_is_requested():
    if not os.environ.get("LEXIOR_QWEN7B_URL"):
        pytest.skip("LEXIOR_QWEN7B_URL absent: serveur Qwen local non lancé")
    config = load_config(overrides={"evidence_first_enabled": True})
    model = os.environ.get("QWEN_MODEL") or os.environ.get(
        "TEACHER_MODEL") or config.teacher.model
    assert "qwen" in model.casefold() or "7b" in model.casefold(), model
