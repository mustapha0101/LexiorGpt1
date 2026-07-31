# -*- coding: utf-8 -*-
"""Les traces publiques ne doivent pas exposer le raisonnement brut du modèle."""

from lexior.agent_graph.events import StreamTranslator


def test_decision_event_omits_unverified_model_reasoning():
    translator = StreamTranslator()
    events = list(translator.translate_chunk({
        "validate_plan": {
            "step": 3,
            "resolved_jurisdiction": "Québec",
            "latest_decision": {
                "decision": "call_tool",
                "next_tool": "get_ccq_articles",
                "arguments": {"articles": [1457]},
                "thinking_text": (
                    "Cette hypothèse provisoire contient une conclusion "
                    "juridique qui n'a pas encore été vérifiée."
                ),
            },
        },
    }))

    decision = next(event for event in events if event["type"] == "decision")
    assert decision["tool"] == "get_ccq_articles"
    assert "thinking" not in decision


def test_tool_event_exposes_live_schema_correction():
    translator = StreamTranslator()
    list(translator.translate_chunk({
        "plan": {
            "last_tool_normalization": {
                "tool": "search_quebec_jurisprudence",
                "removed_fields": ["legal_terms"],
                "remaining_arguments": {"query": "dommages"},
            },
        },
    }))
    events = list(translator.translate_chunk({
        "execute_tool": {
            "tool_history": [type("Observation", (), {
                "tool_name": "search_quebec_jurisprudence",
                "arguments": {"query": "dommages"},
                "normalized_response": "[]",
                "ok": True,
            })()],
        },
    }))
    call = next(event for event in events if event["type"] == "tool_call")
    assert call["schema_correction"] == ["legal_terms"]
