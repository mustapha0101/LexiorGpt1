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
