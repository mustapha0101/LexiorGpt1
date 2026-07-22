from agentic_generation.schemas import (
    GenerationMetadata, GroundingEntry, Message, Role, ToolObservation,
    TrainingTrajectory,
)


def test_training_trajectory_round_trip():
    observation = ToolObservation(
        tool_name="get_ccq_articles", server="quebec",
        arguments={"start_article": 1457}, raw_response="fixture",
        normalized_response="Article 1457", mock=True,
    ).finalize_hash()
    trajectory = TrainingTrajectory(
        scenario_id="s1", scenario_family_id="f1", request_type="exact_text_retrieval",
        messages=[
            Message(role=Role.user, content="Article 1457?"),
            Message(role=Role.assistant, content='<tool_call>\n{"name":"get_ccq_articles","arguments":{"start_article":1457}}\n</tool_call>'),
            Message(role=Role.tool, name="get_ccq_articles", content="Article 1457"),
            Message(role=Role.assistant, content="Réponse prudente."),
        ],
        tool_trace=[observation],
        grounding=[GroundingEntry(tool_name="get_ccq_articles", content_hash=observation.content_hash)],
        generation_metadata=GenerationMetadata(tool_catalog_hash="abc"),
    )
    restored = TrainingTrajectory.model_validate_json(trajectory.model_dump_json())
    assert restored.dataset_type == "agentic_legal_intermediate"
    assert restored.final_answer() == "Réponse prudente."
    assert restored.group_key().startswith("fam:f1")
