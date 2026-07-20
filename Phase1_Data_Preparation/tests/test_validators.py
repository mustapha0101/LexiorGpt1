from agentic_generation.schemas import Message, Role, ToolObservation, TrainingTrajectory
from agentic_generation.validators import validate_trajectory


def _trajectory(tool_name="get_ccq_articles", arguments=None):
    arguments = arguments or {"start_article": 1457}
    obs = ToolObservation(tool_name=tool_name, arguments=arguments, raw_response="Article 1457",
                          normalized_response="Article 1457", mock=True).finalize_hash()
    return TrainingTrajectory(
        scenario_id="s", scenario_family_id="f", request_type="article_ccq_precis",
        expected_jurisdiction="Québec", resolved_jurisdiction="Québec",
        messages=[
            Message(role=Role.user, content="Texte officiel de 1457"),
            Message(role=Role.assistant, content=(
                f'<tool_call>\n{{"name":"{tool_name}","arguments":{{"start_article":1457}}}}\n</tool_call>')),
            Message(role=Role.tool, name=tool_name, content="Article 1457"),
            Message(role=Role.assistant, content="Article 1457"),
        ], tool_trace=[obs])


def test_valid_mock_trajectory_is_accepted_offline(catalog):
    result = validate_trajectory(_trajectory(), catalog, allow_mock=True)
    assert result.valid, result.errors


def test_mock_is_rejected_in_real_dataset(catalog):
    result = validate_trajectory(_trajectory(), catalog, allow_mock=False)
    assert not result.valid
    assert any("mock" in error for error in result.errors)


def test_unpaired_tool_message_is_rejected(catalog):
    row = _trajectory()
    row.messages.pop(1)
    result = validate_trajectory(row, catalog, allow_mock=True)
    assert not result.valid
    assert any("sans tool_call" in error for error in result.errors)


def test_invented_url_is_rejected(catalog):
    row = _trajectory()
    row.messages[-1].content += " https://invented.example/decision"
    result = validate_trajectory(row, catalog, allow_mock=True)
    assert any("URL absente" in error for error in result.errors)


def test_invented_article_is_rejected(catalog):
    row = _trajectory()
    row.messages[-1].content += " L’article 9999 règle définitivement la question."
    result = validate_trajectory(row, catalog, allow_mock=True)
    assert any("article 9999 absent" in error for error in result.errors)
