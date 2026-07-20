import asyncio
import json

import pytest

from agentic_generation.mcp_executor import (
    MCPExecutionError, MCPExecutor, MockMCPTransport,
    compact_legal_search_response,
)
from agentic_generation.schemas import ToolCall


def test_mock_executor_marks_and_normalizes_fixture(catalog):
    transport = MockMCPTransport({
        "get_ccq_articles": "Article 1457\nhttps://example.test/source"
    })
    executor = MCPExecutor(catalog, transport, max_retries=0)
    obs = asyncio.run(executor.aexecute(ToolCall(name="get_ccq_articles",
                                                arguments={"start_article": 1457})))
    assert obs.ok and obs.mock
    assert obs.raw_response is not None
    assert obs.content_hash
    assert obs.source_urls == ["https://example.test/source"]


def test_invalid_arguments_are_rejected_before_transport(catalog):
    transport = MockMCPTransport({"get_ccq_articles": "ne doit pas être appelé"})
    executor = MCPExecutor(catalog, transport)
    with pytest.raises(MCPExecutionError):
        asyncio.run(executor.aexecute(ToolCall(name="get_ccq_articles",
                                               arguments={"start_article": "1457"})))
    assert transport.calls == []


def test_error_is_structured_and_never_fabricated(catalog):
    transport = MockMCPTransport({"get_ccq_articles": RuntimeError("panne")})
    executor = MCPExecutor(catalog, transport, max_retries=0)
    obs = asyncio.run(executor.aexecute(ToolCall(name="get_ccq_articles",
                                                arguments={"start_article": 1457})))
    assert not obs.ok
    assert obs.raw_response is None
    assert obs.error and "RuntimeError" in obs.error


def test_long_document_is_truncated_explicitly(catalog):
    transport = MockMCPTransport({"fetch_document": {"citation_en": "2020 SCC 5",
                                                      "unofficial_text_en": "x" * 500}})
    executor = MCPExecutor(catalog, transport, max_response_chars=80, max_retries=0)
    obs = asyncio.run(executor.aexecute(ToolCall(
        name="fetch_document", arguments={"citation": "2020 SCC 5", "start_char": 0,
                                          "end_char": 500})))
    assert obs.truncated
    assert "TRONQUÉ" in obs.normalized_response
    assert "2020 SCC 5" in obs.citations


def test_legal_search_compaction_keeps_valid_titles_and_citations():
    raw = json.dumps({"results": [{
        "dataset": "LEGISLATION-FED", "name_en": "Bank Act",
        "name_fr": "Loi sur les banques", "citation_fr": "LC 1991, c 46",
        "snippet": "x" * 20_000, "upstream_license": "y" * 20_000,
    }]})
    compact = compact_legal_search_response(raw)
    assert compact["results"][0]["name_en"] == "Bank Act"
    assert compact["results"][0]["citation_fr"] == "LC 1991, c 46"
    assert "snippet" not in compact["results"][0]
