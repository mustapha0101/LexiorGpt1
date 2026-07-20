import copy


def test_catalog_has_only_canonical_tools(catalog):
    assert len(catalog.tools) == 13
    assert "get_ccq_articles" in catalog.tools
    assert "semantic_search_ccq" in catalog.tools
    assert "semantic_search_cpc" in catalog.tools
    assert all(not name.startswith("mcp_") for name in catalog.tools)


def test_call_validation_rejects_unknown_missing_type_and_enum(catalog):
    assert catalog.validate_call("outil_invente", {})
    assert catalog.validate_call("get_ccq_articles", {})
    assert catalog.validate_call("get_ccq_articles", {"start_article": "1457"})
    assert catalog.validate_call("get_ccq_articles", {"start_article": 1457, "extra": 1})
    assert catalog.validate_call("coverage", {"doc_type": "articles"})
    assert not catalog.validate_call("get_ccq_articles", {"start_article": 1457})


def test_live_schema_drift_is_detected(catalog):
    live = {name: copy.deepcopy(spec.input_schema) for name, spec in catalog.tools.items()}
    live["get_ccq_articles"]["properties"]["start_article"]["type"] = "string"
    problems = catalog.compare_with_live(live)
    assert any("type divergent" in problem for problem in problems)
