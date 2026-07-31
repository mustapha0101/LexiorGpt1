# -*- coding: utf-8 -*-

from lexior.agent_graph.nodes.validate_final import (
    _articles_cites,
    _source_bounded_fallback,
)


def test_source_bounded_fallback_reproduces_only_selected_official_text():
    result = _source_bounded_fallback(
        {
            "1457": "Toute personne a le devoir de respecter les règles de conduite.",
            "1465": "Le gardien d'un bien est tenu de réparer le préjudice.",
        },
        ["1457"],
    )

    assert "Article 1457" in result
    assert "respecter les règles de conduite" in result
    assert "Article 1465" not in result
    assert "sont réunies" in result


def test_articles_cites_extracts_only_article_references():
    assert _articles_cites(
        "Selon l'article 1457, puis l'art. 1465, la date 2026 ne suffit pas."
    ) == ["1457", "1465"]
