# -*- coding: utf-8 -*-
"""Fixtures MCP marquées et exclusivement utilisées par dry-run/tests offline."""


def _article_fixture(code):
    def build(arguments):
        number = arguments.get("start_article", 1)
        return (
            f"Article {number}\n"
            f"Texte officiel {code} simulé pour test offline; aucune valeur juridique."
        )
    return build


def _legal_search_fixture(arguments):
    query = str(arguments.get("query", ""))
    if arguments.get("doc_type") == "laws":
        statutes = {
            "Bankruptcy and Insolvency Act": (
                "RSC 1985, c B-3", "LRC 1985, c B-3",
                "Bankruptcy and Insolvency Act", "Loi sur la faillite et l’insolvabilité",
            ),
            "Bank Act": (
                "SC 1991, c 46", "LC 1991, c 46", "Bank Act", "Loi sur les banques",
            ),
        }
        citation_en, citation_fr, name_en, name_fr = statutes.get(
            query, ("RSC 1985, c B-3", "LRC 1985, c B-3",
                    query or "Federal Act", query or "Loi fédérale"))
        return {"results": [{
            "citation_en": citation_en, "citation_fr": citation_fr,
            "dataset": "LEGISLATION-FED", "name_en": name_en, "name_fr": name_fr,
            "url_fr": "https://laws-lois.justice.gc.ca/fra/",
        }]}
    return {"results": [{
        "citation_en": "2020 SCC 5", "dataset": "SCC",
        "name_en": "Fixture case", "url_en": "https://decisions.scc-csc.ca/",
    }]}

MOCK_MCP_FIXTURES = {
    "semantic_search_ccq": (
        "Article 1726 — résultat sémantique simulé sur la garantie et les vices cachés.\n"
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991"
    ),
    "semantic_search_cpc": "Article 110 — résultat sémantique simulé CPC.",
    "get_ccq_articles": _article_fixture("CCQ"),
    "search_ccq_keywords": (
        "Article 1726 — résultat de fixture sur la garantie et les vices cachés.\n"
        "https://www.legisquebec.gouv.qc.ca/fr/document/lc/CCQ-1991"
    ),
    "get_cpc_articles": _article_fixture("CPC"),
    "search_cpc_keywords": "Article 110 — fixture de résultat CPC.",
    "search_quebec_regulations": (
        "Q-2, r. 17.1 — règlement de fixture.\n"
        "https://www.legisquebec.gouv.qc.ca/fr/document/rc/Q-2,%20r.%2017.1"
    ),
    "get_quebec_regulation": (
        "Q-2, r. 17.1 — texte de règlement simulé pour test offline.\n"
        "https://www.legisquebec.gouv.qc.ca/fr/document/rc/Q-2,%20r.%2017.1"
    ),
    "search_quebec_jurisprudence": (
        "2020 QCCA 100 — décision de fixture, sans portée juridique.\n"
        "https://www.canlii.org/fr/qc/qcca/doc/2020/2020qcca100/2020qcca100.html"
    ),
    "get_quebec_legal_info": (
        "Information d’entrée en vigueur simulée.\n"
        "https://www.legisquebec.gouv.qc.ca/fr/contenu/eevlois"
    ),
    "coverage": {"results": [{"dataset": "SCC", "number_of_documents": 10000}]},
    "search_legal_documents": _legal_search_fixture,
    "fetch_document": {
        "citation_en": "Bank Act, SC 1991, c 46",
        "url_en": "https://laws-lois.justice.gc.ca/eng/acts/b-1.01/",
        "unofficial_text_en": "Fixture de document fédéral; aucune valeur juridique.",
    },
}
