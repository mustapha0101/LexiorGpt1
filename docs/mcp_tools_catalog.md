# MCP Tools Catalog

Catalogue des outils MCP inspectés dans cette session.

## Serveurs

| Serveur canonique | Observé comme | Type | URL |
| --- | --- | --- | --- |
| lexior-legisquebec-mcp | lexior-ccq | sse | https://lexior-ccq-mcp.onrender.com/sse |
| Canadian Legal Data / a2aj | a2aj | streamable-http | https://mcp.a2aj.ca/mcp |
| local-rag | local-rag | local | — |

## Outils RAG locaux

| canonicalName | Description | Exemple minimal |
| --- | --- | --- |
| semantic_search_ccq | Recherche dense puis reranking hybride dans le CCQ; les candidats doivent être vérifiés par `get_ccq_articles`. | `{ "query": "Mon voisin a déplacé la clôture sur mon terrain", "top_k": 8 }` |
| semantic_search_cpc | Recherche dense puis reranking hybride dans le CPC; les candidats doivent être vérifiés par `get_cpc_articles`. | `{ "query": "Comment assigner un témoin à comparaître?", "top_k": 8 }` |

## Outils Québec

| canonicalName | vscodeObservedName | Description | Exemple minimal |
| --- | --- | --- | --- |
| get_ccq_articles | mcp_lexior-legisq_get_ccq_articles | Récupère le texte officiel d'un ou plusieurs articles du Code civil du Québec (CCQ). | `{ "start_article": 1457 }` |
| search_ccq_keywords | mcp_lexior-legisq_search_ccq_keywords | Recherche par mots-clés dans le texte des articles du Code civil du Québec (CCQ). | `{ "keyword": "contrat" }` |
| search_cpc_keywords | mcp_lexior-legisq_search_cpc_keywords | Recherche par mots-clés dans le texte des articles du Code de procédure civile du Québec (CPC / C-25.01). | `{ "keyword": "signification" }` |
| get_cpc_articles | mcp_lexior-legisq_get_cpc_articles | Récupère le texte officiel d'un ou plusieurs articles du Code de procédure civile du Québec (CPC / C-25.01). | `{ "start_article": 1 }` |
| get_quebec_legal_info | mcp_lexior-legisq_get_quebec_legal_info | Récupère des métadonnées et tableaux de modifications, entrées en vigueur et dispositions non en vigueur de LégisQuébec. | `{ "type": "decision" }` |
| search_quebec_regulations | mcp_lexior-legisq_search_quebec_regulations | Recherche sémantique parmi les règlements du Québec sur LégisQuébec. | `{ "keyword": "environnement" }` |
| get_quebec_regulation | mcp_lexior-legisq_get_quebec_regulation | Récupère le contenu complet d'un règlement spécifique ou d'un acte législatif sur LégisQuébec à partir de son URL. | `{ "url": "https://www.legisquebec.gouv.qc.ca/fr/document/rc/Q-2,%20r.%2017.1" }` |
| search_quebec_jurisprudence | mcp_lexior-legisq_search_quebec_jurisprudence | Recherche de la jurisprudence (décisions de justice) québécoise sur CanLII et SOQUIJ. | `{ "query": "responsabilité civile" }` |

## Outils Canadian Legal Data / a2aj

| canonicalName | vscodeObservedName | Description | Exemple minimal |
| --- | --- | --- | --- |
| coverage | mcp_canadian_lega_coverage | Get dataset coverage for Canadian case law and legislation showing earliest/latest dates and document counts. | `{}` |
| fetch_document | mcp_canadian_lega_fetch_document | Retrieve full text of Canadian legal documents by citation. | `{ "citation": "2020 SCC 5", "doc_type": "cases", "output_language": "en" }` |
| search_legal_documents | mcp_canadian_lega_search_legal_documents | Search Canadian case law and legislation. | `{ "query": "contract", "search_type": "name", "doc_type": "cases", "size": 1, "search_language": "en", "sort_results": "default" }` |

## Notes

Les structures de réponse détaillées et les schémas JSON complets se trouvent dans [docs/mcp_tools_catalog.json](docs/mcp_tools_catalog.json).
