# Politique de routage agentique

Le Planner choisit exactement une action : `ask_clarification`, `call_tool`,
`final_answer` ou `cannot_conclude`. Un appel invalide est rejeté avant le MCP.

| Demande | Route minimale |
|---|---|
| Article CCQ connu | `get_ccq_articles` puis arrêt |
| Article CPC connu | `get_cpc_articles` puis arrêt |
| Sujet CCQ | `semantic_search_ccq` (question complète) → `get_ccq_articles` |
| Sujet CPC | `semantic_search_cpc` (question complète) → `get_cpc_articles` |
| Cas civil québécois | CCQ, puis jurisprudence seulement si les faits le justifient |
| Règlement inconnu | `search_quebec_regulations` → URL retournée vers `get_quebec_regulation` |
| Droit fédéral | `search_legal_documents` → citation retournée vers `fetch_document` |
| Couverture incertaine/vide | `coverage`, sans appel systématique |
| Juridiction ou faits critiques manquants | clarification avant recherche |
| Hors droit | aucun outil |

La jurisprudence est justifiée si l'utilisateur demande des décisions, si une
notion ouverte doit être appliquée à des faits ou si des exceptions peuvent
changer l'issue. Elle est interdite pour la simple copie d'un article.

Les recherches CCQ/CPC ne dépendent plus d'un mot-clé choisi par le Teacher.
Le garde-fou remplace systématiquement l'argument par toute la conversation
utilisateur pertinente. L'index fait une sélection dense, reranke les
candidats avec BM25, puis le Teacher réordonne uniquement les numéros déjà
récupérés; le texte officiel reste obligatoirement récupéré par MCP.

`fetch_document` utilise une section connue ou des plages `start_char/end_char`.
Le contexte est borné, la réponse brute reste hors prompt et toute troncature
est déclarée. La réponse finale ne conclut jamais sur une partie non récupérée.
