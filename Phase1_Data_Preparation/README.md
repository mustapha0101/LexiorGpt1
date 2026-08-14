# Lexior — démonstration live

Ce dossier contient uniquement la démonstration interactive du système
multi-agent Lexior : API FastAPI, graphe LangGraph, recherche juridique
hybride, outils MCP et interface React.

## Contenu

```text
apps/chat-web/                 interface de chat et journal des agents
configs/agentic_generation.yaml  configuration du mode live
data/agentic/rag_index/       index sémantique CCQ/CPC (NON versionné)
data/agentic/cache/mcp-real/  cache local des recherches MCP
src/lexior/agent_graph/       orchestration LangGraph
src/lexior/agentic/           modèles, prompts, RAG et exécution MCP
src/lexior/api/app.py         API live : /health et /api/chat
src/lexior/services/          services juridiques et garde-fous
tests/                        garde-fous hors ligne (aucun appel modèle)
```

## Index sémantique

`data/` est exclu du dépôt : un clone neuf ne reçoit **aucun** index et
`/health` répond alors `rag.loaded: false` avec un repli lexical MCP. Pour
le reconstruire (corpus Hugging Face + embeddings OpenAI, ~0,01 USD) :

```powershell
..\.venv\Scripts\python.exe -m pip install -e ".[index]"
..\.venv\Scripts\python.exe -m lexior.agentic.build_rag_index
```

Le fichier `.env`, le catalogue MCP et la configuration MCP restent à la
racine du dépôt parent :

```text
../.env
../.mcp.json
../docs/mcp_tools_catalog.json
```

## Lancer la démonstration

Depuis `Phase1_Data_Preparation`, lancer l’API :

```powershell
..\.venv\Scripts\python.exe -m uvicorn lexior.api.app:app --port 8000 --reload
```

Dans un deuxième terminal :

```powershell
cd .\apps\chat-web
npm install
npm run dev
```

Ouvrir ensuite `http://127.0.0.1:5173`.

Vérification rapide de l’API :

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

La réponse doit indiquer `status: ok` et `rag.loaded: true`.

## Tests

La suite est hors ligne : aucun appel modèle, aucun appel MCP, aucune clé
d'API requise.

```powershell
..\.venv\Scripts\python.exe -m pytest tests
```

## Parcours présenté

1. L’utilisateur pose une question juridique.
2. Le graphe détermine la juridiction et demande une clarification bloquante
   seulement lorsqu’elle est nécessaire.
3. Le planner choisit les outils de recherche.
4. Le RAG hybride retrouve des articles candidats.
5. Les outils MCP récupèrent les textes officiels.
6. Les agents vérifient la pertinence, construisent la règle et produisent une
   réponse fondée avec ses limites.

Les pipelines de génération de datasets, notebooks, déploiements RunPod,
benchmarks et interfaces d’évaluation ont été retirés de cette copie de démo.
