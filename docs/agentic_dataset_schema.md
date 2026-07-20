# Schéma du dataset agentique

Chaque ligne est validée par `TrainingTrajectory` (Pydantic), avec
`schema_version=agentic-1.0` et `dataset_type=agentic_legal`.

Les champs principaux sont les identifiants de scénario et de famille, le type
de demande, les juridictions attendue/résolue, `messages`, `tool_trace`,
`grounding`, `generation_metadata` et `quality`. Une observation d'outil
enregistre le nom canonique, le serveur, les arguments, l'heure, les réponses
brute et normalisée, le hash de contenu, les URLs, citations, la troncature et
l'erreur éventuelle. `mock=true` est permis uniquement en tests/dry-run.

Les messages suivent strictement ChatML : `system`, `user`, `assistant`, `tool`.
Un tour assistant peut contenir un court `<thinking>` et un unique
`<tool_call>`. Le message `tool` suivant porte exactement la réponse normalisée
de l'observation correspondante.

Les releases contiennent `train.jsonl`, `validation.jsonl`, `test.jsonl`,
`agentic_eval.jsonl`, `dataset_info.json`, `generation_manifest.json`,
`audit_report.json` et `README.md`. Les groupes sont déterminés par famille de
scénario, citations, article principal ou URL source afin qu'une même source ne
traverse pas les splits.

Les données historiques portent `legacy_legal_federal` ou
`legacy_legal_quebec`. Elles sont exclues par défaut. Les données d'identité
restent incluses selon la configuration YAML.
