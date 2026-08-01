# Mode d’évaluation humaine — 40 situations

## Principe

Le mode `Évaluation — 40 situations` s’utilise dans le chat habituel. La personne écrit chaque question elle-même. Le mode n’envoie jamais la description de référence, ne génère pas de question et ne simule pas de clarification.

La référence est affichée uniquement dans le panneau pour rappeler la situation évaluée. Les catégories et les notes humaines sont des métadonnées et ne sont pas transmises au planner, au writer, au retrieval ni aux outils juridiques.

## Utilisation

1. Sélectionner `Mode évaluation — 40 situations`.
2. Cliquer sur `Commencer une nouvelle évaluation` ou saisir un `run_id` pour reprendre une évaluation.
3. Écrire la question dans la zone normale du chat.
4. Répondre normalement aux clarifications.
5. Après la réponse finale, remplir l’attente, l’évaluation globale et, si souhaité, la route, les critères de qualité et les notes.
6. Cliquer sur `Enregistrer et terminer cette situation`, puis passer à la suivante.
7. Utiliser `Exporter / ouvrir le fichier` pour ouvrir la source JSON unique.

Le bouton de complétion d’une situation exige une attente et une évaluation globale. Une situation peut être interrompue puis reprise. Le retour à une situation déjà visitée est enregistré comme une reprise et lui attribue un nouveau thread.

## Fichier source de vérité

Les runs sont enregistrés dans :

```text
data/evaluations/human_40/<run_id>.json
```

Un seul fichier contient les 40 objets `scenarios`, même si une seule situation a commencé. Chaque scénario contient notamment :

- `conversation` : messages humains et réponses de LexiorGPT;
- `tool_calls` : arguments, résultat ou aperçu, hash, classification, métadonnées et durées;
- `graph_events` et `validation_events` : étapes publiques du graphe et validations;
- `timing` : première réponse, réponse finale, outils et revue humaine;
- `expected_answer_note` et `human_evaluation` : saisie humaine;
- `technical_summary` : statistiques déterministes, sans jugement automatique de la qualité juridique.

Les écritures utilisent un fichier temporaire puis `os.replace`, sous verrou par run. Les résultats d’outils sous `HUMAN_EVAL_MAX_INLINE_TOOL_RESULT_CHARS` (20 000 par défaut) sont conservés; les plus gros gardent un aperçu, la taille et un SHA-256. Les champs sensibles éventuels sont redigés.

## API

```text
POST  /api/evaluations/human-40/runs
GET   /api/evaluations/human-40/runs/{run_id}
GET   /api/evaluations/human-40/runs/{run_id}/file
POST  /api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}/start
PATCH /api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}
POST  /api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}/events
POST  /api/evaluations/human-40/runs/{run_id}/scenarios/{scenario_id}/complete
PATCH /api/evaluations/human-40/runs/{run_id}  # interruption explicite
POST  /api/evaluations/human-40/runs/{run_id}/complete
```

En mode chat, chaque requête doit porter `evaluation_run_id`, `evaluation_scenario_id` et le `thread_id` appartenant au scénario. Une requête sans ces identifiants est refusée; une requête normale reste inchangée.

## Source des descriptions

La source officielle est `40_situations.md.pdf`, maintenant versionnée à la racine de Phase 1. Le chargeur la privilégie également dans `questions_reelles` et conserve un fallback Markdown si le PDF est absent. Les catégories suivent l’ordre du document : 1–12, 13–20, 21–28 et 29–40. Les libellés « ta question » et « ce que tu attendais » du PDF sont retirés de la description affichée; la question saisie reste exclusivement celle de la personne.

## Limites connues

- Aucun rapport Markdown séparé n’est généré : le JSON reste la seule source de vérité.
- Le script frontend ne fournit actuellement ni `npm test` ni `npm run typecheck`; `npm run build` exécute néanmoins TypeScript (`tsc -b`) et Vite.
- La qualité juridique finale est volontairement laissée à l’évaluation humaine; `accepted` reste une acceptation technique du chat.
