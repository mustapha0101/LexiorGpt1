# Phase 1 — réparation du flux retrieval/reasoning live

## 1. Causes racines

Le flux mélangeait la justification interne du reviewer avec la question
posée à l’utilisateur, sélectionnait parfois le premier article conditionnel
du dictionnaire et traitait une recherche jurisprudentielle comme une preuve.
La garde de progression arrêtait aussi la lecture dès qu’un seul article était
marqué applicable. Enfin, le planner pouvait extraire une URL d’un résultat
rejeté, dépasser son budget de décisions et le traducteur SSE rejouait tout
`tool_history` après une interruption.

## 2. Clarifications factuelles

`lexior.services.article_review` produit un profil déterministe pour chaque
texte : `retrieval_group` (`primary`, `contextual`, `incompatible`),
`legal_operation`, `rule_roles`, `missing_fact_keys` et une priorité stable.
Une clarification ne peut venir que d’un article `primary`, de statut
`conditionally_applicable`, avec au moins un fait manquant. La sélection est
triée par priorité décroissante, rang de reranking puis identifiant d’article.
Le `reason` reste interne.

Le contrat en attente contient `clarification_id`, `category`, `fact_keys`,
`question`, `answer_type`, `source_articles` et `status`. Les faits déjà
présents dans le dossier, les déclarations précédentes et l’historique des
clarifications sont exclus avant la génération de la question.

## 3. Réponses courtes

`handle_clarification` interprète `oui`, `non`, les formulations prudentes et
`je ne sais pas` relativement au contrat en attente. Il inscrit un objet de
fait avec sa valeur, sa source `user_clarification`, son niveau de confiance
`asserted_by_user` et son statut (`affirmative`, `negative` ou `unresolved`).
La réponse courte n’est pas ajoutée comme une nouvelle description autonome.

## 4. Suffisance législative

`assess_legislative_sufficiency` déduit les rôles nécessaires du dossier et ne
compte que les textes `primary` applicables ou conditionnels. Les articles
contextuels, incompatibles ou non revus ne couvrent aucun rôle principal.
L’objet retourné conserve les rôles requis/couverts/manquants, les articles
par groupe et `should_fetch_next_batch`. La progression peut donc atteindre un
candidat situé au-delà du premier lot, jusqu’à 20 candidats selon la
configuration, puis s’arrête lorsque les rôles essentiels sont couverts.

## 5. Requête jurisprudentielle

`PlannerAgent._build_quebec_case_law_query` assemble de manière déterministe
le dossier complet : question active, description, faits structurés, étape de
l’événement, dommage, opérations juridiques et seuls les articles
`primary` applicables ou conditionnels. Les articles contextuels,
incompatibles, rejetés ou non revus n’y sont pas ajoutés.

## 6. Candidat, résultat accepté et décision vérifiée

Une recherche sémantique ou jurisprudentielle produit des candidats, jamais
une preuve. Le gate conserve seulement les candidats québécois pertinents et
dotés d’une URL dans `usable_case_sources`, avec le statut
`candidates_pending_fetch`. Seul ce statut autorise `get_quebec_regulation`.
Une décision récupérée doit contenir une citation québécoise, un texte complet
et passer l’évaluation de preuve avant d’entrer dans `case_law_verified` et
`prior_evidence`. Un résultat `irrelevant`, sans URL ou incomplet n’est jamais
fetché ni cité.

## 7. Chaînes d’outils

La chaîne québécoise est `search_quebec_jurisprudence` →
`get_quebec_regulation`. Une proposition `fetch_document` après une recherche
québécoise est redirigée vers cette chaîne ou finalisée sans source acceptée.
La recherche de règlements québécois conserve sa propre provenance. La chaîne
fédérale `search_legal_documents` → `fetch_document` reste autorisée.

## 8. Budgets live

Les budgets sont séparés dans `AgenticConfig` et dans l’état : appels d’outils,
clarifications, reformulations de jurisprudence et décisions du planner.
L’épuisement live force `final_answer` avec les preuves disponibles et le
motif interne `planner_budget_exhausted`; il ne produit pas de rejet technique.
Le mode dataset conserve son rejet strict pour les contrôles de trajectoire.

## 9. Échec de jurisprudence

Une seule reformulation est permise. Après un second résultat vide, échoué ou
irrelevant, le système conserve les textes législatifs retenus et finalise une
réponse conditionnelle. Il ne déclare pas automatiquement la responsabilité et
n’invente pas de décision.

## 10. Persistance et SSE

Les faits, clarifications, revues législatives, preuves officielles et
décisions complètes vérifiées sont recopiés dans `case_context` pour les tours
suivants. Les candidats non fetchés n’y entrent pas comme preuves.

`StreamTranslator` reçoit le nombre d’observations déjà présentes lors d’une
reprise et n’émet que les nouveaux appels. Les événements `tool_result`
annoncent `preview_truncated`, la longueur de l’aperçu et les nombres de
candidats/articles. Une nouvelle reformulation avec des arguments différents
reste donc visible.

## 11. Invariants de sécurité

La logique de production ne mappe aucun mot-clé vers un numéro d’article.
Les textes préventifs ne deviennent pas des recours postérieurs, les articles
contextuels ne soutiennent pas la conclusion, les résultats de recherche ne
sont pas citables et les décisions non vérifiées ne sont pas persistées comme
preuves. Le contrat final et le fallback limité aux sources restent les
dernières protections.

## 12. Limites restantes

Le profil d’article est déterministe et lexical : il structure le flux mais ne
remplace pas une revue juridique humaine. Les faits saisis par l’utilisateur
restent des assertions, non des faits judiciairement établis. La persistance
du graphe live est actuellement celle du checkpointer de l’application; les
JSONL de session servent à l’audit et ne remplacent pas ce checkpointer après
un redémarrage du processus. Les résultats externes dépendent enfin de la
qualité et de la disponibilité des serveurs MCP.
