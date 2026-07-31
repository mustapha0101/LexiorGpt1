# Recherche juridique live — Phase 1

Le chat maintient un dossier par `thread_id`. Le tour courant conserve son
propre budget d'outils; le dossier conserve uniquement les faits explicitement
fournis, les clarifications catégorisées et les textes officiels déjà lus.
Les résultats de recherche sémantique et les simples extraits de jurisprudence
ne deviennent jamais des preuves persistantes.

## Séquence

1. Le planner lit le schéma réel du catalogue. En chat, il retire seulement
   les champs absents du schéma et expose cette correction dans l'événement
   `tool_call`; une valeur connue mais invalide reste une erreur récupérable.
2. La recherche sémantique classe des candidats. Elle ne constitue pas une
   source citée.
3. Les articles sont récupérés par lots (`initial_article_fetch_k`, puis
   `article_fetch_batch_size`), dans l'ordre renvoyé par le RAG, jusqu'à
   `max_articles_per_issue`. Une revue marque chaque texte `applicable`,
   `conditionally_applicable`, `incompatible` ou `unreviewed`.
4. Une règle conditionnelle déclenche une clarification factuelle distincte de
   la clarification de juridiction. La réponse ne peut alors présenter une
   conclusion comme certaine.
5. Après une règle revue et des faits suffisants, la jurisprudence est cherchée
   avec les dispositions retenues et la description factuelle complète. Les
   décisions demeurent candidates tant que leur contenu/citation n'est pas
   vérifié.

Le reranker distingue `primary`, `contextual` et `rejected`. La réserve de
rappel garde des candidats dans la liste, mais ne bloque plus leur rang : un
article initialement loin peut devenir premier.

## Vérification

Exécuter depuis `Phase1_Data_Preparation` :

```powershell
python -m pytest -q
npm --prefix apps/chat-web run build
```

Les tests couvrent notamment la normalisation de schéma live, la provenance de
plage, les lots progressifs, la promotion d'un candidat au rang 17 et la
transparence SSE.
