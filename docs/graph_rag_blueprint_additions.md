# Évolution de l'Architecture : De Vector RAG à Graph RAG pour le CCQ

Pour dépasser les limites inhérentes à la recherche vectorielle classique sur un corpus aussi interdépendant et complexe que le Code civil du Québec (CCQ), l'architecture de **LexiorGPT** prévoit une transition vers une approche **Graph RAG** (ou hybride Vector + Graph). Cette stratégie permet de résoudre le défi des renvois législatifs implicites et explicites (ex. : *"Sous réserve de l'article X..."*).

Dans un Vector RAG traditionnel, la requête *"conditions de la responsabilité civile"* récupère l'article 1457 CCQ grâce à sa proximité sémantique, mais omet les articles traitant des exceptions spécifiques ou de la responsabilité du fait d'autrui situés plus loin, en raison de termes sémantiques divergents.

---

## 1. La Modélisation des Nœuds (Nodes)

Le découpage du code respecte scrupuleusement la hiérarchie législative et conceptuelle :
*   **Nœud Article** : Le texte littéral et le numéro de l'article (ex. : *Art. 1457*).
*   **Nœud Hiérarchique** : Le contexte structurel (Livre, Titre, Chapitre, Section, ex. : *Livre Cinquième - Des Obligations*).
*   **Nœud Concept** : Les entités juridiques extraites par NER/LLM (ex. : *Responsabilité extracontractuelle*, *Faute*, *Préjudice*).

---

## 2. La Création des Arêtes (Edges / Relations)

Un script de parsing structurel crée des liaisons explicites entre les nœuds :
*   `[Art. 1457]` ── **FAIT_REFERENCE_A** ──> `[Art. 1458]`
*   `[Art. 1457]` ── **APPARTIENT_A** ──> `[Livre Cinquième]`
*   `[Art. 1457]` ── **CONCERNE_LE_CONCEPT** ──> `[Responsabilité civile]`

---

## 3. Résolution des requêtes dans LexiorNotebook (Le moment de vérité)

Lorsqu'une question est soumise à LexiorGPT via le serveur MCP :
1.  **Recherche Initiale** : L'outil MCP interroge la base de données vectorielle (Vector Search) pour identifier les premiers articles pertinents.
2.  **Expansion du Graphe** : Le système parcourt le graphe de relations (ex. : dans Neo4j) à partir de ces articles pour extraire les dépendances liées par les relations **FAIT_REFERENCE_A**.
3.  **Construction du Contexte** : Les textes de tous les articles interconnectés sont agrégés dans un contexte structuré, fournissant au LLM local (8B/32B) une vision exhaustive des exceptions et règles applicables avant de générer la réponse.

---

## 4. Forces de l'Architecture et Points de Vigilance Techniques

L'architecture souveraine modélisée pour LexiorGPT répond directement aux exigences de confidentialité et de précision requises par la *LegalTech*.

### 4.1 Forces Majeures
*   **Découplage Connaissance/Logique** : La connaissance brute (lois et jurisprudence) est déchargée sur le serveur MCP et les bases externes (Vector & Graph DB). Le modèle fine-tuné (QLoRA) se concentre uniquement sur la méthodologie de raisonnement (IRAC) et le suivi d'instructions.
*   **Souveraineté des Données** : L'utilisation de modèles compacts optimisés (Qwen-2.5-32B AWQ / LLaMA-3 8B) permet un déploiement local performant (Consumer GPU / Disques rapides NVMe) garantissant la confidentialité des dossiers clients.
*   **Flexibilité MCP/ReAct** : La mise à jour des lois n'exige pas de réentraînement (fine-tuning) du modèle ; il suffit de mettre à jour l'index de la base de données pointée par l'outil MCP.

### 4.2 Points de Vigilance pour l'Implémentation
*   **Granularité et Tagging des Métadonnées** : La structure hautement relationnelle du CCQ impose un étiquetage strict lors de l'ingestion afin que l'expansion du graphe ne sature pas la fenêtre de contexte de 4096 tokens.
*   **Qualité et Alignement du Corpus CoT** : Le dataset d'apprentissage de la réflexion CoT (IRAC) doit utiliser le lexique du droit civil québécois pour éviter le glissement vers la terminologie de la *common law* anglo-saxonne.
*   **Fiabilité de la Syntaxe JSON (Model Constraints)** : Les modèles locaux de taille moyenne peuvent fluctuer dans le formatage de leurs appels MCP. L'intégration de décodages contraints par grammaire (Grammar-constrained decoding de type fichiers `.gbnf` ou JSON schema strict sous vLLM) is indispensable pour éliminer les erreurs de syntaxe de l'agent.
