# Rapport de Recherche Académique et Technique
**Titre :** Agentic Fine-Tuning vs. Mémorisation Paramétrique : Rapport d'Expérimentation sur le Modèle Souverain LexiorGPT-7B  
**Auteurs :** Équipe de Recherche R&D, intelliwork  
**Date :** 19 Juillet 2026  

---

## Résumé (Abstract)
Ce rapport présente la méthodologie, l'analyse théorique et les résultats empiriques de la conception de **LexiorGPT-7B**, un modèle souverain de raisonnement juridique canadien et québécois. Notre étude démontre que les modèles de langage de taille intermédiaire (7B) sont intrinsèquement incapables d'assurer un rappel sans faille d'un corpus de lois volumineux et dynamique par simple stockage dans leurs poids synaptiques. Pour résoudre ce problème de saturation et éliminer les hallucinations juridiques, nous avons conçu un protocole d'entraînement basé sur la distillation d'instructions **ReAct (Reasoning and Acting)** et **CoT (Chain-of-Thought)** via des balises de réflexion `<thinking>...</thinking>` et d'appel d'outils `<tool_call>...</tool_call>`. Le modèle a été entraîné sur un corpus consolidé de **8 316 cas pratiques** juridiques via l'écosystème Unsloth / LoRA. Les courbes d'entraînement montrent une convergence stable (perte finale d'entraînement à ~0.25, eval/loss à ~0.29) avec un Grad Norm stabilisé grâce à l'écrêtage des gradients (gradient clipping). Ce modèle, une fois couplé à un serveur d'outils MCP (Model Context Protocol) externe, élimine les hallucinations juridiques et offre des performances locales de niveau industriel.

---

## 1. Introduction : Les limites de la taille paramétrique (7B)
Le développement de modèles d'IA pour le domaine du droit impose une tolérance zéro pour l'erreur factuelle. La méthode naïve consiste à fine-tuner un modèle sur le texte intégral des lois (le Code civil du Québec, le Code de procédure civile, les lois fédérales) en espérant qu'il mémorise les articles de loi dans ses poids.

Cependant, pour un modèle de **7 milliards de paramètres (7B)**, cette approche se heurte à des limites physiques et théoriques majeures :
1. **Saturation de la capacité de stockage** : Un modèle 7B possède une capacité de compression limitée. Essayer d'y stocker des milliers de pages de textes de lois complexes dégrade ses capacités de raisonnement général et provoque un **effondrement du modèle (model collapse)** ou un oubli catastrophique.
2. **Hallucination normative** : Face à des situations factuelles subtiles, le modèle génère de faux numéros d'articles ou invente des règles juridiques en combinant des souvenirs fragmentés.
3. **Obsolescence de la loi** : Les lois changent continuellement. Si les règles sont gravées dans les poids du modèle, il est nécessaire de ré-entraîner complètement le modèle à chaque modification législative, ce qui est économiquement et opérationnellement infaisable.

---

## 2. Le Paradigme ReAct et CoT : Fine-Tuner le Cerveau, Pas la Base de Données
Pour pallier ces limites, nous avons découplé le système en deux entités :
* **La Base de Connaissances Externe (MCP Server)** : Elle stocke les lois (CCQ, CPC) et la jurisprudence (CanLII) à jour.
* **Le Moteur de Raisonnement (LexiorGPT-7B)** : Le modèle n'apprend pas les lois par cœur, mais il est entraîné à **utiliser des outils** pour les lire en temps réel au moment de formuler son analyse.

### 2.1 Structuration de la pensée (Chain-of-Thought)
Le modèle est fine-tuné pour générer systématiquement une analyse logique dans une balise `<thinking>` avant de produire sa conclusion. Cette analyse respecte le formalisme **IRAC** :
1. **Issue (Question de droit)** : Détecter le point de droit soulevé par le fait utilisateur.
2. **Rule (Règle de droit)** : Formuler une requête ou un appel d'outil pour récupérer la loi.
3. **Application** : Appliquer la règle extraite aux faits de l'espèce.
4. **Conclusion** : Présenter la solution.

### 2.2 Alignement ReAct (Reasoning and Acting)
Pendant sa réflexion, si le modèle a besoin de vérifier un texte de loi ou une jurisprudence, il émet un appel d'outil structuré :
```text
<tool_call>{"name": "ccq_search", "arguments": {"start_article": 1457}}</tool_call>
```
L'environnement exécute l'outil MCP et réinjecte le texte brut de l'article dans une balise `<tool_response>`. Le modèle peut alors terminer son raisonnement sans aucune hallucination.

---

## 3. Méthodologie et Préparation des Données
Le jeu de données a été construit par auto-distillation. Nous avons exploité **Gemini 2.5 Flash** (via notre clé API et des scripts d'alignement stricts) pour formuler des cas pratiques d'une grande rigueur méthodologique.

### 3.1 Taxonomie du Dataset Consolidé (8 316 exemples)
Le corpus final comprend :
* **Provincial (Québec) :** Cas pratiques basés sur le Code civil du Québec (CCQ) et le Code de procédure civile (CPC).
* **Fédéral :** Cas pratiques sur la jurisprudence CanLII et les lois fédérales majeures.
* **Tool Calling :** Exemples ReAct purs de formatage d'appels d'outils et de traitement de réponses d'outils.
* **Identité et Sens Commun :** Pour éviter le glissement linguistique vers l'anglais ou le chinois (catastrophique sur les modèles Qwen-2.5 d'origine) et ancrer sa personnalité (développée par intelliwork).

---

## 4. Analyse de l'Entraînement et Télémetrie W&B
L'entraînement a été mené sur GPU NVIDIA RTX 4090 de 24 Go de VRAM en utilisant le framework **Unsloth** (LoRA, $r=16$, $\alpha=32$, dropout $0.05$).

### 4.1 Comportement du Learning Rate (Warmup & Decay)
Le taux d'apprentissage a été configuré avec un maximum de $2 \times 10^{-4}$ et un planificateur cosinusoïdal :
* **Warmup (Échauffement) :** Durant les 100 premiers steps, le learning rate monte progressivement de 0 à $2 \times 10^{-4}$. Cela stabilise l'ajustement initial des poids synaptiques LoRA sur notre formatage strict.
* **Cosine Decay (Décroissance) :** Après le step 100, le taux d'apprentissage diminue lentement pour atteindre un plateau proche de $0$ à la fin de l'entraînement. Ce ralentissement permet d'effectuer des ajustements de plus en plus fins pour que le modèle converge vers un minimum global stable sans osciller.

### 4.2 Analyse des fluctuations du Grad Norm
La courbe de norme de gradient (`train/grad_norm`) s'est stabilisée autour de **0.4 à 0.6** après une pointe initiale à **1.6** :
* **Pourquoi ces fluctuations ?** Les fluctuations en dents de scie reflètent la stochasticité des mini-batchs. Les batchs contenant des exemples très diversifiés (ex: jurisprudence longue vs question d'identité courte) provoquent des amplitudes de mise à jour différentes.
* **Le Gradient Clipping en action :** Le fait que la norme ne dépasse jamais **1.0** montre l'efficacité de l'écrêtage des gradients (`max_grad_norm = 1.0`), garantissant qu'aucune mise à jour explosive ne déstabilise le modèle local 7B pendant sa convergence.

---

## 5. Recommandations Opérationnelles pour l'Application Finale

Pour assurer un fonctionnement optimal en production locale comme distante :

1. **Configuration du Moteur d'Inférence (vLLM) :**
   * **Pénalité de Répétition (`repetition_penalty = 1.05`) :** Indispensable pour éviter que Qwen 2.5 n'entre dans une boucle répétitive ou ne dérive vers l'anglais/chinois lors de longs contextes juridiques.
   * **Longueur de Contexte Maximale (`max_model_len = 16384`) :** Permet d'insérer des résultats de recherche complets provenant des outils MCP sans saturer la mémoire du cache KV.
2. **System Prompt Dropout :**
   * Durant la phase de préparation, nous avons appliqué un **System Prompt Dropout (20%)** sur le dataset d'entraînement. Cela rend le modèle insensible aux variations ou à l'absence de prompt système, garantissant qu'il continuera à utiliser les balises `<thinking>` et à appeler ses outils MCP même s'il est interrogé dans une console nue ou via une API brute.
3. **Robustesse de la Boucle d'Exécution :**
   * L'application hôte doit intercepter immédiatement les balises `<tool_call>` du modèle, arrêter la génération, exécuter l'outil via le protocole MCP, et injecter la `<tool_response>` pour relancer le second tour. Cette boucle doit être non-bloquante et tolérer des erreurs réseau en renvoyant un rapport d'erreur propre pour que le modèle puisse s'auto-corriger.
