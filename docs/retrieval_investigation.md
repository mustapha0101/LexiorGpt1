# Recherche d'articles : sept pistes, une retenue

Rapport de synthèse — juillet 2026

Point de départ : pour la question « mon locateur refuse que j'apporte mon
chat », le système retournait des articles du CCQ sans rapport, avec un
score de pertinence de 1,000, et ne déclenchait aucune reformulation.

Ce rapport rend compte des sept pistes explorées pour y remédier, des
mesures qui les ont tranchées, et des erreurs de méthode commises en
chemin — celles-ci comptant autant que les résultats, puisqu'elles ont
failli conduire à trois décisions erronées.

---

## 1. L'instrument de mesure

Rien n'était mesurable au départ : aucun jeu de test de recherche
n'existait.

**`tests/fixtures/retrieval_gold.jsonl`** — 52 questions annotées, dont 40
répondables et 12 sans réponse dans le corpus. Chaque article attendu est
vérifié présent dans l'index par un test. Les questions sont écrites en
langage d'usager, délibérément éloignées du vocabulaire du Code.

Les vecteurs des questions sont mis en cache
(`retrieval_gold_queries.npz`), ce qui rend toute re-mesure gratuite et
reproductible sans clé d'API.

### La ligne de base était fausse

Le jeu annoté donnait **hit@3 = 0,500**. Sur **40 articles tirés au
sort**, avec des questions écrites selon le même protocole, la même chaîne
donne :

| | 40 articles au hasard | jeu annoté |
|---|---|---|
| hit@1 | 0,175 | — |
| **hit@3** | **0,275** | **0,500** |
| hit@8 | 0,400 | 0,575 |

La cause est structurelle : les questions du jeu annoté ont été écrites en
partant d'articles choisis pour être trouvables. Le tirage au sort ramène
la substitution, le grevé, l'avis de délaissement — des matières que
personne n'aurait retenues.

**0,275 est la ligne de base réelle.** Toutes les mesures ci-dessous ont
été prises sur l'échelle optimiste et surestiment d'autant.

---

## 2. Le diagnostic

Sur les 20 questions dont la bonne réponse ne sort pas en top-3 :

| médiane | succès | échecs |
|---|---|---|
| recouvrement de vocabulaire question ↔ article | **27,3 %** | **10,0 %** |
| longueur de l'article attendu | 478 car. | 348 car. |
| numéro d'article | 1466 | 1457 |
| recouvrement **nul** | 4/20 | **8/20** |

Le numéro d'article ne distingue rien : ce n'est pas une question de
position dans le Code. Ce qui distingue, c'est que **l'usager écrit dans
le vocabulaire de sa situation et le Code dans celui de la règle**.

> « Mon voisin a laissé un vieil **arbre pourri** tomber sur mon
> **garage** »
> attendu 1457 : « Toute personne a le devoir de respecter les **règles de
> conduite**… causer un **préjudice** à autrui »
> sortis : 985 (branches et racines), 984 (fruits qui tombent), 1139
> (usufruitier qui abat des arbres)

Second motif : **le principe fondateur perd contre la disposition de
détail**. La question sur l'obligation alimentaire des parents (art. 585)
ramène trois articles sur le *calcul* de la contribution. Les articles
fondateurs sont courts et abstraits — donc génériques dans l'espace
vectoriel — alors que ce sont ceux qui s'appliquent le plus souvent.

---

## 3. Les sept pistes

### 3.1 Planchers de pertinence absolus — rejetée

La normalisation min-max ramène toujours le meilleur candidat à 1,000,
même hors sujet. Deux planchers sur l'échelle absolue ont été ajoutés
(`min_dense_score`, `min_hybrid_score`), et le score **affiché** est
désormais le score absolu — le 1,000 fabriqué a disparu des trajectoires.

Le balayage montre qu'aucun seuil ne sépare les deux populations :

| `min_dense` | faux positifs | recall@8 | questions vidées |
|---|---|---|---|
| 0,40 | 1,00 | 0,525 | 1 |
| 0,50 | 0,67 | 0,350 | 15 |
| 0,60 | **0,08** | 0,150 | **32 sur 40** |

Les scores des questions sans réponse (0,407–0,670) chevauchent ceux des
répondables (0,245–0,656). La pire question sans réponse dépasse la
meilleure question répondable.

**Retenu malgré tout** : `min_dense_score = 0.40`, qui améliore le
classement (MRR 0,340 → 0,397) en écartant du bruit. Il ne fait rien
contre les faux positifs, et c'était son objet initial.

### 3.2 BGE-M3, modèle multilingue — rejetée

Hypothèse : `text-embedding-3-small` étant anglo-centré, un modèle
multilingue franchirait l'écart de vocabulaire juridique français.

| | OpenAI sans rerank | OpenAI avec | BGE sans | BGE avec |
|---|---|---|---|---|
| faux positifs | 1,000 | 0,333 | 1,000 | 0,333 |
| MRR | 0,397 | **0,490** | 0,312 | 0,443 |
| hit@3 | 0,500 | **0,525** | 0,325 | 0,475 |
| recall@8 | 0,525 | 0,512 | 0,463 | 0,475 |

Moins bon sur toutes les métriques. Et sur le cas décisif — l'article 1594
(« mise en demeure ») pour une question disant « lettre d'avertissement » —
BGE échoue identiquement, même à `candidate_k=400`.

**L'écart n'est donc pas un problème de couverture linguistique.** Aucun
bi-encodeur généraliste ne relie une paraphrase familière à un terme
technique avec lequel elle ne co-occurre jamais.

Licence MIT, usage commercial libre. Coût : 35 min de CPU, 2,3 Go.

### 3.3 Recentrage des vecteurs — rejetée

Hypothèse : une part de la compression des scores vient de ce que tous les
articles partagent ; la soustraire devrait les étaler.

**L'étalement est confirmé et double** :

| | aucun | global | par code |
|---|---|---|---|
| écart rang 1 → 400 | 0,1661 | 0,2785 | **0,2929** |
| écart-type | 0,0486 | 0,0824 | **0,0858** |

**Il n'achète rien.** À plancher désactivé, pour isoler le classement :

| | aucun | global | par code |
|---|---|---|---|
| MRR | 0,3398 | 0,3401 | 0,3369 |
| hit@3 | 0,4000 | 0,4000 | 0,3750 |
| faux positifs | 1,000 | 1,000 | 1,000 |

Les deux populations s'étalent du même facteur : le recouvrement relatif
ne bouge pas. La compression ne cachait pas une séparation — il n'y en a
pas.

*Erreur évitée de justesse* : mesurée d'abord avec le plancher calibré sur
l'échelle **non** recentrée, la qualité s'effondrait (MRR 0,221, 23
questions vidées). Comparer des configurations d'échelles différentes à
seuil absolu fixe ne mesure que le seuil.

### 3.4 Texte indexé — rejetée après contrôle

`search_text()` préfixait chaque article de quatre étiquettes — titre,
libellé, domaine, taxonomie. Elles pèsent **22 % du texte embarqué** en
médiane, jusqu'à 71 % sur un article court, pour **16 taxonomies
distinctes** sur 4 278 articles.

À `dense_weight = 1.0`, qui isole le canal des embeddings, retirer les
étiquettes améliore tout : MRR 0,407 → 0,438, hit@3 0,475 → 0,525, et les
faux positifs bougent pour la première fois (1,000 → 0,917).

Une dissociation apparaît : **les étiquettes nuisent aux vecteurs mais
aident BM25**, qui exploite le numéro d'article et le domaine. En
découplant les canaux, le meilleur point atteint hit@3 = 0,6250 contre
0,5000.

**Le contrôle de surajustement l'annule.** Jeu coupé en deux moitiés,
30 configurations balayées sur la moitié A, gagnant vérifié sur la
moitié B :

| | moitié A (sélection) | moitié B (contrôle) |
|---|---|---|
| gain hit@3 | **+0,2000** | **+0,0000** |

Sur la moitié de contrôle, les **mêmes 12 questions** réussissent avec la
configuration retenue et avec la référence. Le +12,5 points annoncé était
le maximum d'une trentaine de configurations mesurées sur les mêmes 40
questions.

*Variante testée* : garder domaine et taxonomie, retirer seulement titre
et libellé (pure répétition). **Pire que les deux autres.** Ce n'est donc
pas la redondance qui nuit, c'est la taxonomie elle-même.

### 3.5 Poids, candidats, expansions — effet nul ou marginal

**`dense_weight`**, balayé de 0,0 à 1,0, plancher actif puis désactivé :
l'écart entre les deux colonnes se referme quand le poids monte et
s'inverse à 0,9. Le plancher est un **proxy grossier** de « faire
davantage confiance au sens ». Aucun poids ne fait bouger les faux
positifs : 1,000 aux onze valeurs.

**`candidate_k`** 40 / 100 / 200 : aucun effet mesurable (MRR 0,3969 /
0,3927 / 0,3933).

**Expansion « avertissement → demeure »** : le rang BM25 de l'article 1594
passe de 2269 à **179**, douze fois mieux — et aucun effet sur les
métriques. Même à `candidate_k=200`, il reste 180ᵉ sur 200 : « demeure »
est fréquent dans tout le CCQ, BM25 ne discrimine pas.

### 3.6 Union de deux formulations — **retenue**

La question part telle quelle, sa traduction en vocabulaire du Code part à
côté, et les deux recherches sont **réunies** : le score dense d'un
article est le meilleur des deux.

Ce qui a décidé la forme — traduire **en remplacement** dégrade dix des
trente-deux questions de contrôle, celles où l'usager employait déjà le
mot juste. « La liste des clients de mon employeur » passe du rang 1 au
rang 74.

| hit@3, configuration de production | 8 cibles | 32 contrôles |
|---|---|---|
| question seule, rerank actif | **0/8** | **21/32** |
| union, rerank actif | **3/8** | **19/32** |

**Net : +1 question sur 40** — dans la bande de bruit. La valeur est
concentrée sur les cas durs, pas globale.

**Retenue sur décision produit**, la propriétaire du projet jugeant que
ses usagers réels ressemblent davantage aux 8 cibles qu'aux 32 contrôles,
dont plusieurs emploient déjà le vocabulaire juridique exact. Ce n'est pas
une conclusion des chiffres.

Implémentation : argument optionnel `legal_terms` sur `semantic_search_*`,
produit par le planner **dans la même réponse** que sa décision — aucun
appel de modèle supplémentaire.

### 3.7 Résumés d'articles — rejetée par la courbe de couverture

Deux textes par article, en français facile : une ligne énonçant la règle,
et trois à quatre phrases. Score dense = max(texte, courte, B1).

Premiers signaux excellents. Sur un essai à l'aveugle — résumés générés
sans qu'aucune question n'existe, questions écrites sans lire les
résumés — le top-3 doublait (4/20 → 9/20).

Qualité en série : 200 résumés générés, 20 relus contre le texte de loi.
**8 sur 20 défectueux** — 4 faux, 2 à côté, 2 vagues. Deux corrections de
consigne (interdire les exemples inventés, obliger la courte à énoncer la
règle) ramènent le taux à **1 sur 20**, sur un tirage neuf.

Puis la mesure décisive, à couverture croissante, échantillons
**emboîtés**, les 40 cibles résumées à tous les paliers pour que le
bénéfice reste constant :

| couverture | % du corpus | top-1 | top-3 | top-10 | intrusions |
|---|---|---|---|---|---|
| 0 — référence | 0 % | 6 | **11** | 21 | 0 |
| 200 | 4,7 % | 15 | **24** | 32 | 20 |
| 800 | 18,7 % | 11 | **19** | 30 | 44 |
| 2 000 | 46,8 % | 10 | **13** | 25 | 58 |
| **4 273** | **99,9 %** | 7 | **11** | **18** | 64 |

**Le gain ne diminue pas : il disparaît.** À couverture complète le top-3
revient exactement à la référence, et le top-10 tombe **sous** elle.

Le mécanisme est auto-annulant, et c'est celui des étiquettes de taxonomie
du § 3.4 : les résumés rapprochent tous les articles d'un même registre —
français simple, phrases courtes, vocabulaire commun. **Un avantage
universel n'est pas un avantage.** Il ne reste que le bruit, qui
s'accumule.

Corollaire : **la qualité des résumés n'était pas le problème**. Le taux
de défectueux passé de 40 % à 5 % ne change rien à la courbe.

---

## 4. Ce qui a réellement fonctionné

Le seul levier qui agit sur les faux positifs n'est aucune des sept
pistes : c'est le **rerank LLM autorisé à rejeter**, implémenté au titre
du lot 1.4.

| | sans rerank | avec rerank |
|---|---|---|
| faux positifs | 1,000 | **0,333** |
| MRR | 0,397 | **0,490** |
| hit@3 | 0,500 | 0,525 |

Sept des douze questions sans réponse sont entièrement rejetées, dont le
cas du chat qui a ouvert l'enquête. Aucun numéro d'article inventé sur
51 appels. Coût : 0,0122 $ pour 52 questions.

C'est le seul point de la chaîne où la question rencontre le **texte** des
articles. Là où la géométrie vectorielle échoue, la lecture réussit.

---

## 5. Erreurs de méthode

Trois d'entre elles ont failli produire une décision erronée.

**La tautologie.** L'union a d'abord été mesurée par
`min(rang_brute, rang_traduite)` sur des rangs denses bruts, hors de la
chaîne. Or `min(a,b) ≤ a` toujours : la colonne « zéro régression » ne
*pouvait pas* montrer de régression. Une propriété de l'arithmétique
présentée comme un constat empirique. La mesure de bout en bout donne
2 régressions.

**Le rang dense confondu avec le rang final.** L'article 1594 « au rang
17, donc visible par le reranker » : le rang 17 était dense, le rang final
après mélange hybride est tout autre, et la fenêtre du reranker vaut 10,
pas 40. Trois erreurs dans une phrase.

**L'asymétrie de couverture.** Les résumés mesurés à 5,6 % de couverture
donnaient hit@3 = 0,550, présenté comme un plafond. Le bénéfice était
pleinement réalisé — les 40 cibles résumées — mais la nuisance ne l'était
qu'à un dix-huitième. La valeur réelle est 0,275.

**Le surajustement**, attrapé par le contrôle en deux moitiés (§ 3.4).

Ce que ces quatre cas ont en commun : une mesure prise sur un sous-système
et rapportée comme si elle valait pour le système. Le correctif appliqué
depuis — mesurer sur la chaîne complète, en configuration de production,
avec un contrôle jamais utilisé pour choisir.

---

## 6. Ce qui reste ouvert

**Aucune question d'utilisateur réel n'existe.** `/api/chat` diffuse la
réponse sans rien conserver. Toutes les questions de tous les jeux de test
ont été écrites par un modèle ou par l'assistant, à partir des articles
eux-mêmes — leur vocabulaire est donc plus proche de celui du Code que
celui d'un vrai usager. **Tous les chiffres de ce rapport sont des
plafonds.**

Journaliser la seule question posée suffirait à lever ce biais en
quelques semaines. C'est une donnée personnelle : décision de rétention et
de consentement, pas un changement technique.

**Le diagnostic du § 2 n'a pas été traité.** Le fossé entre le vocabulaire
de la situation et celui de la règle reste entier. Les six pistes rejetées
agissaient toutes sur la géométrie — modèle, recentrage, texte indexé,
poids, candidats, seuils. Aucune ne pouvait rien contre un problème de
nature doctrinale : savoir quel régime juridique gouverne quels faits
n'est pas une distance dans un espace vectoriel.

**La fenêtre du reranker** vaut 10 et est plafonnée en dur à 20. La
doubler coûte exactement le double de jetons — 4 519 → 9 011 caractères
par appel, mesuré — et repêche une question sur huit. `pre_rerank_count`
ignorerait silencieusement une valeur supérieure à 20.

---

## 7. Coût

| poste | coût |
|---|---|
| génération des 4 273 résumés | 0,4424 $ |
| séries d'essai des résumés | 0,041 $ |
| mesures du reranker | 0,062 $ |
| index « texte seul » et « texte + taxonomie » | 0,020 $ |
| traduction des questions, vecteurs du jeu de test | 0,003 $ |
| **total** | **≈ 0,57 $** |

BGE-M3 n'y figure pas : 2,3 Go de téléchargement et 35 minutes de CPU
local, aucun coût d'API, hypothèse rejetée.

Sept pistes, une retenue — et sur décision produit plutôt que sur ses
chiffres. Le taux d'échec est élevé, mais chaque piste a été close par une
mesure. Les scripts et les sorties brutes sont versionnés : aucun de ces
résultats n'exige de réindexer quoi que ce soit pour être réexaminé.

---

## Annexe — reproduire les mesures

| mesure | commande |
|---|---|
| référence du jeu de test | `python tests/test_retrieval_gold.py` |
| BGE-M3 contre OpenAI | `python scripts/compare_embedders.py` |
| recentrage | `python scripts/compare_centering.py` |
| texte indexé | `python scripts/compare_search_text.py` |
| contrôle de surajustement | `python scripts/validate_search_text.py` |
| poids, candidats, expansions | `python scripts/sweep_retrieval.py` |
| traduction des questions | `python scripts/measure_query_translation.py` |
| génération des résumés | `python scripts/generate_article_summaries.py` |

Sorties brutes conservées dans `Phase1_Data_Preparation/tests/fixtures/` :
`comparaison_embedders.json`, `comparaison_centering.json`,
`comparaison_search_text.json`, `validation_search_text.json`,
`sweep_retrieval.json`, `summary_quality_review.json`,
`summary_quality_review_v2.json`, `summary_coverage_curve.json`,
`retrieval_gold_baseline.json`.
