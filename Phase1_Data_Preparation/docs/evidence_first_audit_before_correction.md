# Audit evidence-first avant correction

## Provenance observée

- `article_reviews` est construit dans `update_research_state` par
  `enrich_article_review`, qui appelle `infer_article_profile`.
- `rule_roles` et `missing_fact_keys` proviennent donc du profil lexical de
  `article_review.py`, et non d'une extraction bornée au texte officiel.
- Les questions de clarification viennent de `build_clarification`, qui
  parcourt directement les `article_reviews`; `validate_plan` peut aussi
  reconstruire une question depuis les anciennes listes de faits manquants.
- `RuleContract` est construit par `evidence_first.build_rule_contract`, mais
  il réutilise encore `rule_roles`/`missing_fact_keys` et laisse
  `facts_not_required` vide.
- `build_answer_contract` transmet encore `conditional_reasoning_contract`,
  dont le contenu impose des conditions génériques de responsabilité.
- `validate_final` construit bien un `ClaimLedger`, mais la preuve reste
  fondée sur une sous-chaîne courte; les échecs sont parallèlement projetés
  dans `grounding_failures`, ce qui exige une conservation append-only
  explicite jusqu'à `compute_acceptance`.
- `update_active_task` réinitialise une partie du state, mais la projection
  persistante et la logique de follow-up doivent être vérifiées avec un test
  de contamination entre tâches.

## Écarts principaux

1. `infer_article_profile` attribue `fault`, `causation`,
   `prior_knowledge`, `failure_to_take_reasonable_action` et
   `damage_assessment` à partir de mots génériques.
2. `required_rule_roles()` impose toujours `general_liability_basis`.
3. La sélection des autorités utilise le profil lexical et un tie-break par
   identifiant de source.
4. Une clarification peut être déclenchée par les anciens `missing_fact_keys`,
   même lorsqu'une réponse conditionnelle serait possible.
5. Les réponses incertaines ne sont pas toutes normalisées en
   `asked_but_uncertain` avec déduplication par identifiant de clarification.
6. Le planner et `PlannerAgent` utilisent encore les paramètres legacy de lots
   d'articles, malgré la configuration evidence-first.
7. Les renvois réglementaires sont détectés, mais `get_quebec_regulation` est
   validé par `is_verified_quebec_decision`, qui est un contrôle de décisions.
8. `build_claim_ledger` considère une correspondance de 80 caractères comme
   une preuve directe et ne conserve pas les prémisses d'une inférence.

Cette note constitue le diagnostic de référence avant les modifications.
