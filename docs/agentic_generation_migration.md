# Migration du pipeline legacy

Le pipeline historique, ses cleaners, sa reprise, son suivi des coûts, ses
audits d'identité et son split par groupes sont conservés. Il reste le défaut :

```bash
GENERATION_MODE=legacy ./Phase1_Data_Preparation/run_generation.sh
```

Le nouveau mode se sélectionne explicitement :

```bash
GENERATION_MODE=agentic ./Phase1_Data_Preparation/run_generation.sh --dry-run
```

Le one-shot juridique est remplacé, pour les nouveaux pilotes, par une machine
à états avec MCP réels et critiques. Les datasets legacy ne sont pas inclus par
défaut (`include_legacy_legal: false`), tandis que l'identité reste active.

Le changement d'entraînement est obligatoire : `dataset_formatter.py` conserve
les messages, et `assistant_loss.py` construit les `labels` explicitement.
`system`, `user` et `tool` valent `-100`; seuls le contenu assistant et sa fin
de tour portent la loss. Les trainers HF et Unsloth utilisent `Trainer` avec ce
masque, sans comportement implicite de `SFTTrainer`.

Le gabarit est ChatML strict sans injection « You are Qwen ». Le même template
est sauvegardé avec le tokenizer et utilisé par vLLM. Hermes n'est plus activé.

Avant une expérience comparative, activer explicitement les données legacy
avec `INCLUDE_LEGACY_LEGAL=true` dans le YAML/manifeste. Par défaut,
`INCLUDE_IDENTITY_DATA=true`. Ne mélangez jamais une release agentique auditée avec
des réponses MCP mockées ou des rejets bruts.
