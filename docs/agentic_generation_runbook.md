# Runbook de génération agentique

## Diagnostic

Depuis `Phase1_Data_Preparation` :

```bash
python -m pip install pydantic PyYAML openai mcp httpx pytest numpy datasets
python -m agentic_generation.cli doctor
python -m agentic_generation.cli doctor --allow-remote-calls
```

Sans le drapeau réseau, les connexions sont marquées `skipped`. La sortie ne
montre ni clé ni URL complète.

## Dry-run sans réseau

```bash
python -m agentic_generation.cli generate \
  --config configs/agentic_generation.yaml \
  --target-accepted 10 --max-scenarios 30 --seed 3407 --dry-run
```

Les observations portent explicitement `mock=true`. Elles ne peuvent pas être
publiées comme données réelles.

## Pilote réel préparé (ne pas lancer automatiquement)

```bash
python -m agentic_generation.cli generate \
  --config Phase1_Data_Preparation/configs/agentic_generation.yaml \
  --target-accepted 100 --seed 3407 --allow-remote-calls
```

Variables requises : `TEACHER_BASE_URL`, `TEACHER_API_KEY`, `TEACHER_MODEL`.
Avant le premier run réel, construire une fois l'index CCQ/CPC :

```powershell
python -m dotenv -f ..\.env run --no-override -- python -u -m agentic_generation.cli build-rag-index `
  --config .\configs\agentic_generation.yaml `
  --allow-remote-calls
```

L'index est conservé sous `data/agentic/rag_index`. Les appels d'embeddings et
leur coût sont affichés pour chaque lot. `--force` reconstruit explicitement
un index existant.

Variables Critic facultatives : `CRITIC_BASE_URL`, `CRITIC_API_KEY`,
`CRITIC_MODEL`. Pour publier explicitement, ajouter `--push-to-hf`, définir
`HF_TOKEN` et `HF_DATASET_REPO_ID=intelliwork/agentic_cot_data`.

Les dossiers `raw`, `accepted`, `rejected`, `checkpoints`, `manifests` et
`cache` sont écrits ligne par ligne sous `data/agentic`. `--resume` relit les
identifiants déjà terminés. Les raisons de rejet et coûts par rôle figurent
dans le manifeste. Aucun appel n'est fait après l'objectif accepté.

Chaque scénario affiche aussi une ligne de progression avec son nombre
d'appels OpenAI — Teacher, Critics et requêtes RAG —, son coût marginal et le
coût cumulé de l'exécution. Les
jetons d'entrée mis en cache sont facturés selon le tarif `cached input`
configuré pour le modèle. Exemple :

```text
[12/500] ACCEPTE | acceptés 8/100 | rejetés 4 | appels API +5 (cumul 61) | coût +$0.001234 USD (cumul $0.017890 USD)
```

## OpenAI gpt-4o-mini et publication Hugging Face (PowerShell)

Depuis `Phase1_Data_Preparation`, avec `OPENAI_API_KEY` et `HF_TOKEN` dans le
fichier `.env` de la racine :

```powershell
$env:TEACHER_BASE_URL = "https://api.openai.com/v1"
$env:TEACHER_MODEL = "gpt-4o-mini"

python -m dotenv -f ..\.env run -- python -m agentic_generation.cli generate `
  --config .\configs\agentic_generation.yaml `
  --target-accepted 100 `
  --max-scenarios 500 `
  --max-tool-calls 4 `
  --seed 3407 `
  --run-id openai-4o-mini-100 `
  --allow-remote-calls `
  --push-to-hf
```

La publication n'est tentée qu'après 100 trajectoires acceptées. En cas
d'échec Hugging Face, les données et le manifeste locaux restent conservés.

Pour préparer une expérience combinant les trajectoires acceptées et le jeu
d'identité, sans droit legacy :

```bash
python mix_datasets.py \
  --agentic_file data/agentic/accepted/<run-id>.jsonl \
  --identity_file data/processed/generated_identity_cot.jsonl \
  --no-include_legacy_legal --include_agentic --include_identity_data
```

`INCLUDE_LEGACY_LEGAL=false` et `INCLUDE_IDENTITY_DATA=true` sont les valeurs
par défaut du pilote. Le mode legacy du script maître passe ses options
explicitement afin de conserver son comportement historique.

## Tests

```bash
python -m pytest Phase1_Data_Preparation/tests -q
```

Ces tests sont offline et ne démarrent ni pod, ni appel Teacher/MCP distant, ni
publication Hugging Face.
