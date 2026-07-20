# Phase 1 — Data Preparation

This document explains how Phase 1 generates the training dataset for LexiorGPT:
what the pipeline does at each stage, the modules involved, and the design
decisions behind them.

Phase 1 turns raw legal corpora into a single formatted, audited chat dataset
made of three kinds of examples:

- **`legal_federal`** — Canadian federal law (from `a2aj/canadian-laws`)
- **`legal_quebec`** — Quebec civil law (from `intelliwork/canadian-quebec-law-corpus`)
- **`identity`** / **`identity_control`** — LexiorGPT product-identity alignment

Each legal example is a chain-of-thought (CoT) resolution in French structured
with the **IRAC** method (Issue, Rule, Application, Conclusion), inside a
`<thinking>...</thinking>` block, followed by a plain-French answer and a single
footnote citation.

---

## Pipeline overview

`run_generation.sh` is the orchestrator. It runs six stages, then an audit gate:

```
[1/6] resume_from_hf.py -> generator_a2aj.py     federal legal CoT   (Teacher API)
[2/6] generate_ccq_data.py                       Quebec legal CoT    (Teacher API)
[3/6] generate_identity_data.py                  identity pool       (no API, templates)
[4/6] push_to_hf.py x3                            each corpus -> its own private repo
[5/6] mix_datasets.py                            combine the three sources (ratio-controlled)
[6/6] dataset_formatter.py                       render to chat text, split train/test
[audit] audit_training_dataset.py                BLOCKING gate before any upload
```

The three source corpora are pushed to separate private repositories, which
also serve as resume checkpoints:

- `intelliwork/canadian-cot-dataset-federal-french`
- `intelliwork/canadian-cot-dataset-quebec-french`
- `intelliwork/canadian-cot-dataset-identity-french`

The legacy Teacher is reached through an OpenAI-compatible API. The agentic
pipeline uses provider-neutral `TEACHER_BASE_URL`, `TEACHER_API_KEY`, and
`TEACHER_MODEL` (with temporary `OPENAI_*`/`GEN_MODEL` fallback).

---

## Stage 1 — Federal legal data (`generator_a2aj.py` + `a2aj_cleaner.py`)

Reads raw Canadian legislation from `a2aj/canadian-laws`, sends the section text
to the Teacher, and stores the returned IRAC CoT.

**Source cleaning (`a2aj_cleaner.py`).** Before any Teacher call, each law passes
through `clean_law()`, which keeps only usable material:

- **Federal only** (`LEGISLATION-FED`, `REGULATIONS-FED`). The dataset contains
  no Quebec law and 20 jurisdictions in total; provincial/territorial law is
  excluded here (Quebec is handled separately in Stage 2).
- **Not wholly repealed.** A law is dropped if a repeal marker appears in its
  header block **or** every section is a stub. The two rules together catch both
  header-repealed acts (whose sections may still look alive) and acts hollowed
  out section by section — while keeping active *repeal acts* like
  « Loi sur l'abrogation des lois » (judged by content, never by name).
- **Section-level classification.** Every section is labelled `LIVE`, `PARTIAL`
  (live section with a struck sub-item), `DEAD` (`[Abrogé …]`), `BLANK`
  (`[blank]`), `SPENT` (`[Modifications]` / `[Abrogation]` — spent amending
  provisions, not repealed), etc. Only `LIVE` and `PARTIAL` sections reach the
  Teacher. Repealed and empty sections are never cited.
- **French only** for federal law (constitutionally bilingual; a missing French
  version is an anomaly, not something to fall back to English on).

The corpus is **ordered by jurisdiction**, so the first federal law sits at row
~1,016. The generator therefore pre-filters to federal rows *before* applying
`--limit`, otherwise `--limit 1000` would scan only Alberta/BC and produce zero
rows.

**`--whole_laws_only`** (default on for the first run) keeps only laws that fit
entirely in the context budget, rather than truncating long ones. A truncated
statute is almost always reduced to its opening definitions, so it is set aside
to await a future section-level split rather than misrepresented.

**Citation grounding.** After generation, the URL the Teacher wrote is
overwritten with the row's real `source_url` from the dataset, so every federal
citation points at the actual `laws-lois.justice.gc.ca` document.

## Stage 2 — Quebec legal data (`generate_ccq_data.py` + `ccq_cleaner.py`)

Reads articles from `intelliwork/canadian-quebec-law-corpus` (Code civil du
Québec + Code de procédure civile, article-level), and asks the Teacher to write
a realistic Quebec scenario and resolve it in IRAC.

**Source cleaning (`ccq_cleaner.py`).** `clean_article()` drops Quebec stubs,
which use a different syntax from the federal corpus — parentheses, not brackets:
`(Abrogé).`, `(Omis).`, `(Modification intégrée au c. B-1, a. 125).`. The domain
label is rebuilt from `chemin_taxonomy` (the 10 books of the CCQ, 6 titles of the
CPC).

**Constructed citation URL.** The Quebec corpus has no `source_url` column. The
URL is therefore *built* deterministically from the article id
(`legisquebec.gouv.qc.ca/fr/document/lc/ccq-1991#se:1457`) and passed into the
prompt, giving the Quebec generator the same grounding guarantee the federal one
has. It is rewritten into the citation after generation, never left to the
Teacher.

## Stage 3 — Identity data (`generate_identity_data.py` + `identity_templates.py` + `identity_policy.py`)

Generates the product-identity alignment set from templates — **no API calls**.
The goal is for the model to consistently identify as *LexiorGPT, developed by
IntelliWork, a legal assistant for Canadian and Quebec law*, even with no system
prompt, and to never disclose or confirm its underlying foundation model.

- **`identity_policy.py`** is the single source of truth: the product/developer
  names, the list of forbidden technical terms (Qwen, GPT-4, Alibaba, …), and the
  forbidden false claims (« entraîné à partir de zéro », « aucun modèle
  sous-jacent »). It is imported by generation, audit, and (future) evaluation so
  all three enforce the same rule. Word boundaries are handled so that `GPT-4`
  and `ChatGPT` are caught while `LexiorGPT` is not.
- **`identity_templates.py`** holds 59 template families across 18 categories
  (direct identity, developer, false premise, prompt injection, role-play,
  technical provenance, forced yes/no, one-word-model, multi-turn, …) and 5
  languages (fr, en, mixed, informal Quebec French, typos). Language is a
  separate dimension rather than a category, so an English "who are you" is still
  `direct_identity`. Each category has several families so the train/test split
  can cover every category.
- Identity records carry **no system message** and **no `<thinking>` block** — an
  identity question needs neither. Every assistant target is validated against
  the policy before it is written; a violation fails the run.
- A small set of **`identity_control`** records are pure legal Q&A with *no*
  self-introduction. They are the counterweight against over-branding: without
  them, the model would learn to announce itself before every ordinary legal
  answer.

## Stage 5 — Mixing (`mix_datasets.py`)

Replaces the previous shell `cat`. A `cat` made the identity proportion
accidental — it was whatever the files happened to contain. The mixer makes it
**requested, verified, and recorded**:

- Labels federal rows `legal_federal` and Quebec rows `legal_quebec`; identity
  rows keep their own metadata.
- Accepts either an **absolute** identity count (`--identity_count`, default 500)
  or a **target ratio** (`--identity_ratio`, for the 2 % / 5 % / 8 % experiments).
- Deterministic shuffle; oversamples the identity pool only if needed, and
  **fails** if the repeat factor gets too high (repetition teaches memorization).
- Writes a JSON manifest: source counts, final counts, percentages, identity
  category distribution, seed, input paths.

## Stage 6 — Formatting (`dataset_formatter.py`)

Renders each record to the `text` field the trainer consumes, and splits
train/test.

- **Metadata survives.** `dataset_type`, `identity_category`, `template_group`,
  `source_id`, `language` are preserved (the previous version dropped every
  column but `text`, making it impossible to prove which identity rows were
  trained).
- **Conditional system prompt.** Legal examples get the canonical IRAC system
  prompt; identity examples get **none**. A configurable
  `--legal_system_prompt_dropout` (default 0.15) removes the system prompt from a
  deterministic 15 % of legal examples, so the model does not lose its legal
  behaviour when no system prompt is supplied at inference.
- **Strict chat template.** Qwen's stock chat template injects
  `"You are Qwen, created by Alibaba Cloud"` whenever no system message is given.
  Removing the system message would therefore have trained every identity example
  under a prompt asserting the model is Qwen. The formatter installs a strict
  ChatML template that renders exactly the messages provided and injects nothing,
  and verifies the injection is gone (hard-fails otherwise).
  **The inference server must use the same template** (`deploy_vllm.py
  --chat-template`), or training and serving diverge.
- **Group-aware split.** Legal examples are grouped by `source_id`; identity
  examples by `template_group`. A whole template family (all its paraphrases)
  goes entirely to train or entirely to test — a random per-row split would leak
  near-identical paraphrases across the boundary. Writes a `split_audit.json`.

## Audit gate (`audit_training_dataset.py`)

Runs on the formatted train/test files and **exits non-zero on any critical
identity-policy violation**, so a faulty dataset is never uploaded. It reports
records by type, tokens by type (row-% and token-% differ because legal examples
are far longer), duplicate questions/answers, and it flags: a system message on
an identity example, a `<thinking>` block on an identity example, a forbidden
technical term in an identity target, missing identity records, or missing
identity in the test split.

---

## Supporting features

**Resume + row caps.** Both legal generators are resumable and support a
`--max_rows` total-row cap:

- Federal keys resume on `original_index`; Quebec on `(source_id,
  scenario_index)` so a partially-done article resumes at the right scenario.
- `--max_rows N` counts the **total** rows targeted in the file, existing rows
  included, so `--max_rows 1000` then `--max_rows 5000` yields 1000, then 4000
  more. The threaded federal generator stops workers *before* the API call once
  the target is hit.
- This is why the first run can be a small validation batch (e.g. 1000 each) and
  later runs simply continue instead of restarting.

**Cost tracking (`api_cost.py`).** Token counts are read from the API `usage`
field (not estimated); cost is derived from a configurable price table. Each
generator prints a running cumulative cost and writes a `*_cost.json` next to its
output. Self-hosted models report zero cost, tokens only. Measured rate with
`gpt-4o-mini`: **~$0.00045 per kept row**.

**Rejection instrumentation.** The Quebec generator counts *why* each attempt is
rejected (missing situation, unclosed `<thinking>`, empty answer, missing
citation, …) and prints the breakdown, so a too-strict gate can never silently
discard most calls.

**Tests.** `test_a2aj_cleaner.py` (50 cases) and `test_ccq_cleaner.py` (39 cases)
cover the cleaners' classification and edge cases — repeal detection, the
repeal-act name trap, stub syntaxes, context-budget truncation, the
jurisdiction filter — and run offline with no GPU or API.

---

## Files

New:

| File | Role |
|---|---|
| `a2aj_cleaner.py` | Federal source cleaning: jurisdiction, repeal, section classification |
| `ccq_cleaner.py` | Quebec source cleaning + LegisQuébec URL construction |
| `identity_policy.py` | Canonical identity policy (names, forbidden terms/claims, validation) |
| `identity_templates.py` | Identity template families (59 families, 18 categories, 5 languages) |
| `mix_datasets.py` | Ratio-controlled mixer + manifest (replaces `cat`) |
| `audit_training_dataset.py` | Blocking pre-upload audit |
| `api_cost.py` | Token/cost accounting from API `usage` |
| `test_a2aj_cleaner.py`, `test_ccq_cleaner.py` | Offline unit tests |

Modified:

| File | Change |
|---|---|
| `generator_a2aj.py` | Federal-only pre-filter, source cleaning, IRAC-template prompt, empty-answer & dialogue gates, `--whole_laws_only`, `--max_rows`, cost tracking |
| `generate_ccq_data.py` | Corpus-driven (hardcoded articles removed), scenario-as-user-turn, URL grounding, resume index, `--max_rows`, rejection counters, cost tracking |
| `generate_identity_data.py` | Rewritten: template-based, policy-validated, no system prompt, no `<thinking>`, metadata, deterministic |
| `dataset_formatter.py` | Metadata preserved, conditional system prompt + dropout, strict ChatML template, group-aware split |
| `push_to_hf.py` | Per-source repo mapping (three repos), private by default |
| `resume_from_hf.py` | Reads checkpoints as parquet datasets (the format `push_to_hf.py` writes) |
| `run_generation.sh` | Mixer instead of `cat`, identity env vars, audit gate, three-repo upload |

---

## How to run

```bash
cd Phase1_Data_Preparation
# Load your local environment; never commit .env or API keys.

# First validation batch (federal + Quebec ~1000 each, identity pool 1000):
MAX_ROWS_FED=1000 MAX_ROWS_QC=1000 \
IDENTITY_POOL_SIZE=1000 IDENTITY_COUNT=500 \
LEGAL_SYSTEM_PROMPT_DROPOUT=0.15 \
bash run_generation.sh

# Continue later (resume, no restart) by raising the caps:
MAX_ROWS_FED=5000 MAX_ROWS_QC=5000 bash run_generation.sh
```

Key environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MAX_ROWS_FED`, `MAX_ROWS_QC` | 1000 | Total rows targeted per legal source |
| `WHOLE_LAWS_ONLY` | true | Federal: use only laws that fit the context budget |
| `GEN_WORKERS` | 8 | Federal concurrency (raise for a self-hosted vLLM Teacher) |
| `IDENTITY_POOL_SIZE` | 1000 | Identity conversations generated |
| `IDENTITY_COUNT` | 500 | Absolute identity rows in the mix |
| `IDENTITY_RATIO` | (unset) | Target ratio instead of a count (2 %/5 %/8 % experiments) |
| `LEGAL_SYSTEM_PROMPT_DROPOUT` | 0.15 | Fraction of legal rows with the system prompt removed |
| `IDENTITY_SEED` | 3407 | Seed for identity generation, mixing, dropout, split |

Individual stages can be run and inspected on their own (see each script's
`--help`).

---

## Inspect agentic results locally

The read-only result inspector follows live files in `data/agentic` and shows
accepted and rejected questions, conversations, tool calls, grounding, critic
scores, metadata, and raw JSON. It uses only the Python standard library.

Agentic prompt version `agentic-1.3` also enforces two dataset guarantees:

- precise CCQ/CPC article requests reproduce the complete MCP text verbatim,
  without an LLM paraphrase or generic disclaimer;
- accepted examples are steered toward per-category taxonomy quotas, so easy
  one-tool article requests cannot fill the entire target by themselves.

Use a new run id when comparing this policy with an older pilot; already
accepted rows are append-only and are not silently rewritten.

From PowerShell in `Phase1_Data_Preparation`:

```powershell
python .\serve_results_ui.py --open
```

Without `--open`, navigate to `http://127.0.0.1:8765`. Use `--port 9000` to
select another port. The page refreshes every five seconds, so it can remain
open while a generation run is writing its JSONL files.

---

## Known limitations and not-yet-done

- **Content correctness is not verified.** The gates check *structure* (IRAC
  present, citation present and grounded to the right article, no forbidden
  terms), not legal *truth*. When the Teacher cites a specific article from
  memory (e.g. a decimal sub-article), no gate can confirm it exists or says what
  is claimed. Specific-article citations should be spot-checked by a human before
  a production run. The optional grounding filter in `generator_a2aj.py` remains
  disabled (it was rejecting too much).
- **Federal is currently law-level.** One law → one record. Long statutes
  (~78 % of federal Acts exceed the budget) are excluded under
  `--whole_laws_only` and await a section-level split, which would both raise the
  ceiling above ~2,400 laws and cover the substantive statutes.
- **Serving must mirror the strict chat template**, or the identity training is
  undermined at inference.
- **`deploy_dual_pods.py`** does not yet forward `MAX_ROWS_*` / `WHOLE_LAWS_ONLY`
  / identity env vars into the pod; it would fall back to the script defaults.
- **Phase 2/3 identity items are out of scope here**: assistant-only training
  loss, explicit dataset selection / checkpoint safety in `run_training.sh`, and
  the no-system-prompt identity benchmark are not implemented in this phase.
