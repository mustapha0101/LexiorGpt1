# Migration Map

Baseline: 315 tests passing (2026-07-23).

## File inventory and target locations

### Root-level scripts and libraries

| Old path | New path | Status | Wrapper | Reason |
|---|---|---|---|---|
| `a2aj_cleaner.py` | `src/lexior/data/cleaning/a2aj.py` | move | yes | Library used by generator_a2aj |
| `ccq_cleaner.py` | `src/lexior/data/cleaning/ccq.py` | move | yes | Library used by generate_ccq_data |
| `api_cost.py` | `src/lexior/observability/costs.py` | move | yes | Imported by runtime (teacher_client) |
| `identity_policy.py` | `src/lexior/data/identity/policy.py` | move | yes | Library for identity generation |
| `identity_templates.py` | `src/lexior/data/identity/templates.py` | move | yes | Library for identity generation |
| `generate_ccq_data.py` | `scripts/dataset_generation/generate_ccq.py` | move | no | Standalone script |
| `generate_identity_data.py` | `scripts/dataset_generation/generate_identity.py` | move | no | Standalone script |
| `generator_a2aj.py` | `scripts/dataset_generation/generate_a2aj.py` | move+rename | no | Standalone script |
| `dataset_formatter.py` | `scripts/dataset_processing/format_dataset.py` | move | no | Standalone script |
| `mix_datasets.py` | `scripts/dataset_processing/mix_datasets.py` | move | no | Standalone script |
| `audit_training_dataset.py` | `scripts/dataset_processing/audit_dataset.py` | move | no | Standalone script |
| `push_to_hf.py` | `scripts/huggingface/push_dataset.py` | move | no | Standalone script |
| `resume_from_hf.py` | `scripts/huggingface/resume_generation.py` | move | no | Standalone script |
| `deploy_dual_pods.py` | `deployment/runpod/deploy_dual_pods.py` | move | no | Deployment script |
| `deploy_generation_runpod.py` | `deployment/runpod/deploy_generation.py` | move | no | Deployment script |
| `deploy_teacher_runpod.py` | `deployment/runpod/deploy_teacher.py` | move | no | Deployment script |
| `serve_results_ui.py` | `apps/results-viewer/server.py` | move | no | Results viewer server |
| `run_agentic_generation.sh` | `scripts/run/run_agentic_generation.sh` | move | no | Shell script |
| `run_generation.sh` | `scripts/run/run_generation.sh` | move | no | Shell script (legacy pipeline) |
| `dataset_eda.ipynb` | `notebooks/dataset_eda.ipynb` | move | no | Notebook |
| `test_a2aj_cleaner.py` | `tests/data/test_a2aj_cleaner.py` | move | no | Root-level test |
| `test_ccq_cleaner.py` | `tests/data/test_ccq_cleaner.py` | move | no | Root-level test |

### results_ui/

| Old path | New path | Status | Reason |
|---|---|---|---|
| `results_ui/index.html` | `apps/results-viewer/index.html` | move | UI asset |
| `results_ui/app.js` | `apps/results-viewer/app.js` | move | UI asset |
| `results_ui/styles.css` | `apps/results-viewer/styles.css` | move | UI asset |

### lexior/ (central package)

| Old path | New path | Status | Reason |
|---|---|---|---|
| `lexior/` | `src/lexior/` | move | Standard src layout |
| `lexior/agent_graph/` | `src/lexior/agent_graph/` | move | Core graph engine |
| `lexior/services/` | `src/lexior/services/` | move | Shared services |
| `lexior/api/` | `src/lexior/api/` | move | FastAPI app |
| `lexior/evaluation/` | `src/lexior/evaluation/` | move | Comparison tool |
| `lexior/web/` | `apps/chat-web/` | move | Active chat frontend |

### agentic_generation/ → src/lexior/ subpackages

| Old path | New path | Status | Wrapper | Reason |
|---|---|---|---|---|
| `schemas.py` | `src/lexior/domain/schemas.py` | move | yes | Shared schemas |
| `taxonomy.py` | `src/lexior/domain/taxonomy.py` | move | yes | Request type taxonomy |
| `config.py` | `src/lexior/config.py` | move | yes | Shared configuration |
| `tool_catalog.py` | `src/lexior/tools/catalog.py` | move | yes | Tool catalog |
| `teacher_client.py` | `src/lexior/models/client.py` | move | yes | Model client |
| `scenario_generator.py` | `src/lexior/dataset/scenario_generator.py` | move | yes | Dataset generation |
| `storage.py` | `src/lexior/dataset/storage.py` | move | yes | Run storage |
| `publisher.py` | `src/lexior/dataset/publisher.py` | move | yes | HF publisher |
| `training_formatter.py` | `src/lexior/dataset/formatter.py` | move | yes | ChatML formatter |
| `anchor_bank.py` | `src/lexior/dataset/anchor_bank.py` | move | yes | Anchor bank |
| `fixtures.py` | `src/lexior/dataset/fixtures.py` | move | yes | Mock MCP fixtures |
| `migration.py` | `src/lexior/dataset/migration.py` | move | yes | Schema migration |
| `legal_rag.py` | `src/lexior/retrieval/legal_rag.py` | move | yes | RAG retrieval |
| `case_law_gate.py` | `src/lexior/validation/case_law.py` | move | yes | Case law gate |
| `response_verifier.py` | `src/lexior/validation/tool_results.py` | move | yes | Response verification |
| `validators.py` | `src/lexior/validation/trajectory.py` | move | yes | Trajectory validation |
| `acceptance.py` | `src/lexior/validation/acceptance.py` | move | yes | Acceptance logic |
| `legal_critic.py` | `src/lexior/critics/legal.py` | move | yes | Legal critic |
| `agentic_critic.py` | `src/lexior/critics/agentic.py` | move | yes | Agentic critic |
| `critic_context.py` | `src/lexior/critics/context.py` | move | yes | Critic context |
| `critic_profiles.py` | `src/lexior/critics/profiles.py` | move | yes | Critic profiles |
| `prompts.py` | `src/lexior/prompts/` (split) | move | yes | Prompt templates |
| `planner_agent.py` | `src/lexior/services/planner.py` (merge) | move | yes | Planner implementation |
| `mcp_executor.py` | `src/lexior/services/tool_execution.py` (merge) | move | yes | Tool executor |
| `trajectory_agent.py` | `src/lexior/services/answer_generation.py` (merge) | move | yes | Answer writer |
| `orchestrator.py` | `legacy/orchestrator.py` | move | yes | Deprecated thin wrapper |
| `cli.py` | keep (updated imports) | keep | no | CLI entry point |
| `__init__.py` | keep (re-exports) | keep | no | Package init |
| `__main__.py` | keep | keep | no | Module entry |

### configs/

| Old path | New path | Status | Reason |
|---|---|---|---|
| `configs/` | `configs/` | keep | Already organized |

### tests/

| Old path | New path | Status | Reason |
|---|---|---|---|
| `tests/` | `tests/` | keep | Already organized (update imports) |

### Frontend determination

| Frontend | Location | Status | Reason |
|---|---|---|---|
| `lexior/web/` | `apps/chat-web/` | move | Active chat UI, connected to lexior/api/app.py |
| `Phase4_ChatApp/` | keep in Phase4 | keep | Evaluation UI, not a chat frontend |

### Generated files to remove

| Path | Action | Reason |
|---|---|---|
| `__pycache__/` | delete | Generated |
| `.pytest_cache/` | delete | Generated |
| `*.pyc` | delete | Generated |
| `data/agentic/cache/` | gitignore | Large cache dir |
| `node_modules/` | gitignore | Package manager |
| `dist/` | gitignore | Build output |

## Migration phases

1. **Commit 1** — This document + runtime-paths.md + test baseline (no moves)
2. **Commit 2** — Notebooks, root tests, deployment, shell scripts, results viewer, .gitignore
3. **Commit 3** — Data scripts (cleaning, generation, processing, identity, HuggingFace)
4. **Commit 4** — src/lexior layout (central package move + pyproject.toml)
5. **Commit 5** — agentic_generation → src/lexior subpackages (schemas, config, tools, dataset, validation, critics, prompts)
6. **Commit 6** — Frontend consolidation (apps/chat-web/)
7. **Commit 7** — Legacy cleanup, final docs, full test verification
