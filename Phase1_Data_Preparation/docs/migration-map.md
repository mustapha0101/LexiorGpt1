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

### agentic_generation/ → src/lexior/agentic/

All 29 modules moved as a single package to preserve the dense internal import
graph (60+ relative imports). Splitting into domain/dataset/critics/etc. would
require rewriting every cross-import and risks Pydantic class-identity failures.

| Old path | New path | Status | Wrapper | Reason |
|---|---|---|---|---|
| `agentic_generation/*.py` (29 modules) | `src/lexior/agentic/*.py` | move | yes | All modules moved as a package |
| `agentic_generation/*.py` (old paths) | compatibility wrappers | wrapper | — | Re-export from lexior.agentic to ensure single class identity |

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

## Final structure

```
Phase1_Data_Preparation/
├── src/lexior/                  # Installable package (pip install -e .)
│   ├── agent_graph/             # LangGraph 25-node central graph
│   │   ├── nodes/               # Individual graph nodes
│   │   ├── runner.py            # GraphRunner (live + dataset)
│   │   ├── state.py             # LexiorState
│   │   └── ...
│   ├── agentic/                 # Pipeline implementation (ex-agentic_generation)
│   │   ├── schemas.py           # Pydantic models
│   │   ├── config.py            # Configuration
│   │   ├── tool_catalog.py      # MCP tool catalog
│   │   ├── teacher_client.py    # LLM client
│   │   └── ... (29 modules)
│   ├── api/                     # FastAPI backend
│   ├── services/                # Shared service layer
│   ├── data/                    # Data cleaning/identity libraries
│   ├── evaluation/              # Comparison tools
│   └── observability/           # Cost tracking
├── agentic_generation/          # Compatibility wrappers → lexior.agentic
├── apps/
│   ├── chat-web/                # React/Vite chat frontend
│   └── results-viewer/          # Results UI
├── scripts/
│   ├── dataset_generation/      # CCQ, A2AJ, identity generators
│   ├── dataset_processing/      # Format, mix, audit
│   ├── huggingface/             # Push/resume from HF
│   └── run/                     # Shell pipeline scripts
├── tests/                       # 315 tests
├── configs/                     # YAML configs
├── deployment/                  # RunPod deployment
├── notebooks/                   # Jupyter notebooks
├── docs/                        # Documentation
├── pyproject.toml               # Package config (src layout)
└── *.py                         # Root compatibility wrappers (5 files)
```

## Commits

1. **Commit 1** — Inventory: migration-map.md, runtime-paths.md, test baseline
2. **Commit 2** — Safe root cleanup: notebooks, root tests, deployment, shell scripts, results viewer, .gitignore
3. **Commit 3** — Data scripts: cleaning, generation, processing, identity, HuggingFace libraries
4. **Commit 4** — src/lexior layout: central package move + pyproject.toml + editable install
5. **Commit 5** — agentic_generation → src/lexior/agentic + compatibility wrappers
6. **Commit 6** — Frontend consolidation: apps/chat-web/
7. **Commit 7** — Final cleanup, docs update, test verification
