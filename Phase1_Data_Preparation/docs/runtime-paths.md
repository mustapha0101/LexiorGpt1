# Runtime Paths

Two runtime modes share a single LangGraph (`lexior/agent_graph/graph.py`).

## Live request

```
User message
→ FastAPI  (lexior/api/app.py)
→ GraphRunner.run_live()  (lexior/agent_graph/runner.py)
→ Central LangGraph  (25 nodes, compiled once)
→ Shared services  (lexior/services/*)
→ interrupt() / Command(resume=...)  for clarification
→ SSE stream  (type: status | decision | tool_call | tool_result | thinking | done)
→ return_live_answer node
```

## Dataset scenario

```
CLI  (agentic_generation/cli.py → python -m agentic_generation.cli generate)
→ AgenticOrchestrator  (thin wrapper, deprecated)
→ GraphRunner.run_dataset()  (lexior/agent_graph/runner.py)
→ Same Central LangGraph  (same 25 nodes)
→ Same shared services
→ Synthetic clarification  (no interrupt)
→ export_dataset node  → JSONL + ChatML
```

## Key invariants

- One compiled graph serves both modes; mode is a state flag, not a graph variant.
- Services (planner, verification, critics, validation) are instantiated once
  and shared across runs.
- `mode="live"` uses `interrupt()` for clarification; `mode="dataset"` uses
  synthetic answers from ScenarioSpec.
- The coverage gate, evidence classification, and acceptance blockers apply
  identically in both modes.
