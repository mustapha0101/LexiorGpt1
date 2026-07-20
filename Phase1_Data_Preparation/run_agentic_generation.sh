#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python3 -m agentic_generation.cli generate \
  --config "${AGENTIC_CONFIG:-configs/agentic_generation.yaml}" \
  --target-accepted "${TARGET_ACCEPTED:-100}" \
  --seed "${GENERATION_SEED:-3407}" \
  "$@"
