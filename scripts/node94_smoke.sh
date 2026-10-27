#!/usr/bin/env bash
set -euo pipefail
# Explicit isolated working directory and interpreter supplied by the caller.
PYTHON=${PYTHON:?Set PYTHON to an existing isolated Python runtime}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT outside the repository}
mkdir -p "$RUN_ROOT"
"$PYTHON" -m allbench.cli demo --out "$RUN_ROOT/synthetic"
"$PYTHON" -m allbench.cli validate "$RUN_ROOT/synthetic/benchmark.json"
"$PYTHON" -m allbench.cli plan "$RUN_ROOT/synthetic/benchmark.json" --out "$RUN_ROOT/jobs.json"
"$PYTHON" -m allbench.cli evaluate "$RUN_ROOT/synthetic/benchmark.json" --features "$RUN_ROOT/synthetic/toy-features/features.json" --out "$RUN_ROOT/toy-results"
"$PYTHON" -m allbench.cli evaluate "$RUN_ROOT/synthetic/benchmark.json" --features "$RUN_ROOT/synthetic/toy-features/features.json" --out "$RUN_ROOT/toy-results"
if [[ -n "${DINO_PATH:-}" ]]; then
  "$PYTHON" -m allbench.cli encode "$RUN_ROOT/synthetic/benchmark.json" --out "$RUN_ROOT/dino-features" --dino "$DINO_PATH" --device "${DEVICE:-cpu}"
  "$PYTHON" -m allbench.cli evaluate "$RUN_ROOT/synthetic/benchmark.json" --features "$RUN_ROOT/dino-features/features.json" --out "$RUN_ROOT/dino-results"
  "$PYTHON" -m allbench.cli evaluate "$RUN_ROOT/synthetic/benchmark.json" --features "$RUN_ROOT/dino-features/features.json" --out "$RUN_ROOT/dino-results"
fi
