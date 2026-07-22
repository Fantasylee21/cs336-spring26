#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR/cs336-basics:$PYTHONPATH"

MODES=("forward" "forward-backward" "full")

for mode in "${MODES[@]}"; do
    echo "=============================================="
    echo "  MODE: $mode"
    echo "=============================================="
    python "$SCRIPT_DIR/benchmark.py" \
        --mode "$mode" \
        --all-sizes \
        --batch-size 16 \
        --seq-len 1024 \
        --warmup-steps 5 \
        --measure-steps 10
    echo ""
done

echo "Done."
