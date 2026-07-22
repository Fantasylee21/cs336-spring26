#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_DIR/cs336-basics:$PYTHONPATH"

MODES=("forward" "forward-backward" "full")
AMP_MODES=("none" "bf16")

# ============================================================
# Part 1: FP32 baseline — all modes, all sizes
# ============================================================
for mode in "${MODES[@]}"; do
    echo "=============================================="
    echo "  MODE: $mode (FP32)"
    echo "=============================================="
    python "$SCRIPT_DIR/benchmark.py" \
        --mode "$mode" \
        --all-sizes \
        --batch-size 16 \
        --seq-len 1024 \
        --warmup-steps 5 \
        --measure-steps 10 \
        --amp none
    echo ""
done

# ============================================================
# Part 2: Mixed precision comparison — forward+backward, all sizes
# ============================================================
for amp in "${AMP_MODES[@]}"; do
    echo "=============================================="
    echo "  MODE: forward-backward | AMP: $amp"
    echo "=============================================="
    python "$SCRIPT_DIR/benchmark.py" \
        --mode forward-backward \
        --all-sizes \
        --batch-size 16 \
        --seq-len 1024 \
        --warmup-steps 5 \
        --measure-steps 10 \
        --amp "$amp"
    echo ""
done

echo "Done."
