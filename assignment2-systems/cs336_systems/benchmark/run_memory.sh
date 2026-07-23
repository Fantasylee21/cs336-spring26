#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/memory_snapshots"

export PYTHONPATH="$PROJECT_DIR/cs336-basics:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

MODEL="xl"
BATCH_SIZE=8
SEQ_LENS=(128 2048)
MODES=("forward" "full")
AMP_MODES=("none" "bf16")

# ============================================================
# Part (a)(b)(c): PyTorch memory profiling → pytorch.org/memory_viz
# ============================================================
for seq_len in "${SEQ_LENS[@]}"; do
    for mode in "${MODES[@]}"; do
        for amp in "${AMP_MODES[@]}"; do
            echo "============================================"
            echo "Memory profile: model=$MODEL  seq=$seq_len  mode=$mode  amp=$amp"
            echo "============================================"

            python "$SCRIPT_DIR/profile_memory.py" \
                --model-size "$MODEL" \
                --seq-len "$seq_len" \
                --batch-size "$BATCH_SIZE" \
                --profile-mode "$mode" \
                --amp "$amp" \
                --memory-profile \
                --output-dir "$OUTPUT_DIR" \
                --warmup-steps 2 \
                --measure-steps 1

            echo ""
        done
    done
done

echo ""
echo "Snapshots in: $OUTPUT_DIR"
echo "To visualize, upload .pickle files to https://pytorch.org/memory_viz"
echo ""

# ============================================================
# Part (f): nsys + memory profiling
# ============================================================
echo "============================================"
echo "NSYS memory profile: xl seq=2048 full+forward"
echo "============================================"

NSYS_OUT="$SCRIPT_DIR/nsys_memory"

nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --stats=true \
    --force-overwrite=true \
    --output="$NSYS_OUT" \
    python "$SCRIPT_DIR/profile_memory.py" \
        --model-size "$MODEL" \
        --seq-len 2048 \
        --batch-size "$BATCH_SIZE" \
        --profile-mode "full" \
        --amp "none" \
        --warmup-steps 2 \
        --measure-steps 1

echo ""
echo "NSYS memory profile saved to: $NSYS_OUT.qdrep"

nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --stats=true \
    --force-overwrite=true \
    --output="${NSYS_OUT}_forward" \
    python "$SCRIPT_DIR/profile_memory.py" \
        --model-size "$MODEL" \
        --seq-len 2048 \
        --batch-size "$BATCH_SIZE" \
        --profile-mode "forward" \
        --amp "none" \
        --warmup-steps 2 \
        --measure-steps 1

echo "Done."
