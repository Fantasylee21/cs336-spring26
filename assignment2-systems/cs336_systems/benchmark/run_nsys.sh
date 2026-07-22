#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/nsys_output"

export PYTHONPATH="$PROJECT_DIR/cs336-basics:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"

# Choose 2 model sizes from Table 1
MODEL_SIZES=("small" "medium")

# 3 power-of-two context lengths > 128
# Adjust the largest value based on your GPU memory
SEQ_LENS=(256 512 1024)

# Profile modes: forward, forward-backward, full
MODES=("forward" "forward-backward" "full")

BATCH_SIZE=8

for size in "${MODEL_SIZES[@]}"; do
    for seq_len in "${SEQ_LENS[@]}"; do
        for mode in "${MODES[@]}"; do
            OUTPUT_NAME="$OUTPUT_DIR/nsys_${size}_seq${seq_len}_${mode}"
            echo "============================================"
            echo "Profiling: size=$size  seq_len=$seq_len  mode=$mode"
            echo "Output: $OUTPUT_NAME"
            echo "============================================"

            nsys profile \
                --trace=cuda,nvtx \
                --stats=true \
                --force-overwrite=true \
                --output="$OUTPUT_NAME" \
                python "$SCRIPT_DIR/profile_nsys.py" \
                    --model-size "$size" \
                    --seq-len "$seq_len" \
                    --batch-size "$BATCH_SIZE" \
                    --profile-mode "$mode" \
                    --warmup-steps 3 \
                    --measure-steps 3

            echo ""
        done
    done
done

echo ""
echo "All profiles saved to: $OUTPUT_DIR"
echo ""
echo "To view a profile in the GUI:"
echo "  nsys-ui $OUTPUT_DIR/nsys_small_seq1024_full.qdrep"
echo ""
echo "To view kernel stats from command line:"
echo "  nsys stats $OUTPUT_DIR/nsys_small_seq1024_full.qdrep"
