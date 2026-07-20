#!/bin/bash
# =============================================================================
# CS336 Transformer LM Inference Scripts
# =============================================================================
# Usage:
#   bash infer.sh tiny "Once upon a time"
#   bash infer.sh tiny "The cat sat on" --temperature 0.6 --top_p 0.8
#   bash infer.sh tiny                     # interactive mode (no prompt)
#   bash infer.sh owt "The meaning of"     # OpenWebText model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----
DEVICE="mps"
MAX_TOKENS=100
TEMPERATURE=0.8
TOP_P=0.9

# ---- Configs (must match train.sh) ----
tiny_config() {
    VOCAB_SIZE=10000
    CONTEXT_LENGTH=256
    D_MODEL=512
    NUM_LAYERS=4
    NUM_HEADS=8
    D_FF=1344
    ROPE_THETA=10000.0
    VOCAB="cs336_basics/output/tiny_stories/vocab.json"
    MERGES="cs336_basics/output/tiny_stories/merges.txt"
    CHECKPOINT="checkpoints/checkpoint_0007000.pt"
}

owt_config() {
    VOCAB_SIZE=32000
    CONTEXT_LENGTH=256
    D_MODEL=512
    NUM_LAYERS=4
    NUM_HEADS=16
    D_FF=1344
    ROPE_THETA=10000.0
    VOCAB="cs336_basics/output/owt_train/vocab.json"
    MERGES="cs336_basics/output/owt_train/merges.txt"
    CHECKPOINT="checkpoints/checkpoint_final.pt"
}

# ---- Parse mode ----
MODE="${1:-tiny}"
case "$MODE" in
    tiny)   tiny_config ;;
    owt)    owt_config ;;
    -h|--help)
        echo "Usage: bash infer.sh {tiny|owt} [prompt] [--temperature T] [--top_p P] [--max_tokens N]"
        echo ""
        echo "  bash infer.sh tiny \"Hello world\""
        echo "  bash infer.sh tiny              # interactive"
        echo "  bash infer.sh owt \"The meaning\""
        exit 0
        ;;
    *)
        # First arg might be a prompt, default to tiny config
        tiny_config
        ;;
esac

# Shift mode if it matched a known config
case "$1" in
    tiny|owt) shift ;;
esac

# ---- Parse extra kwargs ----
PROMPT=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top_p)       TOP_P="$2"; shift 2 ;;
        --max_tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --checkpoint)  CHECKPOINT="$2"; shift 2 ;;
        --device)      DEVICE="$2"; shift 2 ;;
        *)
            if [[ -z "$PROMPT" ]]; then
                PROMPT="$1"
            else
                PROMPT="$PROMPT $1"
            fi
            shift
            ;;
    esac
done

echo "========================================================================="
echo "  Model: d_model=$D_MODEL  layers=$NUM_LAYERS  heads=$NUM_HEADS"
echo "  Checkpoint: $CHECKPOINT"
echo "  temperature=$TEMPERATURE  top_p=$TOP_P  max_tokens=$MAX_TOKENS"
echo "========================================================================="

if [[ -n "$PROMPT" ]]; then
    uv run python -m cs336_basics.infer \
        --vocab_size "$VOCAB_SIZE" \
        --context_length "$CONTEXT_LENGTH" \
        --d_model "$D_MODEL" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --d_ff "$D_FF" \
        --rope_theta "$ROPE_THETA" \
        --checkpoint "$CHECKPOINT" \
        --vocab "$VOCAB" \
        --merges "$MERGES" \
        --prompt "$PROMPT" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --device "$DEVICE"
else
    uv run python -m cs336_basics.infer \
        --vocab_size "$VOCAB_SIZE" \
        --context_length "$CONTEXT_LENGTH" \
        --d_model "$D_MODEL" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --d_ff "$D_FF" \
        --rope_theta "$ROPE_THETA" \
        --checkpoint "$CHECKPOINT" \
        --vocab "$VOCAB" \
        --merges "$MERGES" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --device "$DEVICE"
fi
