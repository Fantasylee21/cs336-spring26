#!/bin/bash
# =============================================================================
# CS336 Transformer LM Training Scripts
# =============================================================================
# Usage:
#   bash train.sh tiny          # Train a small model on TinyStories (quick dev runs)
#   bash train.sh owt           # Train a larger model on OpenWebText
#   bash train.sh tiny-wandb    # Same as tiny, with wandb logging
#   bash train.sh owt-wandb     # Same as owt, with wandb logging
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Common settings ----
DEVICE="mps"
CHECKPOINT_DIR="checkpoints"
TRAIN_SCRIPT="cs336_basics.train"

# ---- Model: TinyStories (47M params) ----
tiny_config() {
    VOCAB_SIZE=10000
    CONTEXT_LENGTH=256
    D_MODEL=512
    NUM_LAYERS=4
    NUM_HEADS=8
    D_FF=1344          
    ROPE_THETA=10000.0

    MAX_LR=1e-3
    MIN_LR=1e-5
    WEIGHT_DECAY=0.01
    BATCH_SIZE=64
    MAX_ITERS=50000
    WARMUP_ITERS=2000
    GRAD_CLIP=1.0

    TRAIN_DATA="data/TinyStoriesV2-GPT4-train.npy"
    VAL_DATA="data/TinyStoriesV2-GPT4-valid.npy"
}

# ---- Model: OpenWebText (162M params) ----
owt_config() {
    VOCAB_SIZE=32000
    CONTEXT_LENGTH=256
    D_MODEL=512
    NUM_LAYERS=4
    NUM_HEADS=16
    D_FF=1344        
    ROPE_THETA=10000.0

    MAX_LR=3e-4
    MIN_LR=3e-6
    WEIGHT_DECAY=0.1
    BATCH_SIZE=32
    MAX_ITERS=100000
    WARMUP_ITERS=5000
    GRAD_CLIP=1.0

    TRAIN_DATA="data/owt_train.npy"
    VAL_DATA="data/owt_valid.npy"
}

# ---- Parse command ----
MODE="${1:-tiny}"
case "$MODE" in
    tiny)
        tiny_config
        WANDB_FLAGS=""
        ;;
    tiny-wandb)
        tiny_config
        WANDB_FLAGS="--wandb --wandb_project cs336-transformer --wandb_run_name tiny-$(date +%m%d-%H%M)"
        ;;
    owt)
        owt_config
        WANDB_FLAGS=""
        ;;
    owt-wandb)
        owt_config
        WANDB_FLAGS="--wandb --wandb_project cs336-transformer --wandb_run_name owt-$(date +%m%d-%H%M)"
        ;;
    *)
        echo "Usage: bash train.sh {tiny|tiny-wandb|owt|owt-wandb}"
        exit 1
        ;;
esac

echo "========================================================================="
echo "  Training config: $MODE"
echo "  vocab_size=$VOCAB_SIZE  d_model=$D_MODEL  layers=$NUM_LAYERS  heads=$NUM_HEADS"
echo "  context_length=$CONTEXT_LENGTH  d_ff=$D_FF  batch_size=$BATCH_SIZE"
echo "  max_iters=$MAX_ITERS  max_lr=$MAX_LR  min_lr=$MIN_LR"
echo "  device=$DEVICE"
echo "========================================================================="

uv run python -m "$TRAIN_SCRIPT" \
    --vocab_size "$VOCAB_SIZE" \
    --context_length "$CONTEXT_LENGTH" \
    --d_model "$D_MODEL" \
    --num_layers "$NUM_LAYERS" \
    --num_heads "$NUM_HEADS" \
    --d_ff "$D_FF" \
    --rope_theta "$ROPE_THETA" \
    --max_lr "$MAX_LR" \
    --min_lr "$MIN_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --batch_size "$BATCH_SIZE" \
    --max_iters "$MAX_ITERS" \
    --warmup_iters "$WARMUP_ITERS" \
    --grad_clip "$GRAD_CLIP" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --device "$DEVICE" \
    --log_interval 10 \
    --eval_interval 500 \
    --checkpoint_interval 1000 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    $WANDB_FLAGS
