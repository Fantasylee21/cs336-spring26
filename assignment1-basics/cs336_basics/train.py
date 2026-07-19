import argparse
import json
import math
import os
import time

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data_loader import data_loading
from cs336_basics.loss import cross_entropy_loss, gradient_clipping, perplexity
from cs336_basics.optim import AdamW, learning_rate_schedule
from cs336_basics.transformer import transformer_lm


def _load_dataset(path: str) -> np.ndarray:
    """Memory-efficient dataset loading via np.memmap for .npy, or load+convert for .txt."""
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    elif path.endswith(".txt"):
        # For .txt files, read once and convert to token IDs
        # Assumes the .txt is already tokenized to whitespace-separated ints
        tokens = []
        with open(path) as f:
            for line in f:
                tokens.extend(int(t) for t in line.split())
        return np.array(tokens, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported data format: {path}. Expected .npy or .txt")


def _evaluate(model, dataset, batch_size, context_length, device, eval_iters=10):
    """Evaluate model on a dataset, returning average loss and perplexity."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = data_loading(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy_loss(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            total_loss += loss.item()
    model.train()
    avg_loss = total_loss / eval_iters
    return avg_loss, math.exp(avg_loss)


def train_model():
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model from scratch."
    )

    # ---- Model hyperparameters ----
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=None,
                        help="FFN hidden dim (default: nearest multiple of 64 to 8/3 * d_model)")
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # ---- Optimizer hyperparameters ----
    parser.add_argument("--lr", "--max_lr", type=float, default=1e-3, dest="max_lr")
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # ---- Training schedule ----
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=100000)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None,
                        help="Total cosine cycle iters (default: max_iters)")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # ---- Data ----
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (.npy or .txt)")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation data (.npy or .txt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # ---- Logging & checkpointing ----
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log training loss every N iterations")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Evaluate on validation set every N iterations")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a checkpoint file")

    args = parser.parse_args()

    # Auto-compute defaults
    if args.d_ff is None:
        raw = int(8 / 3 * args.d_model)
        args.d_ff = ((raw + 63) // 64) * 64  # nearest multiple of 64

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Load datasets ----
    print(f"Loading training data from {args.train_data}")
    train_dataset = _load_dataset(args.train_data)
    print(f"Loading validation data from {args.val_data}")
    val_dataset = _load_dataset(args.val_data)
    print(f"Train tokens: {len(train_dataset):,}  Val tokens: {len(val_dataset):,}")

    # ---- Build model ----
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ---- Build optimizer ----
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ---- Resume from checkpoint ----
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    # ---- Training loop ----
    model.train()
    print(f"Starting training on {args.device} for {args.max_iters} iterations "
          f"(batch_size={args.batch_size}, context_length={args.context_length})")
    print(f"Tokens per iter: {args.batch_size * args.context_length:,}")

    total_tokens = 0
    t_start = time.time()

    for it in range(start_iter, args.max_iters):
        # Update learning rate
        current_lr = learning_rate_schedule(
            it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Get batch
        x, y = data_loading(
            train_dataset, args.batch_size, args.context_length, args.device
        )
        total_tokens += x.numel()

        # Forward
        logits = model(x)
        loss = cross_entropy_loss(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # ---- Logging ----
        if it % args.log_interval == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            print(
                f"iter {it:7d} | "
                f"loss {loss.item():.4f} | "
                f"ppl {math.exp(loss.item()):.2f} | "
                f"lr {current_lr:.2e} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )

        # ---- Validation ----
        if it % args.eval_interval == 0 and it > 0:
            val_loss, val_ppl = _evaluate(
                model, val_dataset, args.batch_size, args.context_length, args.device
            )
            elapsed = time.time() - t_start
            print(
                f"--- VAL iter {it:7d} | "
                f"loss {val_loss:.4f} | "
                f"ppl {val_ppl:.2f} | "
                f"elapsed {elapsed:.0f}s"
            )

        # ---- Checkpoint ----
        if it % args.checkpoint_interval == 0 and it > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{it:07d}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"--- Checkpoint saved to {ckpt_path}")

    # ---- Final checkpoint ----
    final_path = os.path.join(args.checkpoint_dir, f"checkpoint_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")

    # ---- Final evaluation ----
    val_loss, val_ppl = _evaluate(
        model, val_dataset, args.batch_size, args.context_length, args.device
    )
    print(f"Final validation -- loss: {val_loss:.4f}  ppl: {val_ppl:.2f}")
    print(f"Total training time: {time.time() - t_start:.0f}s")
    print(f"Total tokens processed: {total_tokens:,}")


if __name__ == "__main__":
    train_model()
