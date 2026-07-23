from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import timeit

import torch
import torch.nn as nn

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

MODEL_SIZES = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def make_random_batch(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def print_memory_stats(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{prefix}] allocated={allocated:.1f} MiB  reserved={reserved:.1f} MiB  peak_allocated={peak_allocated:.1f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=list(MODEL_SIZES.keys()), required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument(
        "--profile-mode",
        choices=["forward", "forward-backward", "full"],
        default="full",
    )
    parser.add_argument(
        "--amp", choices=["none", "fp16", "bf16"], default="none",
    )
    parser.add_argument(
        "--memory-profile", action="store_true",
        help="Record memory history and dump snapshot for pytorch.org/memory_viz",
    )
    parser.add_argument(
        "--output-dir", default="./memory_snapshots",
    )
    args = parser.parse_args()

    device = "cuda"
    cfg = MODEL_SIZES[args.model_size]
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )

    print(f"Model: {args.model_size} | d_model={cfg['d_model']} d_ff={cfg['d_ff']} "
          f"layers={cfg['num_layers']} heads={cfg['num_heads']}")
    print(f"Seq len: {args.seq_len} | Batch: {args.batch_size} | Mode: {args.profile_mode}")
    print(f"AMP: {args.amp} | Memory profile: {args.memory_profile}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params * 4 / 1024**2:.1f} MiB FP32)")

    x, y = make_random_batch(args.batch_size, args.seq_len, args.vocab_size, device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # --- warmup ---
    for _ in range(args.warmup_steps):
        if args.profile_mode == "forward":
            model.train()
            with autocast_ctx:
                logits = model(x)
            loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            model.train()
            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            if args.profile_mode == "full":
                optimizer.step()
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --- Memory profiling ---
    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=200000)

    torch.cuda.synchronize()

    # --- Timed & profiled step ---
    timings = {"forward": [], "backward": [], "full": []}
    timer = timeit.default_timer

    for step in range(args.measure_steps):
        torch.cuda.nvtx.range_push(f"step_{step}")

        # Forward
        torch.cuda.nvtx.range_push("forward")
        torch.cuda.synchronize()
        t0 = timer()
        model.train()
        with autocast_ctx:
            logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        torch.cuda.synchronize()
        t1 = timer()
        torch.cuda.nvtx.range_pop()
        timings["forward"].append(t1 - t0)

        if args.profile_mode in ("forward-backward", "full"):
            # Backward
            torch.cuda.nvtx.range_push("backward")
            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()
            t2 = timer()
            torch.cuda.nvtx.range_pop()
            timings["backward"].append(t2 - t1)

            if args.profile_mode == "full":
                # Optimizer
                torch.cuda.nvtx.range_push("optimizer")
                torch.cuda.synchronize()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                t3 = timer()
                torch.cuda.nvtx.range_pop()
                timings["full"].append(t3 - t0)
            else:
                t3 = t2

        torch.cuda.nvtx.range_pop()  # step

    # --- Print timing ---
    for phase, times in timings.items():
        if times:
            print(f"[{phase}] mean={statistics.mean(times)*1000:.2f}ms "
                  f"stdev={statistics.stdev(times)*1000:.2f}ms "
                  f"min={min(times)*1000:.2f}ms max={max(times)*1000:.2f}ms")

    # --- Memory stats ---
    print()
    print("=== Memory Usage ===")
    print_memory_stats("final")
    print(f"peak_allocated (cumulative): {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")
    print(f"peak_reserved: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MiB")

    # --- Dump memory snapshot ---
    if args.memory_profile:
        snapshot_path = os.path.join(
            args.output_dir,
            f"memory_{args.model_size}_seq{args.seq_len}_{args.profile_mode}_amp{args.amp}.pickle"
        )
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"\nMemory snapshot saved to: {snapshot_path}")
        print(f"Visualize at: https://pytorch.org/memory_viz")

    del model, x, y, logits, loss, optimizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
