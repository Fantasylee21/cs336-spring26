from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="NSYS profiling script for Transformer")
    parser.add_argument("--model-size", choices=list(MODEL_SIZES.keys()), required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--profile-mode", choices=["forward", "forward-backward", "full"], default="full")
    args = parser.parse_args()

    device = "cuda"
    cfg = MODEL_SIZES[args.model_size]

    print(f"Model: {args.model_size} | d_model={cfg['d_model']} d_ff={cfg['d_ff']} "
          f"layers={cfg['num_layers']} heads={cfg['num_heads']}")
    print(f"Seq len: {args.seq_len} | Batch: {args.batch_size} | Mode: {args.profile_mode}")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    x, y = make_random_batch(args.batch_size, args.seq_len, args.vocab_size, device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    loss_fn = nn.CrossEntropyLoss()

    # --- warmup ---
    for _ in range(args.warmup_steps):
        if args.profile_mode == "forward":
            model.train()
            logits = model(x)
            loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            if args.profile_mode == "full":
                optimizer.step()
        torch.cuda.synchronize()

    # --- profiled section ---
    torch.cuda.synchronize()

    for step in range(args.measure_steps):
        torch.cuda.nvtx.range_push(f"step_{step}")

        if args.profile_mode == "forward":
            torch.cuda.nvtx.range_push("forward")
            model.train()
            logits = model(x)
            torch.cuda.nvtx.range_pop()  # forward

            torch.cuda.nvtx.range_push("loss")
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            torch.cuda.nvtx.range_pop()  # loss

        elif args.profile_mode == "forward-backward":
            torch.cuda.nvtx.range_push("forward")
            model.train()
            logits = model(x)
            torch.cuda.nvtx.range_pop()  # forward

            torch.cuda.nvtx.range_push("loss_backward")
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.cuda.nvtx.range_pop()  # loss_backward

        else:  # full
            torch.cuda.nvtx.range_push("forward")
            model.train()
            logits = model(x)
            torch.cuda.nvtx.range_pop()  # forward

            torch.cuda.nvtx.range_push("loss_backward")
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.cuda.nvtx.range_pop()  # loss_backward

            torch.cuda.nvtx.range_push("optimizer")
            optimizer.step()
            torch.cuda.nvtx.range_pop()  # optimizer

            optimizer.zero_grad()

        torch.cuda.nvtx.range_pop()  # step
        torch.cuda.synchronize()

    # --- timing (no nvtx, cleaner) ---
    timings = {"forward": [], "backward": [], "full": []}
    timer = timeit.default_timer

    for _ in range(args.measure_steps):
        torch.cuda.synchronize()
        t0 = timer()
        model.train()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        torch.cuda.synchronize()
        t1 = timer()
        timings["forward"].append(t1 - t0)

        loss.backward()
        torch.cuda.synchronize()
        t2 = timer()
        timings["backward"].append(t2 - t1)

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t3 = timer()
        timings["full"].append(t3 - t0)

    for phase, times in timings.items():
        if times:
            print(f"[{phase}] mean={statistics.mean(times)*1000:.2f}ms "
                  f"stdev={statistics.stdev(times)*1000:.2f}ms "
                  f"min={min(times)*1000:.2f}ms max={max(times)*1000:.2f}ms")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
