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


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def make_random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def run_benchmark(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    warmup_steps: int,
    measure_steps: int,
    device: str,
) -> dict[str, float]:
    optimizer = AdamW(model.parameters(), lr=1e-3)

    def forward_step():
        model.train()
        logits = model(x)
        nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )

    def forward_backward_step():
        model.train()
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()

    def full_step():
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        loss.backward()
        optimizer.step()

    mode_to_fn = {
        "forward": forward_step,
        "forward-backward": forward_backward_step,
        "full": full_step,
    }
    step_fn = mode_to_fn[mode]

    for _ in range(warmup_steps):
        step_fn()
        sync_device(device)

    timings = []
    timer = timeit.default_timer
    for _ in range(measure_steps):
        sync_device(device)
        start = timer()
        step_fn()
        sync_device(device)
        end = timer()
        timings.append(end - start)

    return {
        "mean": statistics.mean(timings),
        "stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "min": min(timings),
        "max": max(timings),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer forward/backward/optimizer steps")
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_SIZES.keys()),
        default="small",
        help="Which model size to benchmark",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "forward-backward", "full"],
        required=True,
        help="forward=forward only, forward-backward=forward+backward, full=forward+backward+optimizer",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--all-sizes", action="store_true", help="Benchmark all model sizes")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"Warmup steps: {args.warmup_steps}, Measure steps: {args.measure_steps}")

    sizes_to_run = MODEL_SIZES if args.all_sizes else {args.model_size: MODEL_SIZES[args.model_size]}

    for name, cfg in sizes_to_run.items():
        print(f"\n--- {name} ---")
        print(f"  d_model={cfg['d_model']}, d_ff={cfg['d_ff']}, "
              f"num_layers={cfg['num_layers']}, num_heads={cfg['num_heads']}")

        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.seq_len,
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        x, y = make_random_batch(args.batch_size, args.seq_len, args.vocab_size, device)

        results = run_benchmark(
            model, x, y, args.mode,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            device=device,
        )

        print(f"  Mean: {results['mean']*1000:.2f} ms")
        print(f"  Stdev: {results['stdev']*1000:.2f} ms")
        print(f"  Min: {results['min']*1000:.2f} ms")
        print(f"  Max: {results['max']*1000:.2f} ms")

        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
