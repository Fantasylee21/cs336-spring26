"""Inference script for a trained Transformer language model.

Supports batch prompts from CLI args or interactive mode.
"""

import argparse
import sys

import torch

from cs336_basics.generate import generate
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import transformer_lm
from cs336_basics.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer LM."
    )

    # ---- Model architecture (must match training config) ----
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # ---- Checkpoint & tokenizer ----
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to vocab.json")
    parser.add_argument("--merges", type=str, required=True,
                        help="Path to merges.txt")
    parser.add_argument("--special_tokens", type=str, default="<|endoftext|>",
                        help="Comma-separated list of special tokens")

    # ---- Generation params ----
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt. If omitted, enters interactive mode.")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # ---- Load tokenizer ----
    print(f"Loading tokenizer from {args.vocab} / {args.merges}")
    special_tokens = [t.strip() for t in args.special_tokens.split(",") if t.strip()]
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens)
    print(f"Tokenizer loaded: {len(tokenizer.vocab)} vocab, {len(tokenizer.merges)} merges")

    # ---- Build model ----
    print(f"Building model: d_model={args.d_model}, layers={args.num_layers}, "
          f"heads={args.num_heads}, d_ff={args.d_ff}")
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
    ).to(args.device)

    # ---- Load checkpoint ----
      # Not needed for inference
    ckpt = load_checkpoint(args.checkpoint, model)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint loaded ({args.checkpoint})")
    print(f"Iteration at save: {ckpt}")
    print(f"Model parameters: {num_params:,}")

    # ---- Generate ----
    gen_kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": args.device,
        "end_token": special_tokens[0] if special_tokens else "<|endoftext|>",
    }

    if args.prompt:
        # Single prompt mode
        output = generate(model, tokenizer, args.prompt, **gen_kwargs)
        print("\n" + "=" * 60)
        print(output)
        print("=" * 60)
    else:
        # Interactive mode
        print("\nInteractive mode. Type a prompt and press Enter (Ctrl+D to exit).")
        print(f"Params: temperature={args.temperature}, top_p={args.top_p}, "
              f"max_tokens={args.max_tokens}\n")
        try:
            while True:
                prompt = input(">>> ").strip()
                if not prompt:
                    continue
                output = generate(model, tokenizer, prompt, **gen_kwargs)
                print(f"\n{output}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")


if __name__ == "__main__":
    main()
