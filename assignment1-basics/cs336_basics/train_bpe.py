import json
import os
import time
import tracemalloc

from cs336_basics.tokenizer import Tokenizer


def _bytes_to_str(token_bytes):
    """Convert a byte sequence to a displayable string.

    Uses the GPT-2 / HuggingFace / Qwen convention:
      - Printable ASCII except space (0x21-0x7E) → chr(b)
      - Space (0x20) → 'Ġ' (U+0120)
      - Other control chars and high bytes → chr(b + 256)
    """
    chars = []
    for b in token_bytes:
        if 0x21 <= b <= 0x7E:
            chars.append(chr(b))
        elif b == 0x20:
            chars.append('Ġ')  # Ġ
        else:
            chars.append(chr(b + 256))
    return ''.join(chars)


def save_tokenizer_files(vocab, merges, out_dir):
    """Serialize vocabulary and merges to disk.

    vocab.json maps displayable token strings to IDs (GPT-2 / Qwen convention).
    merges.txt has one merge per line: "token1 token2".
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save vocab: {int: bytes} -> {str: int}
    vocab_dict = {}
    for token_id, token_bytes in sorted(vocab.items()):
        vocab_dict[_bytes_to_str(token_bytes)] = token_id

    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    # Save merges: [(bytes, bytes), ...] -> "token1 token2\\n" per line
    merges_path = os.path.join(out_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for t1, t2 in merges:
            s1 = _bytes_to_str(t1)
            s2 = _bytes_to_str(t2)
            f.write(f"{s1} {s2}\n")

    print(f"Saved vocab ({len(vocab_dict)} tokens) to {vocab_path}")
    print(f"Saved merges ({len(merges)} merges) to {merges_path}")


def main():
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    output_dir = "cs336_basics/output/owt_train"

    # Train BPE tokenizer
    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {input_path}...")
    tracemalloc.start()
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as f:
        tokenizer = Tokenizer.train_bpe(f, vocab_size, special_tokens)

    elapsed = time.time() - t0
    _, peak_mb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    vocab = tokenizer.vocab
    merges = tokenizer.merges

    # Save results
    save_tokenizer_files(vocab, merges, output_dir)

    # Report statistics
    print(f"\n--- Training Statistics ---")
    print(f"Training time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    print(f"Peak memory: {peak_mb / (1024 * 1024):.1f} MB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Find the longest token in the vocabulary
    longest_id = max(vocab, key=lambda tid: len(vocab[tid]))
    longest_bytes = vocab[longest_id]
    print(f"Longest token ID: {longest_id}")
    print(f"Longest token length: {len(longest_bytes)} bytes")

    # Try to decode the longest token for inspection
    try:
        longest_str = longest_bytes.decode("utf-8")
        print(f"Longest token (UTF-8): {repr(longest_str)}")
    except UnicodeDecodeError:
        print(f"Longest token (hex): {longest_bytes.hex()}")

    # Print a few example tokens
    print(f"\n--- Sample Vocabulary ---")
    for tid in sorted(vocab)[:20]:
        tb = vocab[tid]
        try:
            ts = tb.decode("utf-8")
            print(f"  {tid}: {repr(ts)}")
        except UnicodeDecodeError:
            print(f"  {tid}: <{tb.hex()}>")


if __name__ == "__main__":
    main()
