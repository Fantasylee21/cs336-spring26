from cs336_basics.tokenizer import Tokenizer
import time
import numpy as np


def _sample_docs(filepath, n=10):
    """Read the first *n* non-empty documents from a text file.

    Documents are assumed to be separated by ``<|endoftext|>``.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    docs = [d.strip() for d in text.split("<|endoftext|>") if d.strip()]
    return docs[:n]


def benchmark_throughput(tokenizer, docs):
    """Encode a list of documents and report compression ratio and throughput."""
    total_bytes = 0
    total_tokens = 0
    t0 = time.time()
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    elapsed = time.time() - t0

    ratio = total_bytes / total_tokens if total_tokens else 0
    throughput = total_bytes / elapsed if elapsed else 0
    print(f"  {total_bytes:,} bytes -> {total_tokens:,} tokens "
          f"({ratio:.2f} bytes/token)")
    print(f"  {elapsed:.2f}s, throughput: {throughput:,.2f} bytes/s")
    return ratio, throughput


def encode_and_save(tokenizer, src_path, dst_path, chunk_size=1_000_000):
    """Encode a text file to token IDs via ``encode_iterable`` and save as .npy."""
    print(f"  {src_path} -> {dst_path}")
    t0 = time.time()

    chunks = []
    buf = np.empty(chunk_size, dtype=np.uint16)
    idx = 0
    total = 0
    with open(src_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            buf[idx] = token_id
            idx += 1
            total += 1
            if idx >= chunk_size:
                chunks.append(buf)
                buf = np.empty(chunk_size, dtype=np.uint16)
                idx = 0
                if total % (10 * chunk_size) == 0:
                    print(f"    {total:,} tokens...")
    if idx > 0:
        chunks.append(buf[:idx])

    all_ids = np.concatenate(chunks)
    np.save(dst_path, all_ids)

    elapsed = time.time() - t0
    mb = total * 2 / (1024 * 1024)  # uint16 = 2 bytes each
    print(f"    Done: {total:,} tokens ({mb:.1f} MiB) in {elapsed:.1f}s "
          f"({total / elapsed:.2f} tokens/s)")
    return all_ids


def main():
    vocab_filepath = "cs336_basics/output/owt_train/vocab.json"
    merges_filepath = "cs336_basics/output/owt_train/merges.txt"
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # ---- Throughput benchmark on 10 sampled docs ----
    print("=== Throughput Benchmark (10 documents) ===")
    for dataset_name, data_path in [
        ("TinyStories", "data/TinyStoriesV2-GPT4-valid.txt"),
        ("OpenWebText", "data/owt_valid.txt"),
    ]:
        print(f"\n[{dataset_name}]")
        docs = _sample_docs(data_path, n=10)
        benchmark_throughput(tokenizer, docs)

    # ---- Full dataset encoding ----
    print("\n=== Full Dataset Encoding ===")
    for name, src, dst in [
        ("OpenWebText train", "data/owt_train.txt",
         "data/owt_train.npy"),
        ("OpenWebText valid", "data/owt_valid.txt",
         "data/owt_valid.npy"),
    ]:
        print(f"\n[{name}]")
        encode_and_save(tokenizer, src, dst)


if __name__ == "__main__":
    main()
