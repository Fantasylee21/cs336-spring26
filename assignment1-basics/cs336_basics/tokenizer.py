import multiprocessing as mp
import regex as re
from collections import Counter, defaultdict

# Module-level constant for the GPT-2 pretokenization pattern, reused
# by _pretokenize_chunk (needed for multiprocessing pickling).
_GPT2_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _pretokenize_chunk(segments):
    """Pretokenize a list of text segments, returning a word-frequency Counter.

    Defined at module level so it can be pickled by multiprocessing.
    """
    com_pat = re.compile(_GPT2_PATTERN)
    word_counts = Counter()
    for segment in segments:
        for match in com_pat.finditer(segment):
            word_bytes = tuple(match.group().encode("utf-8"))
            word_counts[word_bytes] += 1
    return word_counts


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # GPT-2 pretokenization regex
        self.pretoken_pattern = re.compile(_GPT2_PATTERN)

    @classmethod
    def train_bpe(cls, file, vocab_size, special_tokens, **kwargs):
        """Train a BPE tokenizer on the given file."""
        if vocab_size < 256 + len(special_tokens):
            raise ValueError(
                "vocab_size must be at least 256 + len(special_tokens) "
                "to include all byte values and special tokens."
            )

        special_tokens = special_tokens or []
        text = cls._read_file(file)
        segments = cls._split_by_special_tokens(text, special_tokens)

        word_counts = cls._count_pretokens(segments, special_tokens)
        vocab, bytes_to_id, word_ids = cls._init_vocab_and_word_ids(word_counts, special_tokens)

        num_merges = vocab_size - len(vocab)
        merges = cls._run_bpe_merges(word_ids, word_counts, vocab, bytes_to_id, num_merges)

        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------------
    # Step helpers for train_bpe
    # ------------------------------------------------------------------

    @staticmethod
    def _read_file(file):
        text = file.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        return text

    @staticmethod
    def _split_by_special_tokens(text, special_tokens):
        if not special_tokens:
            return [text]
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped = [re.escape(t) for t in sorted_tokens]
        pattern = re.compile('(' + '|'.join(escaped) + ')')
        return re.split(pattern, text)

    @staticmethod
    def _count_pretokens(segments, special_tokens):
        special_token_set = set(special_tokens)

        # Filter out special-token segments (they are already in the vocab).
        text_segments = [s for s in segments if s not in special_token_set]

        if not text_segments:
            return Counter()

        # Use multiprocessing when there are enough segments to justify the overhead.
        num_workers = min(mp.cpu_count(), len(text_segments), 8)

        if num_workers <= 1:
            return _pretokenize_chunk(text_segments)

        # Distribute segments round-robin for load balancing.
        chunks = [[] for _ in range(num_workers)]
        for i, seg in enumerate(text_segments):
            chunks[i % num_workers].append(seg)
        chunks = [c for c in chunks if c]

        with mp.Pool(processes=len(chunks)) as pool:
            results = pool.map(_pretokenize_chunk, chunks)

        word_counts = Counter()
        for result in results:
            word_counts.update(result)
        return word_counts

    @staticmethod
    def _init_vocab_and_word_ids(word_counts, special_tokens):
        vocab = {}
        bytes_to_id = {}

        # Special tokens get the first IDs (e.g., <|endoftext|> at 0).
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            new_id = len(vocab)
            vocab[new_id] = token_bytes
            bytes_to_id[token_bytes] = new_id

        # Byte tokens start after the special tokens.
        byte_offset = len(vocab)
        for i in range(256):
            new_id = byte_offset + i
            vocab[new_id] = bytes([i])
            bytes_to_id[bytes([i])] = new_id

        word_ids = {}
        for word in word_counts:
            ids = [bytes_to_id[bytes([b])] for b in word]
            word_ids[word] = ids

        return vocab, bytes_to_id, word_ids

    @staticmethod
    def _run_bpe_merges(word_ids, word_counts, vocab, bytes_to_id, num_merges):
        merges = []

        # Build initial pair counts and pair -> words index for incremental updates
        pair_counts = Counter()
        pair_to_words = defaultdict(set)

        for word, ids in word_ids.items():
            cnt = word_counts[word]
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_counts[pair] += cnt
                pair_to_words[pair].add(word)

        for _ in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(
                pair_counts.items(),
                key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]),
            )[0]

            new_id = len(vocab)
            t1_bytes = vocab[best_pair[0]]
            t2_bytes = vocab[best_pair[1]]
            merged_bytes = t1_bytes + t2_bytes

            vocab[new_id] = merged_bytes
            bytes_to_id[merged_bytes] = new_id
            merges.append((t1_bytes, t2_bytes))

            # Only update words that actually contain the merged pair
            affected_words = list(pair_to_words.get(best_pair, set()))
            if not affected_words:
                pair_counts.pop(best_pair, None)
                pair_to_words.pop(best_pair, None)
                continue

            for word in affected_words:
                ids = word_ids[word]
                cnt = word_counts[word]

                # Remove this word's old pair contributions
                for i in range(len(ids) - 1):
                    old_pair = (ids[i], ids[i + 1])
                    pair_counts[old_pair] -= cnt

                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_to_words.pop(old_pair, None)
                    else:
                        words_with_pair = pair_to_words.get(old_pair)
                        if words_with_pair is not None:
                            words_with_pair.discard(word)

                # Apply merge left-to-right
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1

                word_ids[word] = new_ids

                # Add new pair contributions
                for i in range(len(new_ids) - 1):
                    new_pair = (new_ids[i], new_ids[i + 1])
                    pair_counts[new_pair] += cnt
                    pair_to_words[new_pair].add(word)

        return merges
