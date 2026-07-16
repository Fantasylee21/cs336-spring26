from collections.abc import Iterator
import multiprocessing as mp
import os
from typing import BinaryIO, Iterable
import regex as re
from collections import Counter, defaultdict
import json

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


def _split_by_special_tokens_static(text, special_tokens):
    """Module-level helper for splitting text around special tokens (pickleable)."""
    if not special_tokens:
        return [text]
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_tokens]
    pattern = re.compile('(' + '|'.join(escaped) + ')')
    return re.split(pattern, text)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Chunk the file into parts that can be counted independently.

    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def _pretokenize_file_chunk(args):
    """Worker for parallel pretokenization: read a file chunk and count pretokens."""
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    text = chunk_bytes.decode("utf-8", errors="replace")

    special_token_set = set(special_tokens)
    segments = _split_by_special_tokens_static(text, special_tokens)
    text_segments = [s for s in segments if s not in special_token_set]

    if not text_segments:
        return Counter()

    com_pat = re.compile(_GPT2_PATTERN)
    word_counts = Counter()
    for segment in text_segments:
        for match in com_pat.finditer(segment):
            word_bytes = tuple(match.group().encode("utf-8"))
            word_counts[word_bytes] += 1
    return word_counts


def _str_to_bytes(token_str):
    """Reverse the GPT-2 byte-to-display-string convention.

    ``_bytes_to_str`` in train_bpe.py maps:
      - 0x20 (space) → 'Ġ' (U+0120)
      - 0x21–0x7E (printable ASCII) → chr(b)
      - all other bytes → chr(b + 256)
    This function undoes that mapping.
    """
    result = []
    for c in token_str:
        cp = ord(c)
        if cp == 0x120:          # 'Ġ' → space
            result.append(0x20)
        elif 0x21 <= cp <= 0x7E:  # printable ASCII → itself
            result.append(cp)
        else:                     # shifted bytes → cp - 256
            result.append(cp - 256)
    return bytes(result)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # GPT-2 pretokenization regex
        self.pretoken_pattern = re.compile(_GPT2_PATTERN)

        # Reverse mapping: bytes -> token ID, built once for efficient encoding
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        # Precomputed set of special tokens (avoid re-creating per encode call)
        self.special_set = set(self.special_tokens)

        # Precompile special-token split pattern (avoid re-compiling per call)
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(t) for t in sorted_tokens]
            self._special_pattern = re.compile('(' + '|'.join(escaped) + ')')
        else:
            self._special_pattern = None

        # Precompute merge priority lookup: (id1, id2) -> (rank, merged_id)
        # This lets _encode_word find the best merge in O(L^2) instead of O(M*L).
        self._merge_rank = {}   # (id1, id2) -> rank (lower = higher priority)
        self._merge_result = {} # (id1, id2) -> merged_id
        for rank, (t1_bytes, t2_bytes) in enumerate(self.merges):
            id1 = self.bytes_to_id.get(t1_bytes)
            id2 = self.bytes_to_id.get(t2_bytes)
            merged = self.bytes_to_id.get(t1_bytes + t2_bytes)
            if id1 is not None and id2 is not None and merged is not None:
                self._merge_rank[(id1, id2)] = rank
                self._merge_result[(id1, id2)] = merged

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens."""
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
        vocab = {v: _str_to_bytes(k) for k, v in vocab_dict.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                t1, t2 = line.strip().split()
                merges.append((_str_to_bytes(t1), _str_to_bytes(t2)))

        return cls(vocab, merges, special_tokens)
    
    def _encode_word(self, word_bytes):
        """Encode a single pretokenized word (as bytes) into token IDs using learned BPE merges.

        Uses precomputed merge priorities to find the best merge at each step,
        giving O(L^2) per word instead of O(M*L) where M is the total merge count.
        """
        ids = [self.bytes_to_id[bytes([b])] for b in word_bytes]
        if len(ids) <= 1:
            return ids

        merge_rank = self._merge_rank
        merge_result = self._merge_result

        while True:
            best_rank = float('inf')
            best_i = -1
            best_pair = None

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_i = i
                    best_pair = pair

            if best_i == -1:
                break

            ids[best_i] = merge_result[best_pair]
            del ids[best_i + 1]

        return ids

    def encode(self, text: str):
        """Pretokenize the input text and return a list of token IDs according to the tokenizer's vocabulary and merges."""
        chunks = self._split_text(text)
        token_ids = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_set:
                token_ids.append(self.bytes_to_id[chunk.encode("utf-8")])
            else:
                for segment in self.pretoken_pattern.findall(chunk):
                    token_ids.extend(self._encode_word(segment.encode("utf-8")))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into memory
        """
        for text in iterable:
            chunks = self._split_text(text)
            for chunk in chunks:
                if not chunk:
                    continue
                if chunk in self.special_set:
                    yield self.bytes_to_id[chunk.encode("utf-8")]
                else:
                    for segment in self.pretoken_pattern.findall(chunk):
                        yield from self._encode_word(segment.encode("utf-8"))


    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_seq = b"".join(self.vocab[tid] for tid in ids)
        return byte_seq.decode("utf-8", errors="replace")

    @classmethod
    def train_bpe(cls, file, vocab_size, special_tokens, **kwargs):
        """Train a BPE tokenizer on the given file."""
        if vocab_size < 256 + len(special_tokens):
            raise ValueError(
                "vocab_size must be at least 256 + len(special_tokens) "
                "to include all byte values and special tokens."
            )

        special_tokens = special_tokens or []

        # Use the first special token as the chunk-boundary delimiter.
        # If there are no special tokens, read the whole file and split normally.
        if special_tokens:
            split_token = special_tokens[0]
            file_path = file.name
            num_chunks = min(mp.cpu_count(), 8)
            with open(file_path, "rb") as bf:
                boundaries = find_chunk_boundaries(bf, num_chunks, split_token.encode("utf-8"))
            word_counts = cls._count_pretokens(file_path, boundaries, special_tokens)
        else:
            text = cls._read_file(file)
            segments = cls._split_by_special_tokens(text, special_tokens)
            word_counts = cls._count_pretokens_fallback(segments, special_tokens)

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

    def _split_text(self, text):
        """Split text around special tokens using the precompiled pattern."""
        if self._special_pattern is None:
            return [text]
        return re.split(self._special_pattern, text)

    @staticmethod
    def _split_by_special_tokens(text, special_tokens):
        return _split_by_special_tokens_static(text, special_tokens)

    @staticmethod
    def _count_pretokens(file_path, boundaries, special_tokens):
        num_workers = min(mp.cpu_count(), len(boundaries) - 1, 8)

        chunks = [
            (file_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        if num_workers <= 1:
            return _pretokenize_file_chunk(chunks[0])

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_pretokenize_file_chunk, chunks)

        word_counts = Counter()
        for result in results:
            word_counts.update(result)
        return word_counts

    @staticmethod
    def _count_pretokens_fallback(segments, special_tokens):
        """Fallback: pretokenize segments in-memory (used when no special tokens exist)."""
        special_token_set = set(special_tokens)
        text_segments = [s for s in segments if s not in special_token_set]

        if not text_segments:
            return Counter()

        num_workers = min(mp.cpu_count(), len(text_segments), 8)

        if num_workers <= 1:
            return _pretokenize_chunk(text_segments)

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
