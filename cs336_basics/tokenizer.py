import functools
import pickle
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Type

import regex as re


class Tokenizer:
    """
    A simple BPE-style tokenizer interface.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally)
        a list of special tokens.
        """
        self.vocab: Dict[int, bytes] = vocab
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.special_tokens: List[str] = special_tokens or []

        self.token2id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge2rank: Dict[Tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }

        self.token_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_token_pattern = re.compile(self.token_pattern)
        self.special_tokens_pattern = (
            (
                "("
                + "|".join(
                    [
                        re.escape(f"{token}")
                        for token in sorted(self.special_tokens, key=len, reverse=True)
                    ]
                )
                + ")"
            )
            if self.special_tokens
            else "(?!)"  # a regex that matches nothing
        )  # wrap in capturing group to include in the split results
        self.compiled_special_tokens_pattern = re.compile(self.special_tokens_pattern)

    @classmethod
    def from_files(
        cls: Type["Tokenizer"],
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output) and
        (optionally) a list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    @functools.lru_cache(maxsize=16 * 1024)
    def encode_token(self, word: str) -> List[int]:
        """
        Encode a single token into a sequence of token IDs.
        """
        tokens = [bytes([i]) for i in word.encode("utf-8")]

        def get_merge(tokens: List[bytes]) -> Optional[Tuple[bytes, bytes]]:
            min_rank = float("inf")
            candidate: Optional[Tuple[bytes, bytes]] = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge2rank and self.merge2rank[pair] < min_rank:
                    min_rank = self.merge2rank[pair]
                    candidate = pair
            return candidate

        while True:
            merge = get_merge(tokens)
            if merge is None:
                break
            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge[0]
                    and tokens[i + 1] == merge[1]
                ):
                    new_tokens.append(merge[0] + merge[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.token2id[token] for token in tokens]

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        token_ids = []

        for segment in self.compiled_special_tokens_pattern.split(text):
            if segment in self.special_tokens:
                token_ids.append(self.token2id[segment.encode("utf-8")])
                continue
            # Split text by pattern
            for match in self.compiled_token_pattern.finditer(segment):
                word = match.group(0)
                token_ids += self.encode_token(word)
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a file handle yielding lines), return a
        generator that lazily yields token IDs.

        This is required for memory-efficient tokenization of large inputs that cannot
        be fully loaded into memory.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs back into a Unicode string.
        """
        bytes_list = [self.vocab[token_id] for token_id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")


if __name__ == "__main__":
    import array
    import multiprocessing
    import os

    import numpy as np
    import tqdm

    def encode_with_text(t):
        return (t, tokenizer.encode(t))

    def accumulate_iter(iterable, min_size):
        """
        Accumulate strings from an iterable until reaching at least min_size,
        """
        batch = ""
        for text in iterable:
            batch += text
            if len(batch) >= min_size:
                yield batch
                batch = ""
        if batch:
            yield batch

    tokenizer = Tokenizer.from_files(
        "../data/owt_train/bpe_vocab.pkl",
        "../data/owt_train/bpe_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    token_ids_buf = array.array("H")

    file_path = "../data/owt_valid.txt"
    with open(file_path, "r") as f:
        f.seek(0, os.SEEK_END)
        bytes_len = f.tell()
        f.seek(0)

        with tqdm.tqdm(
            total=bytes_len,
            unit="char",
            desc="Encoding",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:
            with multiprocessing.Pool(processes=8) as pool:
                batch_ids = pool.imap(
                    encode_with_text, accumulate_iter(f, 128 * 1024)
                )  # parallel processing
                for text, ids in batch_ids:
                    token_ids_buf.extend(ids)
                    pbar.update(len(text.encode("utf-8")))
    token_ids = np.frombuffer(token_ids_buf, dtype=np.uint16)
    np.save("token_ids.npy", token_ids)
    print(f"Compression ratio: {bytes_len/(token_ids.size):.2f}")
    exit()

    token_ids = tokenizer.encode("Hello, 世界！<|endoftext|>")
    print("Token IDs:")
    print(token_ids)
    print("Tokens:")
    print([tokenizer.vocab[token_id] for token_id in token_ids])
    text = tokenizer.decode(token_ids)
    print("Decoded text:")
    print(text)
    print("Decoded text:")
    print(text)
