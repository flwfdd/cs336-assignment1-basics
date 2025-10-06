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
        self.token2id: Dict[bytes, int] = {v: k for k, v in vocab.items()}

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

    def encode_token(self, word: str) -> List[int]:
        """
        Encode a single token into a sequence of token IDs.
        """
        tokens = [bytes([i]) for i in word.encode("utf-8")]
        token_counter: Counter[bytes] = Counter(tokens)
        for merge in self.merges:
            if token_counter[merge[0]] == 0 or token_counter[merge[1]] == 0:
                continue
            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    new_tokens.append(merge[0] + merge[1])
                    token_counter[merge[0]] -= 1
                    token_counter[merge[1]] -= 1
                    token_counter[merge[0] + merge[1]] += 1
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens + tokens[i:]
        return [self.token2id[token] for token in tokens]

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        token_ids = []
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for segment in re.split(self.special_tokens_pattern, text):
            if segment in self.special_tokens:
                token_ids.append(self.token2id[segment.encode("utf-8")])
                continue
            # Split text by pattern
            for match in re.finditer(pattern, segment):
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


import json
import os


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "../data/owt_train/bpe_vocab.pkl",
        "../data/owt_train/bpe_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )
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
