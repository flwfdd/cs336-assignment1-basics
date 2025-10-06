from collections import Counter
import os
import pickle
import regex as re
from typing import BinaryIO
import multiprocessing


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenization(
    text: str,
    special_tokens: list[str],
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> Counter[str]:
    """
    A simple pre-tokenization function that removes special tokens and splits on pattern.
    Returns a dictionary mapping each pre-token (called "word") to its count in the text.
    """
    # Split by special tokens and remove them
    special_tokens_pattern = "|".join(
        [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    )
    word_counts = Counter[str]()
    for segment in re.split(special_tokens_pattern, text):
        # Split text by pattern
        for match in re.finditer(pattern, segment):
            word = match.group(0)
            word_counts[word] += 1
    return word_counts


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab: list[bytes] = [
        *[tok.encode("utf-8") for tok in special_tokens],
        *[bytes([i]) for i in range(256)],
    ]
    word_counts: Counter[str] = Counter()

    # Pre-tokenization
    num_processes = 8
    with multiprocessing.Manager() as manager:
        results = manager.list()
        processes: list[multiprocessing.Process] = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, special_tokens[0].encode("utf-8")
            )

            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8")

                def worker(idx, res_list, txt, sp_tokens):
                    try:
                        res = pre_tokenization(txt, sp_tokens)
                        res_list.append(res)
                    except Exception as e:
                        print(f"Error in process {idx}: {e}")

                p = multiprocessing.Process(
                    target=worker, args=(i, results, chunk, special_tokens)
                )
                processes.append(p)
                p.start()
        for p in processes:
            p.join()
        for res in results:
            word_counts += res

    # Init BPE merges
    merges: list[tuple[bytes, bytes]] = []  # List of (byte pair) merges
    words: dict[str, dict[str, str | list[bytes] | int]] = (
        {}
    )  # List of dicts, each with "tokens" and "count"
    pair2word: dict[tuple[bytes, bytes], Counter[str]] = (
        {}
    )  # Reverse lookup from pair to word dicts
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for word_str, count in word_counts.items():
        if (
            word_str.encode("utf-8") in vocab
        ):  # Skip tokens that are already in the vocabulary
            continue
        tokens = [
            bytes([i]) for i in word_str.encode("utf-8")
        ]  # List of bytes [b'h', b'e', b'y']
        word = {
            "str": word_str,
            "tokens": tokens,
            "count": count,
        }
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
            pair2word.setdefault(pair, Counter())[word_str] += 1
        words[word_str] = word

    # BPE merges loop
    while len(vocab) < vocab_size:
        # Find the most common and exicographically greater byte pair
        most_common_pair: tuple[bytes, bytes] | None = None
        most_common_pair_count = 0
        for pair, count in pair_counts.items():
            if (
                not most_common_pair
                or count > most_common_pair_count
                or (count == most_common_pair_count and pair > most_common_pair)
            ):
                most_common_pair = pair
                most_common_pair_count = count

        # Merge the most common pair in all words that contain it
        if most_common_pair is None:
            print(
                f"Ran out of pairs to merge at vocab size {len(vocab)} < {vocab_size}"
            )
            break
        left_token, right_token = most_common_pair
        new_token = left_token + right_token
        vocab.append(new_token)
        merges.append((left_token, right_token))

        # Update all words and pair_counts e.g. a, b, c, d -> a, bc, d
        word_strs = list(pair2word.get(most_common_pair, {}).keys())
        for word_str in word_strs:
            word = words[word_str]
            tokens: list[bytes] = word["tokens"]  # type: ignore
            count: int = word["count"]  # type: ignore
            new_tokens = []

            # Remove all pairs
            for i in range(len(tokens) - 1):
                old_pair = (tokens[i], tokens[i + 1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] == 0:
                    pair_counts.pop(old_pair)
                pair2word[old_pair][word_str] -= 1
                if pair2word[old_pair][word_str] == 0:
                    del pair2word[old_pair][word_str]
                if not pair2word[old_pair]:
                    pair2word.pop(old_pair)

            # Create new tokens
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == left_token and tokens[i + 1] == right_token:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens += tokens[i:]
            word["tokens"] = new_tokens

            # Add new pairs
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                pair_counts[new_pair] += count
                pair2word.setdefault(new_pair, Counter())[word_str] += 1
    return {i: tok for i, tok in enumerate(vocab)}, merges


if __name__ == "__main__":
    # vocab, merges = train_bpe("test.txt", 256 + 1 + 6, ["<|endoftext|>"])
    # vocab, merges = train_bpe(
    #     "../tests/fixtures/corpus.en", 256 + 1 + 6, ["<|endoftext|>"]
    # )
    vocab, merges = train_bpe(
        "../data/TinyStoriesV2-GPT4-train.txt.txt", 10000, ["<|endoftext|>"]
    )
    with open("bpe_vocab.txt", "w", encoding="utf-8") as f:
        for i, tok in vocab.items():
            f.write(f"{i}\t{tok}\n")
    with open("bpe_merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left} {right}\n")
    with open("bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    # print("Vocab:")
    # for i, tok in vocab.items():
    #     print(f"  {i}: {tok}")
    # print("\nMerges:")
    # for left, right in merges:
    #     print(f"  {left} + {right} -> {left + right}")
