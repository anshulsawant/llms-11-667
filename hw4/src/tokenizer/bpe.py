import json
import sys
from collections import Counter
from typing import TypeAlias

import regex as re
from tqdm.auto import trange

Bigram: TypeAlias = tuple[int, int]


def string_to_ascii(s: str) -> list[int]:
    """Convert string to ascii.

    Args:
        s: An ASCII string

    Returns:
        An array of integers (0 <= i < 128) under ASCII encoding
    """
    return list(s.encode("ascii"))


def string_to_unicode_bytes(s: str) -> bytes:
    """Convert string to unicode bytes.

    Args:
        s: A unicode string

    Returns:
        An array of integers (0 <= i < 256) under UTF-8 encoding
    """
    return list(s.encode("utf-8"))


def compute_bigram_statistics(
    token_ids: list[int], counter: Counter | None = None
) -> Counter[Bigram, int]:
    """Compute adjacent token (i.e., bigram) statistics.

    Args:
        token_ids (list[int]): sequence of token ids

    Returns:
        A counter (dict-like object) with frequencies of bigrams

    For examples, [1, 2, 3, 1, 2, 3] -> {(1, 2): 2, (2, 3): 2, (3, 1): 1}
    """

    bigram_counter = Counter() if counter is None else counter
    for left, right in zip(token_ids, token_ids[1:]):
        bigram_counter[(left, right)] += 1
    return bigram_counter


def replace_bigram(token_ids: list[int], bigram: Bigram, bigram_id: int) -> list[int]:
    """Replaces all copies of a bigram with `bigram_id`.

    Args:
        token_ids: List of token ids
        bigram: The bigram to replace. A bigram is a tuple of exactly two token ids.
        bigram_id: New id for the bigram

    Returns:
        list[int]: sequence with bigram replaced
    """

    idx = 0
    new_token_ids = []
    while idx < len(token_ids):
        if token_ids[idx : idx + 2] == list(bigram):
            new_token_ids.append(bigram_id)
            idx += 2
        else:
            new_token_ids.append(token_ids[idx])
            idx += 1
    return new_token_ids


class ASCIIBPETokenizer:
    def __init__(self):
        self.vocab: list[str] = [chr(i) for i in range(128)]
        self.merge_rules: dict[Bigram, int] = {}

    def merge(self, token_ids: list[int]) -> list[int]:
        """Perform one merge in the BPE algorithm.

        Specifically, implement the following:
          1. Find the most frequent bigram B, and create a new token T for it.
            If two bigram have the same frequency, break tie by taking
            the lexicographically smaller bigram.
          2. Replace all occurrences of B in token_ids with T

        Args:
            token_ids (list[int]): Current list of token ids

        Returns:
            list[int]: New list of token ids, after one merge step
        """
        bigram_counter = compute_bigram_statistics(token_ids)
        bigram_to_merge = min(
            bigram_counter, key=lambda tup: (-bigram_counter[tup], tup)
        )
        token_left, token_right = bigram_to_merge
        new_token = self.vocab[token_left] + self.vocab[token_right]

        new_token_id = len(self.vocab)
        self.vocab.append(new_token)
        self.merge_rules[bigram_to_merge] = new_token_id

        return replace_bigram(token_ids, bigram_to_merge, new_token_id)

    def encode(self, text: str) -> list[int]:
        """Convert text to tokens.

        Args:
            text: An arbitrary ASCII string

        Returns:
            Tokens produced under BPE
        """

        assert all(ord(c) < 128 for c in text), "input text is not ASCII"
        token_ids = string_to_ascii(text)
        bigram_stats = compute_bigram_statistics(token_ids)
        while bigram_stats.keys() & self.merge_rules.keys():
            for bigram, bigram_id in self.merge_rules.items():
                if bigram_stats[bigram] > 0:
                    token_ids = replace_bigram(token_ids, bigram, bigram_id)
                    break
            bigram_stats = compute_bigram_statistics(token_ids)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert tokens back to text.

        Args:
            token_ids: A list of token ids.

        Returns:
            str: An ASCII string.
        """
        return "".join([self.vocab[token_id] for token_id in token_ids])

    
    @classmethod
    def from_config(cls, config_file: str):
        """Load tokenizer from a JSON config file.

        Args:
            config_file: Path to the config file

        Returns:
            Initialized tokenizer object
        """
        with open(config_file, "r") as f:
            config = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = config["vocab"]
        tokenizer.merge_rules = {
            tuple(bigram): bigram_id for bigram, bigram_id in config["merge-rules"]
        }
        return tokenizer

    def save(self, path: str) -> None:
        """Dumps tokenizer configuration

        Args:
            path: Path to write config
        """
        config = {
            "vocab": self.vocab,
            "merge-rules": [
                [bigram, bigram_id] for bigram, bigram_id in self.merge_rules.items()
            ],
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_data(cls, train_data: str, n_merges: int):
        """Train an ASCIIBPETokenizer from data.

        Args:
            train_data (str): A text corpus in ASCII format.
            n_merges (int): The number of merges to perform.

        Returns:
            Trained BPE tokenizer.
        """
        tokenizer = cls()
        assert all(ord(c) < 128 for c in train_data), "train_data is not ASCII"

        token_ids = string_to_ascii(train_data)
        for _ in trange(n_merges, desc="merging..", file=sys.stdout):
            token_ids = tokenizer.merge(token_ids)

        return tokenizer


class UnicodeBPETokenizer:
    def __init__(self):
        # build initial vocab using all unicode bytes (0 - 255)
        self.vocab: list[bytes] = [bytes(range(256))]

        # self.merge_rules maps a bigram (a pair of token ids) to a single token id
        self.merge_rules: list[Bigram, int] = {}

        # regex pattern that groups preceding whitespace characters with regular characters
        self.compiled_pattern = re.compile(r"(\s+\S*)")

    def merge(self, spans: list[list[int]]) -> list[list[int]]:
        bigram_counter = Counter()
        for token_ids in spans:
            bigram_counter = compute_bigram_statistics(token_ids, bigram_counter)
        bigram_to_merge = min(
            bigram_counter, key=lambda tup: (-bigram_counter[tup], tup)
        )
        token_left, token_right = bigram_to_merge
        new_token = self.vocab[token_left] + self.vocab[token_right]

        new_token_id = len(self.vocab)
        self.vocab.append(new_token)
        self.merge_rules[bigram_to_merge] = new_token_id

        return [
            replace_bigram(token_ids, bigram_to_merge, new_token_id)
            for token_ids in spans
        ]

    def encode(self, text: str) -> list[int]:
        encoded = []
        for chunk in re.findall(self.compiled_pattern, text):
            token_ids = string_to_unicode_bytes(chunk)
            bigram_stats = compute_bigram_statistics(token_ids)
            while bigram_stats.keys() & self.merge_rules.keys():
                for bigram, bigram_id in self.merge_rules.items():
                    if bigram_stats[bigram] > 0:
                        token_ids = replace_bigram(token_ids, bigram, bigram_id)
                        break
                bigram_stats = compute_bigram_statistics(token_ids)
            encoded.extend(token_ids)
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        """Convert tokens back to text.

        Args:
            token_ids: A list of token ids.

        Returns:
            str: A Unicode (UTF-8) string.
        """
        return b"".join([self.vocab[token_id] for token_id in token_ids]).decode(
            "utf-8", errors="ignore"
        )

    @classmethod
    def from_config(cls, config_file: str):
        """Load tokenizer from a JSON config file.

        Args:
            config_file: Path to the config file

        Returns:
            Initialized tokenizer object
        """
        with open(config_file, "r") as f:
            config = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = [bytes(a) for a in config["vocab"]]
        tokenizer.merge_rules = {
            tuple(bigram): bigram_id for bigram, bigram_id in config["merge-rules"]
        }
        return tokenizer

    def save(self, path: str) -> None:
        """Dumps tokenizer configuration

        Args:
            path: Path to write config
        """
        config = {
            "vocab": [list(b) for b in self.vocab],
            "merge-rules": [
                [bigram, bigram_id] for bigram, bigram_id in self.merge_rules.items()
            ],
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_data(cls, train_data: str, n_merges: int):
        """Train an UnicodeBPETokenizer from data.

        Args:
            train_data (str): A text corpus in Unicode format.
            n_merges (int): The number of merges to perform.

        Returns:
            Trained BPE tokenizer.
        """
        tokenizer = cls()

        span_token_ids: list[list[int]] = [
            string_to_unicode_bytes(span)
            for span in re.findall(tokenizer.compiled_pattern, train_data)
        ]

        for _ in trange(n_merges, desc="merging..", file=sys.stdout):
            span_token_ids = tokenizer.merge(span_token_ids)
        return tokenizer
