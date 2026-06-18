# Heavily inspired from Andrej Karpathy's miniBPE repo:
# This is my own implementation with changes — integration into a shared base class,
# explicit special token handling, and save/load hooks.

# Andrej Karpathy's original repo: https://github.com/karpathy/minbpe
# Andrej Karpathy's original video explanation: https://youtu.be/zduSFxRajkE?si=sNlEdHyINCK6wrof

# BPE Tokenizer paper: Neural Machine Translation of Rare Words with Subword Units, Sennrich et al. (2016)
# https://arxiv.org/abs/1508.07909

# GPT-2 paper, where they describe using this type of tokenizer
# https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

import regex as re

from typing import Dict, List, Optional, Tuple, cast
from .base_tokenizer import BaseTokenizer, get_stats, merge


# GPT-4's regex pre-tokenization pattern. Splits text into chunks before
# byte-encoding so merges never cross word/punctuation/whitespace boundaries.
REGEX_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

GPT4_SPECIAL_TOKENS: Dict[str, int] = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


class GPTTokenizer(BaseTokenizer):
    """BPE tokenizer using GPT-4's regex pre-tokenization pattern."""

    def __init__(self, special_tokens: Optional[Dict[str, int]] = None):
        super().__init__(pattern=REGEX_PATTERN, unk_token="<|endoftext|>")
        # Override compiled_pattern: base uses stdlib re, but this pattern
        # requires the regex library for Unicode property support (\p{L}, \p{N}).
        self.pattern = cast(str, self.pattern)  # mypy doesn't know this is set by the base constructor
        self.compiled_pattern = re.compile(self.pattern)
        self.register_special_tokens(special_tokens)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 to cover all byte values.")

        num_merges = vocab_size - 256
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges: Dict[Tuple[int, int], int] = {}
        vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats: Dict[Tuple[int, int], int] = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)  # type: ignore[arg-type]
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """BPE-encode a single pre-tokenized chunk (no special token handling)."""
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Apply the merge with the lowest assigned index (earliest learned merge).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> List[int]:
        """Encode text, ignoring any special tokens."""
        ids: List[int] = []
        for chunk in re.findall(self.compiled_pattern, text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str) -> List[int]:
        """
        Encode text, handling special tokens as indivisible units.
        Falls through to encode_ordinary if no special tokens are registered.
        """
        if not self.special_tokens:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        ids: List[int] = []
        for part in re.split(special_pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        part_bytes: List[bytes] = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Unknown token id: {idx}")
        return b"".join(part_bytes).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Save / Load hooks
    # ------------------------------------------------------------------

    def _save_model_rules(self, f) -> None:
        for idx1, idx2 in self.merges:
            f.write(f"{idx1} {idx2}\n")

    def _load_model_rules(self, f) -> None:
        idx = 256
        for line in f:
            idx1, idx2 = map(int, line.split())
            self.merges[(idx1, idx2)] = idx
            idx += 1

    def _post_load_setup(self) -> None:
        # Re-compile with the regex library; base class used stdlib re which
        # doesn't support Unicode property escapes (\p{L}, \p{N}).
        if self.pattern:
            self.compiled_pattern = re.compile(self.pattern)
