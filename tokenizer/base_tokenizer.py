# this base class is inspired heavily from Andrej Karpathy's miniGPT implementation. So full credit. 
# Andrej Karpathy's original repo: https://github.com/karpathy/minbpe
# Andrej Karpathy's original video explanation: https://youtu.be/zduSFxRajkE?si=sNlEdHyINCK6wrof

# This class is a bit overkill, but that is because it is from a different project I have been working on.

import re
import unicodedata

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

class BaseTokenizer(ABC):
    """
    Shared base for BPE, WordPiece, and SentencePiece tokenizers.

    Vocab is bytes-level: Dict[int, bytes]. Special tokens live in a
    separate dict with explicit IDs so they never collide with the byte
    vocab (0-255) or learned merge tokens (256+).
    """

    def __init__(self, pattern: Optional[str] = None, unk_token: Optional[str] = None):
        self.pattern = pattern
        self.compiled_pattern = None  # subclasses compile with their own regex engine
        self.unk_token = unk_token

        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}

    def register_special_tokens(self, specials: Optional[Dict[str, int]]) -> None:
        """
        Add special tokens with explicit IDs. Pass None to skip.
        Multiple calls are additive — each call merges into the existing dict.
        """
        if specials is None:
            return
        self.special_tokens.update(specials)
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def _build_vocabulary(self) -> Dict[int, bytes]:
        """Reconstruct vocab from merges and special tokens. Called after load()."""
        vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for token, idx in self.special_tokens.items():
            vocab[idx] = token.encode("utf-8")
        return vocab

    def pre_tokenize(self, text: str) -> List[str]:
        """
        Split text into chunks using the regex pattern before byte-encoding.
        If no pattern is set (e.g. raw SentencePiece mode), returns the full
        text as a single chunk.
        """
        if not self.compiled_pattern:
            return [text]
        return re.findall(self.compiled_pattern, text)

    @staticmethod
    def replace_control_characters(s: str) -> str:
        """Escape control characters so tokens are always safe to print."""
        # unicode category "C" = control characters
        # http://www.unicode.org/reports/tr44/#GC_Values_Table
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch)
            else:
                chars.append(f"\\u{ord(ch):04x}")
        return "".join(chars)

    @staticmethod
    def render_token(t: bytes) -> str:
        """Decode bytes to a printable string, escaping any control characters."""
        s = t.decode("utf-8", errors="replace")
        return BaseTokenizer.replace_control_characters(s)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None: ...

    @abstractmethod
    def encode(self, text: str) -> List[int]: ...

    @abstractmethod
    def decode(self, ids: List[int]) -> str: ...

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, file_prefix: str) -> None:
        """
        Writes two files:
          {file_prefix}.model  — machine-readable, round-trips through load()
          {file_prefix}.vocab  — human-readable merge table (lossy, not loadable)

        Inspired by SentencePiece's model saving convention.
        """
        model_file = f"{file_prefix}.model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write(f"{self.__class__.__name__} V1\n")
            f.write(f"{self.pattern or 'None'}\n")
            f.write(f"{self.unk_token or 'None'}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token} {idx}\n")
            self._save_model_rules(f)

        vocab_file = f"{file_prefix}.vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token_bytes in self.vocab.items():
                s = self.render_token(token_bytes)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self.render_token(self.vocab[idx0])
                    s1 = self.render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file: str) -> None:
        if not model_file.endswith(".model"):
            raise ValueError("model_file must end with .model")

        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        with open(model_file, "r", encoding="utf-8") as f:
            f.readline()  # class name line — informational, skip
            pattern = f.readline().strip()
            self.pattern = None if pattern == "None" else pattern
            self.compiled_pattern = None  # _post_load_setup() compiles with the correct engine
            unk = f.readline().strip()
            self.unk_token = None if unk == "None" else unk
            num_special = int(f.readline().strip())
            specials: Dict[str, int] = {}
            for _ in range(num_special):
                parts = f.readline().strip().split()
                specials[parts[0]] = int(parts[1])
            self.register_special_tokens(specials)
            self._load_model_rules(f)

        # Hook for subclasses that need a different regex engine or other
        # post-parse setup (e.g. GPTTokenizer re-compiles with the regex lib).
        self._post_load_setup()
        self.vocab = self._build_vocabulary()

    def _save_model_rules(self, f) -> None:
        """Subclasses write their type-specific state (merge pairs, vocab lists, etc.)."""
        pass

    def _load_model_rules(self, f) -> None:
        """Subclasses read back whatever _save_model_rules wrote."""
        pass

    def _post_load_setup(self) -> None:
        """Called after the model file is parsed, before vocab is rebuilt."""
        pass


# ----------------------------------------------------------------------
# BPE utility functions — kept module-level so notebooks can import and
# call them directly for step-by-step walkthroughs.
# ----------------------------------------------------------------------

def get_stats(
    ids: List[int], counts: Optional[Dict[Tuple[int, int], int]] = None
) -> Dict[Tuple[int, int], int]:
    """Count every consecutive pair in ids. Pass counts to accumulate across chunks."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

#TODO: Perhaps implement this into the BPE tokenizer class itself as a static method
# Though we may need it for the other tokenizer classes so maybe better here at the moment

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replace every occurrence of pair in ids with the new token idx."""
    newids: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
