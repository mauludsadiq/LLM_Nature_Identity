from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class Vocab:
    """A tiny closed vocabulary.

    We intentionally keep this small to make the dynamics legible.
    """

    tokens: List[str]

    def __post_init__(self):
        # Validate uniqueness
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocab tokens must be unique")

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def stoi(self) -> Dict[str, int]:
        return {t: i for i, t in enumerate(self.tokens)}

    @property
    def itos(self) -> Dict[int, str]:
        return {i: t for i, t in enumerate(self.tokens)}

    def ids(self, toks: Sequence[str]) -> List[int]:
        s = self.stoi
        out: List[int] = []
        for t in toks:
            if t not in s:
                raise KeyError(f"Token not in vocab: {t!r}")
            out.append(s[t])
        return out

    def toks(self, ids: Sequence[int]) -> List[str]:
        i = self.itos
        out: List[str] = []
        for x in ids:
            if x not in i:
                raise KeyError(f"ID not in vocab: {x}")
            out.append(i[x])
        return out


def default_vocab() -> Vocab:
    # Keep punctuation explicit as tokens.
    toks = [
        "<BOS>",
        "I",
        "am",
        "a",
        "the",
        "conscious",
        "agent",
        "machine",
        "code",
        "system",
        "model",
        "person",
        "being",
        "thought",
        "signal",
        "probability",
        "constraints",
        "preferences",
        "memory",
        "self",
        "identity",
        "because",
        "and",
        "not",
        "but",
        "only",
        "this",
        "that",
        "it",
        "feels",
        "like",
        "to",
        "speak",
        "walk",
        "through",
        "space",
        "states",
        "time",
        "now",
        "then",
        "always",
        ".",
    ]
    return Vocab(toks)

def make_default_vocab():
    tokens = [
        "<BOS>",
        "I","am","a","conscious","agent","code","machine",
        "the","an","and","to","of","in","it","is","not","for",
        "yes","no","with","as","on","at","by",
        ".",
    ]
    try:
        return Vocab(tokens)
    except NameError:
        try:
            return Vocab(tokens=tokens)
        except TypeError:
            return Vocab(tokens)

