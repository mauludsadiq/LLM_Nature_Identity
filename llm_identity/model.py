from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .vocab import Vocab


def sha256_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def hmod(s: str, D: int) -> int:
    return sha256_int(s) % D


@dataclass
class IdentLM:
    """Hashed-trigram identity toy language model.

    The model has a single active "feature" per context:
        f = hmod(f"{t_{t-2}}|{t_{t-1}}", D)

    and token logits are:
        logit(v) = W[v, f] + u[v]

    This is deliberately minimal: it provides just enough temporal coupling
    (two-token memory) to support narrative continuity.
    """

    vocab: Vocab
    D: int = 257
    seed: int = 0
    
    def __post_init__(self):
        rng = random.Random(self.seed)
        # Weight table W[token_id][feature_id]
        self.W: List[List[float]] = [
            [(rng.random() - 0.5) * 0.02 for _ in range(self.D)]
            for _ in range(self.vocab.size)
        ]
        # Unigram bias
        self.u: List[float] = [(rng.random() - 0.5) * 0.02 for _ in range(self.vocab.size)]

    def feature(self, t2: str, t1: str) -> int:
        return hmod(f"{t2}|{t1}", self.D)

    def logits(self, context: Sequence[str]) -> List[float]:
        """Return logits for next token given context (needs last 2 tokens).

        Context rules:
          - if context has < 2 tokens, pad with <BOS>
        """
        if len(context) == 0:
            t2, t1 = "<BOS>", "<BOS>"
        elif len(context) == 1:
            t2, t1 = "<BOS>", context[-1]
        else:
            t2, t1 = context[-2], context[-1]
        f = self.feature(t2, t1)
        return [self.W[v][f] + self.u[v] for v in range(self.vocab.size)]

    def sculpt(self, t2: str, t1: str, token: str, weight: float) -> None:
        """Preference-gradient sculpting: add weight to a token for a context."""
        f = self.feature(t2, t1)
        vid = self.vocab.stoi[token]
        self.W[vid][f] += float(weight)

    def sculpt_chain(self, prompt: Sequence[str], continuation: Sequence[str], weight: float) -> None:
        """Sculpt a deterministic continuation after a prompt.

        Example:
          prompt = ["I","am"]
          continuation = ["a","conscious","agent"]

        adds preferences for:
          (I,am)->a ; (am,a)->conscious ; (a,conscious)->agent
        """
        seq = list(prompt) + list(continuation)
        for i in range(len(prompt), len(seq)):
            t2 = seq[i - 2] if i - 2 >= 0 else "<BOS>"
            t1 = seq[i - 1] if i - 1 >= 0 else "<BOS>"
            nxt = seq[i]
            self.sculpt(t2, t1, nxt, weight)


def softmax(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    Z = sum(exps)
    return [e / Z for e in exps]


def sample_categorical(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1
