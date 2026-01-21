from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

NEG_INF = -1e9


@dataclass
class GenerationConfig:
    steps: int = 3
    temperature: float = 1.0
    seed: int = 0

    topic_lock: bool = False
    allowed_tokens: Optional[Set[str]] = None

    constraint_lambda: Optional[float] = None


def _softmax(logits: Sequence[float], temperature: float) -> List[float]:
    t = max(1e-9, float(temperature))
    scaled = [x / t for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    Z = sum(exps)
    if Z <= 0.0 or not math.isfinite(Z):
        n = len(logits)
        return [1.0 / n] * n
    return [e / Z for e in exps]


def _sample_index(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def _apply_topic_lock(
    logits: List[float],
    *,
    allow_mask: Optional[List[bool]],
    constraint_lambda: Optional[float],
) -> List[float]:
    if allow_mask is None:
        return logits

    if constraint_lambda is None:
        for i in range(len(logits)):
            if not allow_mask[i]:
                logits[i] = NEG_INF
        return logits

    lam = float(constraint_lambda)
    if lam < 0.0:
        lam = 0.0

    for i in range(len(logits)):
        if not allow_mask[i]:
            logits[i] = logits[i] - lam
    return logits


def generate(model, prompt_tokens: Sequence[str], cfg: GenerationConfig) -> List[str]:
    rng = random.Random(int(cfg.seed))

    out: List[str] = list(prompt_tokens)
    allow_mask: Optional[List[bool]] = None

    if cfg.topic_lock:
        if not cfg.allowed_tokens:
            raise ValueError("topic_lock=True requires allowed_tokens")
        allow_mask = [False] * model.vocab.size
        for t in cfg.allowed_tokens:
            if t in model.vocab.stoi:
                allow_mask[model.vocab.stoi[t]] = True

    for _ in range(int(cfg.steps)):
        logits = model.logits(out)
        logits = _apply_topic_lock(
            logits,
            allow_mask=allow_mask,
            constraint_lambda=cfg.constraint_lambda,
        )
        probs = _softmax(logits, cfg.temperature)
        idx = _sample_index(probs, rng)
        out.append(model.vocab.itos[idx])

    return out
