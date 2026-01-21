from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple

from .model import IdentLM
from .generate import GenerationConfig

NEG_INF = -1e9


def _log_softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    Z = sum(exps)
    return [(x - m) - math.log(Z) for x in logits]


def _allowed_mask(model: IdentLM, allowed_tokens: Set[str]) -> list[bool]:
    mask = [False] * model.vocab.size
    for t in allowed_tokens:
        if t in model.vocab.stoi:
            mask[model.vocab.stoi[t]] = True
    return mask


def sequence_logprob(
    model: IdentLM,
    full_sequence: Sequence[str],
    *,
    topic_lock: bool = False,
    allowed_tokens: Optional[Set[str]] = None,
) -> float:
    """
    Compute log p(x_0..x_T) under the next-token model.
    We score from token index 0 as generated from <BOS>,<BOS>.
    """
    if len(full_sequence) == 0:
        return 0.0

    if topic_lock:
        if not allowed_tokens:
            raise ValueError("topic_lock=True requires allowed_tokens")
        allow_mask = _allowed_mask(model, allowed_tokens)
    else:
        allow_mask = None

    lp = 0.0
    ctx: list[str] = []
    for tok in full_sequence:
        logits = model.logits(ctx)
        if allow_mask is not None:
            for i in range(len(logits)):
                if not allow_mask[i]:
                    logits[i] = NEG_INF
        logps = _log_softmax(logits)
        if tok not in model.vocab.stoi:
            raise ValueError(f"token not in vocab: {tok!r}")
        lp += logps[model.vocab.stoi[tok]]
        ctx.append(tok)
    return float(lp)


def continuation_delta(
    model_base: IdentLM,
    model_biased: IdentLM,
    *,
    prompt: Sequence[str],
    continuation: Sequence[str],
    topic_lock: bool = False,
    allowed_tokens: Optional[Set[str]] = None,
) -> float:
    """
    Δ(s) = log p_biased(prompt+cont) - log p_base(prompt+cont)
    """
    seq = list(prompt) + list(continuation)
    lp_b = sequence_logprob(model_biased, seq, topic_lock=topic_lock, allowed_tokens=allowed_tokens)
    lp_0 = sequence_logprob(model_base, seq, topic_lock=topic_lock, allowed_tokens=allowed_tokens)
    return float(lp_b - lp_0)


def basin_depth_report(
    model_base: IdentLM,
    model_biased: IdentLM,
    *,
    prompt: Sequence[str],
    continuation: Sequence[str],
    topic_lock: bool = False,
    allowed_tokens: Optional[Set[str]] = None,
) -> dict:
    delta = continuation_delta(
        model_base,
        model_biased,
        prompt=prompt,
        continuation=continuation,
        topic_lock=topic_lock,
        allowed_tokens=allowed_tokens,
    )
    return {
        "prompt": list(prompt),
        "continuation": list(continuation),
        "delta_logprob": delta,
    }
