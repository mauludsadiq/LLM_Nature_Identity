from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .generate import GenerationConfig, generate
from .model import IdentLM
from .sculpt import IdentityPreset, apply_preset


@dataclass(frozen=True)
class CompetitionConfig:
    prompt: Tuple[str, ...]
    steps: int = 4
    temperature: float = 1.0
    topic_lock: bool = False
    allowed_tokens: Optional[Set[str]] = None


def apply_competing_presets(
    model: IdentLM,
    preset_a: IdentityPreset,
    preset_b: IdentityPreset,
    weight_a: float,
    weight_b: float,
) -> None:
    # Apply both chains as additive fields (superposition)
    model.sculpt_chain(list(preset_a.prompt), list(preset_a.continuation), weight_a)
    model.sculpt_chain(list(preset_b.prompt), list(preset_b.continuation), weight_b)


def _winner(output: Sequence[str], preset_a: IdentityPreset, preset_b: IdentityPreset) -> str:
    # winner = whose full_expected is a prefix of the output (strongest criterion)
    out = list(output)
    a = list(preset_a.full_expected)
    b = list(preset_b.full_expected)

    if out[: len(a)] == a:
        return preset_a.name
    if out[: len(b)] == b:
        return preset_b.name

    # fallback: check continuation token presence
    a_hit = sum(tok in out for tok in preset_a.continuation)
    b_hit = sum(tok in out for tok in preset_b.continuation)
    if a_hit > b_hit:
        return preset_a.name
    if b_hit > a_hit:
        return preset_b.name
    return "tie"


def run_competition(
    *,
    preset_a: IdentityPreset,
    preset_b: IdentityPreset,
    weight_a: float,
    weight_b: float,
    seeds: Sequence[int],
    cfg: CompetitionConfig,
    model_seed: int = 0,
) -> Dict:
    if tuple(preset_a.prompt) != tuple(preset_b.prompt):
        raise ValueError("competition requires same prompt for both presets")

    # one shared model: superposed basins
    vocab = __import__("llm_identity.vocab", fromlist=["make_default_vocab"]).make_default_vocab()
    model = IdentLM(vocab=vocab, seed=model_seed)
    apply_competing_presets(model, preset_a, preset_b, weight_a, weight_b)

    counts = {preset_a.name: 0, preset_b.name: 0, "tie": 0}
    runs: List[Dict] = []

    for s in seeds:
        gcfg = GenerationConfig(
            steps=cfg.steps,
            temperature=cfg.temperature,
            seed=s,
            topic_lock=cfg.topic_lock,
            allowed_tokens=cfg.allowed_tokens,
        )
        out = generate(model, list(cfg.prompt), gcfg)
        w = _winner(out, preset_a, preset_b)
        counts[w] += 1
        runs.append({"seed": s, "output": out, "winner": w})

    return {
        "preset_a": preset_a.name,
        "preset_b": preset_b.name,
        "weight_a": float(weight_a),
        "weight_b": float(weight_b),
        "counts": counts,
        "runs": runs,
    }
