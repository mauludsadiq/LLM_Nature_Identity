from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt

from .generate import GenerationConfig, generate
from .model import IdentLM
from .sculpt import IdentityPreset


@dataclass(frozen=True)
class Phase2DConfig:
    weights: Tuple[float, ...]
    lambdas: Tuple[float, ...]
    seeds: Tuple[int, ...]
    steps: int = 3
    temperature: float = 1.0
    topic_lock: bool = True


def lock_rate_for_pair(
    preset: IdentityPreset,
    *,
    weight: float,
    constraint_lambda: float,
    seeds: Sequence[int],
    steps: int,
    temperature: float,
    topic_lock: bool,
) -> Dict:
    vocab = __import__("llm_identity.vocab", fromlist=["make_default_vocab"]).make_default_vocab()
    model = IdentLM(vocab=vocab, seed=0)

    model.sculpt_chain(list(preset.prompt), list(preset.continuation), float(weight))

    allowed = set(preset.topic_tokens)
    expected = list(preset.full_expected)

    hits = 0
    for s in seeds:
        cfg = GenerationConfig(
            steps=steps,
            temperature=temperature,
            seed=int(s),
            topic_lock=topic_lock,
            allowed_tokens=allowed if topic_lock else None,
            constraint_lambda=float(constraint_lambda) if topic_lock else None,
        )
        out = generate(model, list(preset.prompt), cfg)
        ok = (out == expected)
        hits += int(ok)

    total = len(seeds)
    return {
        "weight": float(weight),
        "constraint_lambda": float(constraint_lambda),
        "locked": hits,
        "total": total,
        "lock_rate": hits / max(1, total),
    }


def run_phase_transition_2d(
    preset: IdentityPreset,
    cfg: Phase2DConfig,
    out_dir: str,
) -> Dict:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    grid: List[List[float]] = []
    points: List[Dict] = []

    for lam in cfg.lambdas:
        row = []
        for w in cfg.weights:
            p = lock_rate_for_pair(
                preset,
                weight=w,
                constraint_lambda=lam,
                seeds=cfg.seeds,
                steps=cfg.steps,
                temperature=cfg.temperature,
                topic_lock=cfg.topic_lock,
            )
            points.append(p)
            row.append(float(p["lock_rate"]))
        grid.append(row)

    plt.figure()
    plt.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[
            float(cfg.weights[0]),
            float(cfg.weights[-1]),
            float(cfg.lambdas[0]),
            float(cfg.lambdas[-1]),
        ],
    )
    plt.xlabel("sculpt_weight (w)")
    plt.ylabel("constraint_lambda (λ)")
    plt.title(f"2D Phase: lock_rate(w, λ) ({preset.name})")
    plt.colorbar(label="lock_rate")
    fig_path = outp / "phase_transition_2d.png"
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()

    report = {
        "preset": preset.name,
        "prompt": list(preset.prompt),
        "continuation": list(preset.continuation),
        "topic_lock": cfg.topic_lock,
        "steps": cfg.steps,
        "temperature": cfg.temperature,
        "weights": list(cfg.weights),
        "lambdas": list(cfg.lambdas),
        "seeds": list(cfg.seeds),
        "grid_lock_rate": grid,
        "points": points,
        "plot_png": str(fig_path),
        "soft_projector_equation": "l_prime(v)=l(v)-lambda*1[v notin A]",
    }

    json_path = outp / "phase_transition_2d.json"
    json_path.write_text(json.dumps(report, indent=2))
    return report
