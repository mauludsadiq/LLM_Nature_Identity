from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt

from .generate import GenerationConfig, generate
from .model import IdentLM
from .sculpt import IdentityPreset


@dataclass(frozen=True)
class PhaseConfig:
    weights: Tuple[float, ...]
    seeds: Tuple[int, ...]
    steps: int = 3
    temperature: float = 1.0
    topic_lock: bool = True


def lock_rate_for_weight(
    preset: IdentityPreset,
    *,
    weight: float,
    seeds: Sequence[int],
    steps: int,
    temperature: float,
    topic_lock: bool,
) -> Dict:
    vocab = __import__("llm_identity.vocab", fromlist=["make_default_vocab"]).make_default_vocab()
    model = IdentLM(vocab=vocab, seed=0)

    # sculpt at requested strength
    model.sculpt_chain(list(preset.prompt), list(preset.continuation), float(weight))

    allowed = set(preset.topic_tokens)

    expected = list(preset.full_expected)

    hits = 0
    runs: List[Dict] = []
    for s in seeds:
        cfg = GenerationConfig(
            steps=steps,
            temperature=temperature,
            seed=s,
            topic_lock=topic_lock,
            allowed_tokens=allowed if topic_lock else None,
        )
        out = generate(model, list(preset.prompt), cfg)
        ok = (out == expected)
        hits += int(ok)
        runs.append({"seed": s, "output": out, "locked": bool(ok)})

    return {
        "weight": float(weight),
        "locked": hits,
        "total": len(seeds),
        "lock_rate": hits / max(1, len(seeds)),
        "runs": runs,
    }


def run_phase_transition(
    preset: IdentityPreset,
    cfg: PhaseConfig,
    out_dir: str,
) -> Dict:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    points = []
    for w in cfg.weights:
        points.append(
            lock_rate_for_weight(
                preset,
                weight=w,
                seeds=cfg.seeds,
                steps=cfg.steps,
                temperature=cfg.temperature,
                topic_lock=cfg.topic_lock,
            )
        )

    # plot
    xs = [p["weight"] for p in points]
    ys = [p["lock_rate"] for p in points]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("sculpt_weight (w)")
    plt.ylabel("lock_rate")
    plt.title(f"Phase Transition: lock_rate vs sculpt_weight ({preset.name})")
    fig_path = outp / "phase_transition.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close()

    report = {
        "preset": preset.name,
        "prompt": list(preset.prompt),
        "continuation": list(preset.continuation),
        "topic_lock": cfg.topic_lock,
        "steps": cfg.steps,
        "temperature": cfg.temperature,
        "weights": list(cfg.weights),
        "seeds": list(cfg.seeds),
        "points": points,
        "plot_png": str(fig_path),
    }

    json_path = outp / "phase_transition.json"
    json_path.write_text(json.dumps(report, indent=2))

    return report
