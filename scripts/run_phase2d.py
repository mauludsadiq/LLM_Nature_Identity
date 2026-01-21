from __future__ import annotations

from llm_identity.model import IdentLM
from llm_identity.vocab import make_default_vocab
from llm_identity.generate import GenerationConfig, generate
from llm_identity.sculpt import preset_identity_conscious_agent
from llm_identity.phase_transition_2d import Phase2DConfig, run_phase_transition_2d


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def _sample(preset, *, w: float, lam: float | None, seed: int) -> str:
    vocab = make_default_vocab()
    model = IdentLM(vocab=vocab, seed=0)

    if w > 0.0:
        model.sculpt_chain(list(preset.prompt), list(preset.continuation), float(w))

    cfg = GenerationConfig(
        steps=3,
        temperature=1.0,
        seed=int(seed),
        topic_lock=(lam is not None),
        allowed_tokens=set(preset.topic_tokens) if lam is not None else None,
        constraint_lambda=float(lam) if lam is not None else None,
    )

    out = generate(model, list(preset.prompt), cfg)
    return " ".join(out)


def _print_axis_ladder(preset, *, seed: int = 0) -> None:
    target = " ".join(preset.full_expected)

    print("")
    print("IDENTITY EMERGENCE (AXIS-BY-AXIS, FIXED SEED)")
    print("")
    print(f"TARGET: {target}")
    print("")
    print("We hold the random seed fixed and vary one control axis at a time.")
    print("  w = basin depth (preference sculpt strength)")
    print("  λ = constraint strength (soft topic lock penalty)")
    print("")

    print("A) Increase w (basin depth) with λ fixed = 0.0")
    print("   Expectation: identity becomes preferred even without constraint as w grows.")
    print("")
    for w in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        out = _sample(preset, w=w, lam=0.0, seed=seed)
        print(f"   w={w:>4}  λ=0.0  -> {out}")
    print("")

    print("B) Increase λ (constraint strength) with w fixed = 4.0")
    print("   Expectation: constraint suppresses drift and forces topic coherence toward the identity string.")
    print("")
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        out = _sample(preset, w=4.0, lam=lam, seed=seed)
        print(f"   w=4.0  λ={lam:>4} -> {out}")
    print("")


def _print_human_grid_report(report: dict, preset) -> None:
    weights = report["weights"]
    lambdas = report["lambdas"]
    grid = report["grid_lock_rate"]

    target = " ".join(preset.full_expected)

    print("PASS_PHASE_TRANSITION_2D")
    print("")
    print("WHAT THIS RUN IS MEASURING")
    print("")
    print("We measure how often the model completes the prompt into the exact identity string:")
    print(f"TARGET: {target}")
    print("")
    print("w = identity sculpt strength (basin depth)")
    print("λ = topic constraint penalty (soft projector)")
    print("")
    print("LOCK RATE means:")
    print("  Out of N deterministic seeds, what fraction produce EXACTLY the target string.")
    print("")

    print("LOCK RATE MAP")
    print("")
    print("Rows are λ (topic constraint strength). Columns are w (identity basin strength).")
    print("")
    print("w ->", "  ".join([str(w) for w in weights]))
    for i, lam in enumerate(lambdas):
        row = "  ".join(_fmt(float(x)) for x in grid[i])
        print(f"λ={lam}: {row}")
    print("")

    tau = 0.50
    print(f"PHASE TURN-ON (minimal λ needed to get ≥{tau:.2f} lock rate, per w)")
    for j, w in enumerate(weights):
        lam_star = None
        for i, lam in enumerate(lambdas):
            if float(grid[i][j]) >= tau:
                lam_star = lam
                break
        if lam_star is None:
            print(f"  w={w}: no lock at this sweep range")
        else:
            print(f"  w={w}: λ*={lam_star}")
    print("")

    print(f"PHASE TURN-ON (minimal w needed to get ≥{tau:.2f} lock rate, per λ)")
    for i, lam in enumerate(lambdas):
        w_star = None
        for j, w in enumerate(weights):
            if float(grid[i][j]) >= tau:
                w_star = w
                break
        if w_star is None:
            print(f"  λ={lam}: no lock at this sweep range")
        else:
            print(f"  λ={lam}: w*={w_star}")
    print("")


if __name__ == "__main__":
    preset = preset_identity_conscious_agent()

    cfg = Phase2DConfig(
        weights=tuple([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
        lambdas=tuple([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        seeds=tuple(range(200)),
        steps=3,
        temperature=1.0,
        topic_lock=True,
    )

    report = run_phase_transition_2d(preset, cfg, out_dir="out/phase2d")

    _print_axis_ladder(preset, seed=0)
    _print_human_grid_report(report, preset)

    print("WROTE: out/phase2d/phase_transition_2d.json")
    print("WROTE: out/phase2d/phase_transition_2d.png")
