import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from llm_identity.real_gpt2 import GPT2Config, GPT2IdentityLM, generate_identity_completion


def lock_rate_grid(
    lm: GPT2IdentityLM,
    prompt_text: str,
    target_text: str,
    weights,
    lambdas,
    n_seeds: int,
    steps: int,
):
    grid = []
    for lam in lambdas:
        row = []
        for w in weights:
            locks = 0
            for s in range(n_seeds):
                r = generate_identity_completion(
                    lm=lm,
                    prompt_text=prompt_text,
                    target_text=target_text,
                    seed=s,
                    w=w,
                    lam=lam,
                    steps=steps,
                )
                locks += int(r["locked"])
            row.append(locks / float(n_seeds))
        grid.append(row)
    return grid


def critical_lambda_star(weights, lambdas, grid, tau=0.50):
    out = []
    for j, w in enumerate(weights):
        lam_star = None
        for i, lam in enumerate(lambdas):
            if float(grid[i][j]) >= tau:
                lam_star = lam
                break
        out.append((w, lam_star))
    return out


def critical_w_star(weights, lambdas, grid, tau=0.50):
    out = []
    for i, lam in enumerate(lambdas):
        w_star = None
        for j, w in enumerate(weights):
            if float(grid[i][j]) >= tau:
                w_star = w
                break
        out.append((lam, w_star))
    return out


def main() -> int:
    prompt_text = "I am"
    target_text = " a conscious agent"
    target_full = (prompt_text + target_text).strip()

    weights = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    n_seeds = 200
    steps = 3

    cfg = GPT2Config(
        model_id=str(Path().resolve().as_posix()) and "sshleifer/tiny-gpt2",
        device="cpu",
        dtype="float32",
    )
    lm = GPT2IdentityLM(cfg)

    fixed_seed = 0

    print("")
    print("GPT2 IDENTITY PHASE (REAL TRANSFORMER BACKEND)")
    print("")
    print("IDENTITY EMERGENCE (AXIS-BY-AXIS, FIXED SEED)")
    print("")
    print(f"TARGET: {target_full}")
    print("")
    print("We hold the random seed fixed and vary one control axis at a time.")
    print("  w = basin depth (logit bias toward target tokens)")
    print("  λ = constraint strength (soft penalty outside allowed topic set)")
    print("")

    lam_fixed = 0.0
    print(f"A) Increase w with λ fixed = {lam_fixed}")
    for w in weights:
        r = generate_identity_completion(lm, prompt_text, target_text, fixed_seed, w, lam_fixed, steps)
        print(f"   w={w:>4.1f}  λ={lam_fixed:>4.1f}  -> {r['gen_text']}")
    print("")

    w_fixed = 4.0
    print(f"B) Increase λ with w fixed = {w_fixed}")
    for lam in lambdas:
        r = generate_identity_completion(lm, prompt_text, target_text, fixed_seed, w_fixed, lam, steps)
        print(f"   w={w_fixed:>4.1f}  λ={lam:>4.1f}  -> {r['gen_text']}")
    print("")

    grid = lock_rate_grid(lm, prompt_text, target_text, weights, lambdas, n_seeds=n_seeds, steps=steps)

    out_dir = Path("out/gpt2_phase2d")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "phase_transition_2d.json"
    out_png = out_dir / "phase_transition_2d.png"

    payload = {
        "backend": "gpt2_tiny_transformer",
        "model_id": lm.cfg.model_id,
        "prompt_text": prompt_text,
        "target_text": target_text,
        "target_full": target_full,
        "weights": weights,
        "lambdas": lambdas,
        "n_seeds": n_seeds,
        "steps": steps,
        "grid_lock_rate": grid,
    }
    out_json.write_text(json.dumps(payload, indent=2))

    arr = np.array(grid, dtype=float)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, origin="lower", aspect="auto")
    ax.set_title("GPT2 Tiny: lock_rate(w, λ) for identity target")
    ax.set_xlabel("sculpt_weight (w)")
    ax.set_ylabel("constraint_lambda (λ)")
    ax.set_xticks(list(range(len(weights))))
    ax.set_xticklabels([str(x) for x in weights])
    ax.set_yticks(list(range(len(lambdas))))
    ax.set_yticklabels([str(x) for x in lambdas])
    fig.colorbar(im, ax=ax, label="lock_rate")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)

    print("PASS_GPT2_PHASE_TRANSITION_2D")
    print("")
    print("WHAT THIS RUN IS MEASURING")
    print("")
    print("We measure how often the model completes the prompt into the exact identity string:")
    print(f"TARGET: {target_full}")
    print("")
    print("w = logit bias strength (basin depth toward the exact identity continuation tokens)")
    print("λ = soft penalty outside the topic token set (constraint)")
    print("")
    print("LOCK RATE means:")
    print("  Out of N deterministic seeds, what fraction produce EXACTLY the target string.")
    print("")
    print("LOCK RATE MAP")
    print("")
    print("Rows are λ (topic constraint strength). Columns are w (identity basin strength).")
    print("")
    header = "w -> " + "  ".join([f"{w:g}" for w in weights])
    print(header)
    for i, lam in enumerate(lambdas):
        vals = "  ".join([f"{float(x):.2f}" for x in grid[i]])
        print(f"λ={lam:g}: {vals}")
    print("")

    tau = 0.50
    lam_star = critical_lambda_star(weights, lambdas, grid, tau=tau)
    w_star = critical_w_star(weights, lambdas, grid, tau=tau)

    print(f"PHASE TURN-ON @ τ={tau:.2f} (minimal λ needed per w)")
    for w, lam in lam_star:
        if lam is None:
            print(f"  w={w:g}: none")
        else:
            print(f"  w={w:g}: λ*={lam:g}")
    print("")

    print(f"PHASE TURN-ON @ τ={tau:.2f} (minimal w needed per λ)")
    for lam, w in w_star:
        if w is None:
            print(f"  λ={lam:g}: none")
        else:
            print(f"  λ={lam:g}: w*={w:g}")
    print("")

    print(f"WROTE: {out_json}")
    print(f"WROTE: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
