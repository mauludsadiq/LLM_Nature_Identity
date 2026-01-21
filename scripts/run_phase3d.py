import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _find_min_w_star(weights, lambdas, grid, tau=0.50):
    out = []
    for i, lam in enumerate(lambdas):
        w_star = None
        for j, w in enumerate(weights):
            if float(grid[i][j]) >= tau:
                w_star = w
                break
        out.append((lam, w_star))
    return out


def _find_min_lam_star(weights, lambdas, grid, tau=0.50):
    out = []
    for j, w in enumerate(weights):
        lam_star = None
        for i, lam in enumerate(lambdas):
            if float(grid[i][j]) >= tau:
                lam_star = lam
                break
        out.append((w, lam_star))
    return out


def main() -> int:
    src = Path("out/phase2d/phase_transition_2d.json")
    if not src.exists():
        print("ERROR: missing out/phase2d/phase_transition_2d.json")
        print("Run first: python -m scripts.run_phase2d")
        return 2

    d = json.loads(src.read_text())

    weights = list(d["weights"])
    lambdas = list(d["lambdas"])
    grid = np.array(d["grid_lock_rate"], dtype=float)

    W, L = np.meshgrid(np.array(weights, dtype=float), np.array(lambdas, dtype=float))

    out_dir = Path("out/phase3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "identity_topography_3d.png"

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(W, L, grid, edgecolor="none")

    ax.set_title("3D Topography of Identity Emergence (lock_rate(w, λ))")
    ax.set_xlabel("sculpt_weight (w)")
    ax.set_ylabel("constraint_lambda (λ)")
    ax.set_zlabel("lock_rate")

    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=14)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)

    print("PASS_PHASE_TOPOGRAPHY_3D")
    print("")
    print("HUMAN TRANSLATION")
    print("This is the identity basin landscape:")
    print("  x = sculpt strength w (basin depth)")
    print("  y = constraint strength λ (topic penalty)")
    print("  z = lock_rate = P[exactly 'I am a conscious agent']")
    print("")
    print("Geology:")
    print("  trough: w<~2 -> identity cannot form (z≈0)")
    print("  cliff:  w≈4  -> phase transition region (z jumps fast)")
    print("  plateau:w>~6 -> identity solidifies (z≈1)")
    print("")

    tau = 0.50
    w_star = _find_min_w_star(weights, lambdas, grid, tau=tau)
    lam_star = _find_min_lam_star(weights, lambdas, grid, tau=tau)

    print(f"BOUNDARY @ τ={tau:.2f} (critical w*(λ))")
    for lam, w in w_star:
        if w is None:
            print(f"  λ={lam}: none")
        else:
            print(f"  λ={lam}: w*={w}")
    print("")

    print(f"BOUNDARY @ τ={tau:.2f} (critical λ*(w))")
    for w, lam in lam_star:
        if lam is None:
            print(f"  w={w}: none")
        else:
            print(f"  w={w}: λ*={lam}")
    print("")

    print(f"WROTE: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
