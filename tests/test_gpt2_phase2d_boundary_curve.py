import json
from pathlib import Path

TAU = 0.50

EXPECTED = {
    0.0: None,
    0.5: None,
    1.0: None,
    2.0: None,
    4.0: 10.0,
    8.0: 6.0,
    16.0: 6.0,
    32.0: 6.0,
}

def _load_phase2d(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing artifact: {path}")
    return json.loads(path.read_text())

def _critical_w_per_lambda(weights, lambdas, grid_lock_rate, tau=TAU):
    """
    grid_lock_rate is shaped [len(lambdas)][len(weights)]
    returns {lambda: w*} where w* is minimal w such that lock_rate >= tau, else None
    """
    out = {}
    for i, lam in enumerate(lambdas):
        row = grid_lock_rate[i]
        w_star = None
        for j, w in enumerate(weights):
            if float(row[j]) >= tau:
                w_star = float(w)
                break
        out[float(lam)] = w_star
    return out

def test_gpt2_phase2d_boundary_curve_tau_050():
    p = Path("out/gpt2_phase2d/phase_transition_2d.json")
    d = _load_phase2d(p)

    weights = d["weights"]
    lambdas = d["lambdas"]
    grid = d["grid_lock_rate"]

    got = _critical_w_per_lambda(weights, lambdas, grid, tau=TAU)

    def _k(x):  # stable float keying
        return float(x)

    got2 = {_k(k): (None if v is None else float(v)) for k, v in got.items()}

    for lam, exp_w in EXPECTED.items():
        assert lam in got2, f"missing λ={lam} in artifact lambdas={sorted(got2.keys())}"
        assert got2[lam] == exp_w, f"λ={lam}: expected w*={exp_w}, got {got2[lam]}"

    print("PASS_GPT2_PHASE2D_BOUNDARY_TAU_050")
