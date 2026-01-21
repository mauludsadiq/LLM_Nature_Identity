import json
from pathlib import Path

def _get_any(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of these keys exist: {keys}. Present keys: {sorted(list(d.keys()))}")

def _first_w_at_or_above(w_values, row, thresh):
    for w, r in zip(w_values, row):
        if r >= thresh:
            return float(w)
    return None

def test_gpt2_phase2d_phase_sharpness_lambda_8():
    p = Path("out/gpt2_phase2d/phase_transition_2d.json")
    assert p.exists(), "Missing artifact. Run: python -m scripts.run_phase2d_gpt2"

    d = json.loads(p.read_text())

    w_values = _get_any(d, ["w_values", "weights", "ws", "w_grid", "w"])
    lam_values = _get_any(d, ["lambda_values", "lambdas", "lambda_grid", "lam", "λ"])
    grid = _get_any(d, ["grid_lock_rate", "lock_rate_grid", "lock_rate", "grid"])

    lam_target = 8.0
    assert lam_target in lam_values, f"λ={lam_target} missing from artifact grid. Have: {lam_values}"

    j = lam_values.index(lam_target)
    row = grid[j]

    w10 = _first_w_at_or_above(w_values, row, 0.10)
    w90 = _first_w_at_or_above(w_values, row, 0.90)

    assert w10 is not None, "No w achieved lock_rate >= 0.10"
    assert w90 is not None, "No w achieved lock_rate >= 0.90"

    band = w90 - w10
    assert band <= 4.0, f"Transition too wide at λ=8: w90-w10 = {band:.2f} (w10={w10}, w90={w90})"

    print(f"PASS_GPT2_PHASE2D_SHARPNESS_LAMBDA_8 w10={w10} w90={w90} band={band:.2f}")
