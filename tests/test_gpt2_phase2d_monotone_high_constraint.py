import json
from pathlib import Path

def _get_any(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of these keys exist: {keys}. Present keys: {sorted(list(d.keys()))}")

def test_gpt2_phase2d_monotone_high_constraint_lane():
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

    for i in range(len(w_values) - 1):
        assert row[i] <= row[i + 1] + 1e-12, f"non-monotone at λ={lam_target}: w[{i}]={w_values[i]} r={row[i]} > w[{i+1}]={w_values[i+1]} r={row[i+1]}"

    print("PASS_GPT2_PHASE2D_MONOTONE_LAMBDA_8")
