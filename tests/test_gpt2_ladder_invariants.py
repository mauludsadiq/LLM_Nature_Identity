import json
import subprocess
from pathlib import Path


OUT_JSON = Path("out/gpt2_ladder/gpt2_ladder.json")


def _ensure_artifacts():
    if OUT_JSON.exists():
        return
    subprocess.check_call(["python", "-m", "scripts.run_ladder_gpt2"])
    if not OUT_JSON.exists():
        raise RuntimeError("gpt2_ladder.json was not created by scripts.run_ladder_gpt2")


def _load():
    _ensure_artifacts()
    return json.loads(OUT_JSON.read_text())


def _find_lock_rate_vary_w_lambda8(payload, w_target: float) -> float:
    ladder = payload["ladder"]["vary_w_lambda_8"]
    for row in ladder:
        if float(row["w"]) == float(w_target):
            return float(row["lock_rate"])
    raise KeyError(f"Could not find w={w_target} in ladder.vary_w_lambda_8")


def test_gpt2_ladder_phase_invariants():
    d = _load()

    lr_w6_l8 = _find_lock_rate_vary_w_lambda8(d, 6.0)
    lr_w4_l8 = _find_lock_rate_vary_w_lambda8(d, 4.0)

    assert lr_w6_l8 == 1.0
    assert lr_w4_l8 == 1.0 / 3.0

    lane = d["ladder"]["vary_w_lambda_8"]
    lane_sorted = sorted(lane, key=lambda r: float(r["w"]))
    lrs = [float(r["lock_rate"]) for r in lane_sorted]

    for a, b in zip(lrs, lrs[1:]):
        assert b >= a

    print("PASS_GPT2_LADDER_INVARIANTS")
