import json, subprocess, os, sys

def test_identity_invariance_pass(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    phase2d = os.path.join(root, "out", "phase2d", "phase_transition_2d.json")
    out_json = os.path.join(root, "out", "invariance_profile.json")

    cmd = [
        sys.executable,
        os.path.join(root, "analysis", "identity_invariance.py"),
        "--phase2d_json", phase2d,
        "--out_json", out_json,
        "--thr", "0.5",
        "--lambda_pick", "8.0",
        "--variants", "punct_noise,case_noise,swap_adjacent,drop_token",
        "--canon", "lower",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    d = json.load(open(out_json, "r", encoding="utf-8"))
    assert "digest_sha256" in d
    assert "invariance" in d
    assert "overall_invariance_rate" in d["invariance"]

    inv = float(d["invariance"]["overall_invariance_rate"])
    assert 0.0 <= inv <= 1.0

    assert "slice" in d
    assert "w_boundary_first_at_or_above_thr" in d["slice"]

    print("PASS_IDENTITY_INVARIANCE_PROFILE")
