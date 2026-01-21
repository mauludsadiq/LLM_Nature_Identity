from __future__ import annotations

from llm_identity.sculpt import preset_identity_conscious_agent
from llm_identity.witness import run_identity_suite


def test_suite_is_deterministic_under_seed(tmp_path):
    preset = preset_identity_conscious_agent()
    w1 = run_identity_suite(out_dir=str(tmp_path / "r1"), preset=preset, seed=123)
    w2 = run_identity_suite(out_dir=str(tmp_path / "r2"), preset=preset, seed=123)

    assert w1["digest_sha256"] == w2["digest_sha256"]
    assert w1["runs"] == w2["runs"]
