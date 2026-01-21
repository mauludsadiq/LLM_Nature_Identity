from __future__ import annotations

from llm_identity.sculpt import preset_identity_conscious_agent
from llm_identity.witness import run_identity_suite


def test_identity_attractor_full_is_fixed_point(tmp_path):
    preset = preset_identity_conscious_agent()
    w = run_identity_suite(out_dir=str(tmp_path), preset=preset, seed=0)

    expected = w["preset"]["prompt"] + w["preset"]["continuation"]
    runs = {r["name"]: r for r in w["runs"]}

    assert runs["FULL"]["output"] == expected

    # Ablations:
    # - without bias, topic-lock alone should not reliably reconstruct the full identity
    # - blank (no bias, no lock) should drift
    assert runs["NO_BIAS"]["output"] != expected
    assert runs["BLANK"]["output"] != expected

    # Note: depending on the random initialization, bias alone can still succeed.
    # That is an acceptable outcome: it means the basin is deep enough to be stable
    # even without the explicit epistemic boundary.
