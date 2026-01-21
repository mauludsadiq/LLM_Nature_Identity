from llm_identity.sculpt import preset_identity_conscious_agent
from llm_identity.phase_transition_2d import Phase2DConfig, run_phase_transition_2d


def test_phase_transition_2d_smoke(tmp_path):
    preset = preset_identity_conscious_agent()
    cfg = Phase2DConfig(
        weights=(0.0, 2.0, 6.0),
        lambdas=(0.0, 4.0, 16.0),
        seeds=tuple(range(50)),
        steps=3,
        temperature=1.0,
        topic_lock=True,
    )
    out_dir = tmp_path / "phase2d"
    rep = run_phase_transition_2d(preset, cfg, out_dir=str(out_dir))

    assert (out_dir / "phase_transition_2d.json").exists()
    assert (out_dir / "phase_transition_2d.png").exists()
    assert len(rep["grid_lock_rate"]) == 3
    assert len(rep["grid_lock_rate"][0]) == 3
