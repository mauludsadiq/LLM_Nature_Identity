from llm_identity.sculpt import preset_identity_conscious_agent
from llm_identity.phase_transition import PhaseConfig, run_phase_transition


def test_phase_transition_smoke(tmp_path):
    preset = preset_identity_conscious_agent()
    cfg = PhaseConfig(
        weights=(0.0, 2.0, 6.0),
        seeds=tuple(range(50)),
        steps=3,
        temperature=1.0,
        topic_lock=True,
    )
    out_dir = tmp_path / "phase"
    rep = run_phase_transition(preset, cfg, out_dir=str(out_dir))

    assert (out_dir / "phase_transition.json").exists()
    assert (out_dir / "phase_transition.png").exists()
    assert len(rep["points"]) == 3
