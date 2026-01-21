from llm_identity.sculpt import preset_identity_conscious_agent
from llm_identity.phase_transition import PhaseConfig, run_phase_transition

if __name__ == "__main__":
    preset = preset_identity_conscious_agent()
    cfg = PhaseConfig(
        weights=tuple([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
        seeds=tuple(range(200)),
        steps=3,
        temperature=1.0,
        topic_lock=True,
    )
    rep = run_phase_transition(preset, cfg, out_dir="out/phase")
    print("PASS_PHASE_TRANSITION_WROTE_JSON")
    print("WROTE:", "out/phase/phase_transition.json")
    print("WROTE:", "out/phase/phase_transition.png")
