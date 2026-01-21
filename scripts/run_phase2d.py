from llm_identity.sculpt import preset_identity_conscious_agent
from llm_identity.phase_transition_2d import Phase2DConfig, run_phase_transition_2d

if __name__ == "__main__":
    preset = preset_identity_conscious_agent()

    cfg = Phase2DConfig(
        weights=tuple([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
        lambdas=tuple([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        seeds=tuple(range(200)),
        steps=3,
        temperature=1.0,
        topic_lock=True,
    )

    run_phase_transition_2d(preset, cfg, out_dir="out/phase2d")
    print("PASS_PHASE_TRANSITION_2D_WROTE_JSON")
    print("WROTE: out/phase2d/phase_transition_2d.json")
    print("WROTE: out/phase2d/phase_transition_2d.png")
