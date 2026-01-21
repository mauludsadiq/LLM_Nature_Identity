from llm_identity.sculpt import preset_identity_conscious_agent, preset_identity_code
from llm_identity.competition import CompetitionConfig, run_competition


def test_competition_prefers_higher_weight_basin():
    A = preset_identity_conscious_agent()
    B = preset_identity_code()

    # enforce same prompt
    assert A.prompt == B.prompt == ("I", "am")

    seeds = list(range(50))  # deterministic seed set
    cfg = CompetitionConfig(prompt=A.prompt, steps=4, temperature=1.0, topic_lock=False)

    # A heavier than B → A should win majority
    rep = run_competition(
        preset_a=A,
        preset_b=B,
        weight_a=8.0,
        weight_b=2.0,
        seeds=seeds,
        cfg=cfg,
        model_seed=0,
    )

    assert rep["counts"][A.name] > rep["counts"][B.name]
