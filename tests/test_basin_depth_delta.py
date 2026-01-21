from llm_identity.model import IdentLM
from llm_identity.vocab import make_default_vocab
from llm_identity.sculpt import preset_identity_conscious_agent, apply_preset
from llm_identity.metrics import continuation_delta


def test_delta_is_positive_for_sculpted_identity():
    vocab = make_default_vocab()
    preset = preset_identity_conscious_agent()

    base = IdentLM(vocab=vocab, seed=0)
    biased = IdentLM(vocab=vocab, seed=0)
    apply_preset(biased, preset)

    d = continuation_delta(
        base,
        biased,
        prompt=preset.prompt,
        continuation=preset.continuation,
        topic_lock=False,
        allowed_tokens=None,
    )
    assert d > 0.0
