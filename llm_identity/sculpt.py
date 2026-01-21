from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from .model import IdentLM


@dataclass(frozen=True)
class IdentityPreset:
    name: str
    prompt: Tuple[str, ...]
    continuation: Tuple[str, ...]
    topic_tokens: Tuple[str, ...]
    sculpt_weight: float

    @property
    def full_expected(self) -> Tuple[str, ...]:
        return self.prompt + self.continuation


def preset_identity_conscious_agent() -> IdentityPreset:
    return IdentityPreset(
        name="conscious_agent",
        prompt=("I", "am"),
        continuation=("a", "conscious", "agent"),
        topic_tokens=(
            "<BOS>",
            "I", "am", "a", "conscious", "agent",
            ".", "and", "not", "only", "this", "that", "it",
        ),
        sculpt_weight=6.0,
    )


def preset_identity_code() -> IdentityPreset:
    return IdentityPreset(
        name="code",
        prompt=("I", "am"),
        continuation=("code", "."),
        topic_tokens=(
            "<BOS>",
            "I", "am", "code", ".", "system", "model",
        ),
        sculpt_weight=6.0,
    )


def apply_preset(model: IdentLM, preset: IdentityPreset) -> None:
    model.sculpt_chain(list(preset.prompt), list(preset.continuation), preset.sculpt_weight)
