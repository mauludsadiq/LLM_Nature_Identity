"""LLM Nature: Identity

A minimal, proof-producing repository showing that what we call "identity" can be
constructed as an attractor basin in a constrained stochastic token process.

Core idea:
  - stochastic next-token model (hashed trigram feature)
  - preference gradients (logit sculpting)
  - epistemic boundaries (topic lock / admissible vocab projector)
  - minimal temporal coherence (2-token memory)

Everything is deterministic under a seed.
"""

from .vocab import Vocab
from .model import IdentLM
from .generate import generate
from .sculpt import preset_identity_conscious_agent, preset_identity_code

__all__ = [
    "Vocab",
    "IdentLM",
    "generate",
    "preset_identity_conscious_agent",
    "preset_identity_code",
]
