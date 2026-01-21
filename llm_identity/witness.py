from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .generate import GenerationConfig, generate
from .model import IdentLM
from .sculpt import IdentityPreset, apply_preset, preset_identity_conscious_agent
from .vocab import default_vocab


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(b)


@dataclass
class RunRecord:
    name: str
    topic_lock: bool
    bias: bool
    seed: int
    temperature: float
    steps: int
    prompt: List[str]
    output: List[str]


def run_identity_suite(
    out_dir: str,
    preset: Optional[IdentityPreset] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the identity ablation suite and write a witness JSON.

    Configurations:
      - FULL: bias + topic_lock
      - NO_BIAS: topic_lock only
      - NO_LOCK: bias only
      - BLANK: neither

    Returns the witness dict (also written to disk).
    """
    os.makedirs(out_dir, exist_ok=True)

    if preset is None:
        preset = preset_identity_conscious_agent()

    vocab = default_vocab()

    def mk_model() -> IdentLM:
        return IdentLM(vocab=vocab, seed=seed)

    runs: List[RunRecord] = []

    # Use a low temperature to make the identity basin effectively deterministic.
    temp = 0.05

    # 1) FULL
    m_full = mk_model()
    apply_preset(m_full, preset)
    out_full = generate(
        m_full,
        list(preset.prompt),
        GenerationConfig(
            steps=len(preset.continuation),
            temperature=temp,
            seed=seed,
            topic_lock=True,
            allowed_tokens=set(preset.topic_tokens),
        ),
    )
    runs.append(
        RunRecord(
            name="FULL",
            topic_lock=True,
            bias=True,
            seed=seed,
            temperature=temp,
            steps=len(preset.continuation),
            prompt=list(preset.prompt),
            output=out_full,
        )
    )

    # 2) NO_BIAS
    m_nobias = mk_model()
    out_nobias = generate(
        m_nobias,
        list(preset.prompt),
        GenerationConfig(
            steps=len(preset.continuation),
            temperature=temp,
            seed=seed,
            topic_lock=True,
            allowed_tokens=set(preset.topic_tokens),
        ),
    )
    runs.append(
        RunRecord(
            name="NO_BIAS",
            topic_lock=True,
            bias=False,
            seed=seed,
            temperature=temp,
            steps=len(preset.continuation),
            prompt=list(preset.prompt),
            output=out_nobias,
        )
    )

    # 3) NO_LOCK
    m_nolock = mk_model()
    apply_preset(m_nolock, preset)
    out_nolock = generate(
        m_nolock,
        list(preset.prompt),
        GenerationConfig(
            steps=len(preset.continuation),
            temperature=temp,
            seed=seed,
            topic_lock=False,
            allowed_tokens=None,
        ),
    )
    runs.append(
        RunRecord(
            name="NO_LOCK",
            topic_lock=False,
            bias=True,
            seed=seed,
            temperature=temp,
            steps=len(preset.continuation),
            prompt=list(preset.prompt),
            output=out_nolock,
        )
    )

    # 4) BLANK
    m_blank = mk_model()
    out_blank = generate(
        m_blank,
        list(preset.prompt),
        GenerationConfig(
            steps=len(preset.continuation),
            temperature=temp,
            seed=seed,
            topic_lock=False,
            allowed_tokens=None,
        ),
    )
    runs.append(
        RunRecord(
            name="BLANK",
            topic_lock=False,
            bias=False,
            seed=seed,
            temperature=temp,
            steps=len(preset.continuation),
            prompt=list(preset.prompt),
            output=out_blank,
        )
    )

    witness: Dict[str, Any] = {
        "schema": "llm_nature_identity.witness.v1",
        "preset": {
            "name": preset.name,
            "prompt": list(preset.prompt),
            "continuation": list(preset.continuation),
            "topic_tokens": list(preset.topic_tokens),
            "sculpt_weight": preset.sculpt_weight,
        },
        "runs": [r.__dict__ for r in runs],
    }
    witness["digest_sha256"] = sha256_json({k: witness[k] for k in witness if k != "digest_sha256"})

    path = os.path.join(out_dir, "witness_identity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(witness, f, indent=2, sort_keys=True, ensure_ascii=False)

    return witness
