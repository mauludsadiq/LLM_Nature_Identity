from __future__ import annotations

import json

from llm_identity.witness import run_identity_suite, sha256_json
from llm_identity.sculpt import preset_identity_conscious_agent


def test_witness_digest_matches_canonical_json(tmp_path):
    w = run_identity_suite(out_dir=str(tmp_path), preset=preset_identity_conscious_agent(), seed=0)

    core = {k: w[k] for k in w if k != "digest_sha256"}
    assert w["digest_sha256"] == sha256_json(core)

    # Ensure the written file matches the in-memory witness
    p = tmp_path / "witness_identity.json"
    written = json.loads(p.read_text(encoding="utf-8"))
    assert written == w
