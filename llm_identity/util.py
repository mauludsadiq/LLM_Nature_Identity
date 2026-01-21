from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(b)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True)


def hmod(s: str, mod: int) -> int:
    """Stable hash-mod using sha256 (no Python hash randomization)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    # take 8 bytes for an int
    x = int.from_bytes(h[:8], byteorder="big", signed=False)
    return x % mod


def now_iso_utc() -> str:
    # deterministic repo: callers should pass time explicitly if desired
    # but for witness metadata, we keep a separate field they can ignore.
    import datetime as _dt

    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
