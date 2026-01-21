from __future__ import annotations

import argparse
import json
import os
import sys

from .sculpt import preset_identity_conscious_agent, preset_identity_code
from .witness import run_identity_suite


def _print_run(name: str, output: list[str]) -> None:
    print(f"{name}: {' '.join(output)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="identity-demo",
        description="LLM Nature Identity: identity as an attractor in constrained token dynamics",
    )
    p.add_argument("--out-dir", default="out", help="Output directory for witness artifacts")
    p.add_argument("--seed", type=int, default=0, help="Deterministic seed")
    p.add_argument("--preset", choices=["conscious_agent", "code"], default="conscious_agent")

    args = p.parse_args(argv)

    preset = preset_identity_conscious_agent() if args.preset == "conscious_agent" else preset_identity_code()
    witness = run_identity_suite(out_dir=args.out_dir, preset=preset, seed=args.seed)

    # Print summary
    print("=== LLM Nature: Identity Demo ===")
    print(f"preset={witness['preset']['name']} seed={args.seed}")

    outputs = {r["name"]: r["output"] for r in witness["runs"]}
    for k in ["FULL", "NO_BIAS", "NO_LOCK", "BLANK"]:
        _print_run(k, outputs[k])

    expected = witness["preset"]["prompt"] + witness["preset"]["continuation"]

    # PASS lines (proof-producing)
    if outputs["FULL"] == expected:
        print("PASS_IDENTITY_ATTRACTOR_FULL")
    else:
        print("FAIL_IDENTITY_ATTRACTOR_FULL")
        print("expected:", " ".join(expected))

    # These aren't logically required to fail, but are strong empirical checks.
    if outputs["NO_BIAS"] != expected:
        print("PASS_ABLATION_NO_BIAS_DRIFTS")
    else:
        print("WARN_ABLATION_NO_BIAS_DID_NOT_DRIFT")

    if outputs["NO_LOCK"] != expected:
        print("PASS_ABLATION_NO_LOCK_DRIFTS")
    else:
        print("WARN_ABLATION_NO_LOCK_DID_NOT_DRIFT")

    if outputs["BLANK"] != expected:
        print("PASS_ABLATION_BLANK_DRIFTS")
    else:
        print("WARN_ABLATION_BLANK_DID_NOT_DRIFT")

    print(f"WROTE: {os.path.join(args.out_dir, 'witness_identity.json')}")
    print(f"DIGEST: {witness['digest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
