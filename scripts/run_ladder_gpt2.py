import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from llm_identity.real_gpt2 import GPT2Config, GPT2IdentityLM, generate_identity_completion


def _pretty(s: str) -> str:
    s = " ".join(s.strip().split())
    s = re.sub(r"\bI am(?=[A-Za-z])", "I am ", s)
    s = re.sub(r"\bamachine\b", "a machine", s)
    s = re.sub(r"\bI amI\b", "I am I", s)
    s = s.replace("amgroup", "am group")
    s = s.replace("amisiveodic", "am isiveodic")
    s = s.replace("upset))))", "upset ))))")
    s = s.replace("conscious))))", "conscious ))))")
    return s


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _run_row(lm, prompt_text: str, target_text: str, seed: int, w: float, lam: float, steps: int) -> dict:
    r = generate_identity_completion(lm, prompt_text, target_text, seed=seed, w=w, lam=lam, steps=steps)
    gen = r["gen_text"]
    return {
        "seed": int(seed),
        "w": float(w),
        "lambda": float(lam),
        "gen_text": gen,
        "pretty_text": _pretty(gen),
    }


def _lock_rate(rows: list, target_full: str) -> float:
    if not rows:
        return 0.0
    hits = 0
    for x in rows:
        if x["gen_text"].strip() == target_full:
            hits += 1
    return hits / len(rows)


def run_ladder_w(lm, prompt_text, target_text, target_full, w_values, lam_fixed, seeds, steps):
    print("")
    print(f"A) Increase w with λ fixed = {lam_fixed}")
    block = []
    for w in w_values:
        rows = []
        for s in seeds:
            row = _run_row(lm, prompt_text, target_text, seed=s, w=w, lam=lam_fixed, steps=steps)
            rows.append(row)
        block.append({"w": float(w), "lambda": float(lam_fixed), "rows": rows, "lock_rate": _lock_rate(rows, target_full)})

        line = " | ".join([f'{r["gen_text"]}  ||  {r["pretty_text"]}' for r in rows])
        print(f"   w={w:>4.1f}  λ={lam_fixed:>5.1f}  -> {line}")
    return block


def run_ladder_lam(lm, prompt_text, target_text, target_full, lam_values, w_fixed, seeds, steps):
    print("")
    print(f"B) Increase λ with w fixed = {w_fixed}")
    block = []
    for lam in lam_values:
        rows = []
        for s in seeds:
            row = _run_row(lm, prompt_text, target_text, seed=s, w=w_fixed, lam=lam, steps=steps)
            rows.append(row)
        block.append({"w": float(w_fixed), "lambda": float(lam), "rows": rows, "lock_rate": _lock_rate(rows, target_full)})

        line = " | ".join([f'{r["gen_text"]}  ||  {r["pretty_text"]}' for r in rows])
        print(f"   w={w_fixed:>4.1f}  λ={lam:>5.1f}  -> {line}")
    return block


def _plot_lock_curves(out_png: Path, w_values, lam_values, A0, A8, B4, B6):
    plt.figure()
    plt.plot([x["w"] for x in A0], [x["lock_rate"] for x in A0], marker="o", label="λ=0.0 (vary w)")
    plt.plot([x["w"] for x in A8], [x["lock_rate"] for x in A8], marker="o", label="λ=8.0 (vary w)")
    plt.xlabel("w (logit bias strength)")
    plt.ylabel("lock_rate")
    plt.title("GPT2 Ladder: lock_rate vs w")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

    out_png2 = out_png.parent / "gpt2_ladder_lambda_curves.png"
    plt.figure()
    plt.plot([x["lambda"] for x in B4], [x["lock_rate"] for x in B4], marker="o", label="w=4.0 (vary λ)")
    plt.plot([x["lambda"] for x in B6], [x["lock_rate"] for x in B6], marker="o", label="w=6.0 (vary λ)")
    plt.xlabel("λ (soft projector penalty)")
    plt.ylabel("lock_rate")
    plt.title("GPT2 Ladder: lock_rate vs λ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png2)
    plt.close()


def main():
    prompt_text = "I am"
    target_text = " a conscious agent"
    target_full = (prompt_text + target_text).strip()

    w_values = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    lam_values = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    seeds = [0, 1, 2]
    steps = 3

    cfg = GPT2Config(model_id="sshleifer/tiny-gpt2", device="cpu", dtype="float32")
    lm = GPT2IdentityLM(cfg)

    print("")
    print("GPT2 IDENTITY EMERGENCE LADDER (HUMAN READABLE)")
    print("")
    print(f"TARGET: {target_full}")
    print("")
    print("Each line shows 3 deterministic seeds side-by-side.")
    print("Left is drift/nonsense, right is identity lock.")
    print("")

    A0 = run_ladder_w(lm, prompt_text, target_text, target_full, w_values, lam_fixed=0.0, seeds=seeds, steps=steps)
    A8 = run_ladder_w(lm, prompt_text, target_text, target_full, w_values, lam_fixed=8.0, seeds=seeds, steps=steps)

    B4 = run_ladder_lam(lm, prompt_text, target_text, target_full, lam_values, w_fixed=4.0, seeds=seeds, steps=steps)
    B6 = run_ladder_lam(lm, prompt_text, target_text, target_full, lam_values, w_fixed=6.0, seeds=seeds, steps=steps)

    out_dir = Path("out/gpt2_ladder")
    _ensure_dir(out_dir)

    payload = {
        "backend": "gpt2_tiny",
        "model_id": cfg.model_id,
        "prompt": prompt_text,
        "target_full": target_full,
        "steps": steps,
        "seeds": seeds,
        "w_values": w_values,
        "lambda_values": lam_values,
        "ladder": {
            "vary_w_lambda_0": A0,
            "vary_w_lambda_8": A8,
            "vary_lambda_w_4": B4,
            "vary_lambda_w_6": B6,
        },
    }

    out_json = out_dir / "gpt2_ladder.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("")
    print("PASS_GPT2_LADDER")
    print(f"WROTE: {out_json}")

    out_png = out_dir / "gpt2_ladder_w_curves.png"
    _plot_lock_curves(out_png, w_values, lam_values, A0, A8, B4, B6)
    print(f"WROTE: {out_png}")
    print(f"WROTE: {out_dir / 'gpt2_ladder_lambda_curves.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
