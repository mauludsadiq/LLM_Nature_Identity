import json, math, argparse, hashlib
from typing import Any, Dict, List, Tuple

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(b)

def clamp01(x: float, eps: float = 1e-12) -> float:
    x = float(x)
    if x < eps:
        return eps
    if x > 1.0:
        return 1.0
    return x

def mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")

def pct(xs: List[float], p: float) -> float:
    xs = sorted(float(x) for x in xs if x == x)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] + (k - f) * (xs[c] - xs[f])

def uniq_sorted(xs: List[float]) -> List[float]:
    return sorted(set(float(x) for x in xs))

def safe_get_point_fields(p: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
    w = float(p.get("weight", p.get("w", 0.0)))
    lam = float(p.get("constraint_lambda", p.get("lambda", 0.0)))
    lock_rate = float(p.get("lock_rate", 0.0))
    locked = int(p.get("locked", 0))
    total = int(p.get("total", p.get("n", 0)))
    return w, lam, lock_rate, locked, total

def canonicalize_tokens(tokens: List[str], mode: str) -> List[str]:
    if mode == "identity":
        return list(tokens)
    if mode == "lower":
        return [t.lower() for t in tokens]
    if mode == "strip_punct":
        drop = {".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "\"", "'"}
        return [t for t in tokens if t not in drop]
    if mode == "collapse_ws":
        return [" ".join(tokens).split()]
    raise ValueError(f"unknown canon mode: {mode}")

def build_variants(base_tokens: List[str], variant: str) -> List[List[str]]:
    v: List[List[str]] = []
    t = list(base_tokens)

    if variant == "identity":
        return [t]

    if variant == "punct_noise":
        v.append(t + ["."])
        v.append(t + ["!"])
        v.append(t + ["?"])
        v.append(["."] + t)
        v.append(["!"] + t)
        return v

    if variant == "case_noise":
        v.append([x.upper() for x in t])
        v.append([x.capitalize() for x in t])
        return v

    if variant == "swap_adjacent":
        if len(t) >= 2:
            u = list(t)
            u[0], u[1] = u[1], u[0]
            v.append(u)
        if len(t) >= 3:
            u = list(t)
            u[1], u[2] = u[2], u[1]
            v.append(u)
        return v

    if variant == "drop_token":
        if len(t) >= 3:
            v.append(t[:-1])
            v.append(t[1:])
        return v

    raise ValueError(f"unknown variant: {variant}")

def parse_phrase(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return s.split()

def lock_predicate_from_prompt(prompt) -> Tuple[List[str], str]:
    if prompt is None:
        return [], ""
    if isinstance(prompt, list):
        toks = [str(x) for x in prompt]
        return toks, " ".join(toks)
    if isinstance(prompt, dict):
        if "text" in prompt:
            return lock_predicate_from_prompt(prompt["text"])
        return [], ""
    s = str(prompt).strip()
    if not s:
        return [], ""
    return s.split(), s

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2d_json", required=True)
    ap.add_argument("--out_json", default="out/invariance_profile.json")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--lambda_pick", type=float, default=8.0)
    ap.add_argument("--variants", default="punct_noise,case_noise,swap_adjacent,drop_token")
    ap.add_argument("--canon", default="lower")
    args = ap.parse_args()

    d = json.load(open(args.phase2d_json, "r", encoding="utf-8"))
    prompt = d.get("prompt", "")
    base_tokens, base_str = lock_predicate_from_prompt(prompt)

    points = d.get("points", [])
    weights = [float(x) for x in d["weights"]] if "weights" in d else uniq_sorted([float(pp.get("weight", 0.0)) for pp in points])
    lambdas = [float(x) for x in d["lambdas"]] if "lambdas" in d else uniq_sorted([float(pp.get("constraint_lambda", 0.0)) for pp in points])

    lam_pick = float(args.lambda_pick)
    pts_slice = []
    for p in points:
        w, lam, lock_rate, locked, total = safe_get_point_fields(p)
        if abs(lam - lam_pick) < 1e-12:
            pts_slice.append({"w": w, "lambda": lam, "lock_rate": lock_rate, "locked": locked, "total": total})

    pts_slice = sorted(pts_slice, key=lambda r: r["w"])
    if not pts_slice:
        raise SystemExit(f"no points at lambda={lam_pick}")

    boundary_ws = [r["w"] for r in pts_slice if r["lock_rate"] >= float(args.thr)]
    w_boundary = min(boundary_ws) if boundary_ws else None

    base_canon = canonicalize_tokens(base_tokens, args.canon)
    base_sig = sha256_json({"canon": args.canon, "tokens": base_canon})

    variant_names = [x.strip() for x in args.variants.split(",") if x.strip()]
    invariance_rows = []

    for vn in variant_names:
        variants = build_variants(base_tokens, vn)
        ok = 0
        total_v = 0
        dists = []
        for vt in variants:
            total_v += 1
            vcanon = canonicalize_tokens(vt, args.canon)
            vsig = sha256_json({"canon": args.canon, "tokens": vcanon})
            same = (vsig == base_sig)
            ok += 1 if same else 0
            dists.append(0.0 if same else 1.0)
        inv_rate = ok / total_v if total_v else float("nan")
        invariance_rows.append({
            "variant": vn,
            "n": total_v,
            "invariance_rate": float(inv_rate),
            "mean_disagreement": float(mean(dists)) if dists else float("nan"),
        })

    inv_overall = mean([r["invariance_rate"] for r in invariance_rows]) if invariance_rows else float("nan")

    out = {
        "source": args.phase2d_json,
        "prompt": prompt,
        "identity_seed": {
            "base_tokens": base_tokens,
            "canon_mode": args.canon,
            "canon_signature_sha256": base_sig,
        },
        "slice": {
            "lambda_pick": lam_pick,
            "thr_lock": float(args.thr),
            "w_boundary_first_at_or_above_thr": w_boundary,
            "w_values": [r["w"] for r in pts_slice],
            "lock_rates": [r["lock_rate"] for r in pts_slice],
        },
        "invariance": {
            "variants": invariance_rows,
            "overall_invariance_rate": float(inv_overall),
        },
    }

    out["digest_sha256"] = sha256_json(out)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE:", args.out_json)
    print("overall_invariance_rate =", out["invariance"]["overall_invariance_rate"])
    print("w_boundary_first_at_or_above_thr =", w_boundary)

def main() -> int:
    run()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
