import json, math, csv, argparse

def clamp01(p, eps=1e-12):
    p = float(p)
    if p < eps:
        return eps
    if p > 1.0:
        return 1.0
    return p

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")

def otsu_threshold(values):
    vals = sorted(set(float(v) for v in values))
    if len(vals) <= 2:
        return vals[len(vals)//2] if vals else 0.5
    best_t = vals[len(vals)//2]
    best_score = -1.0
    mT = mean(values)
    for t in vals[1:-1]:
        left = [v for v in values if v <= t]
        right = [v for v in values if v > t]
        if not left or not right:
            continue
        mL = mean(left)
        mR = mean(right)
        pL = len(left) / len(values)
        pR = len(right) / len(values)
        score = pL * (mL - mT) ** 2 + pR * (mR - mT) ** 2
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--thr", default="")
    args = ap.parse_args()

    d = json.load(open(args.in_json, "r", encoding="utf-8"))
    pts = d["points"]

    rows = []
    lock_vals = []

    for p in pts:
        w = float(p["weight"])
        lam = float(p["constraint_lambda"])
        lock_rate = float(p["lock_rate"])
        locked = int(p["locked"])
        total = int(p["total"])

        lock_vals.append(lock_rate)

        prob = clamp01(lock_rate)
        E = -math.log(prob)

        rows.append({
            "w": w,
            "lambda": lam,
            "lock_rate": lock_rate,
            "locked": locked,
            "total": total,
            "p": prob,
            "E": E,
        })

    if args.thr.strip():
        thr = float(args.thr.strip())
    else:
        thr = otsu_threshold(lock_vals)

    left_E = mean([r["E"] for r in rows if r["lock_rate"] <= thr])
    right_E = mean([r["E"] for r in rows if r["lock_rate"] > thr])
    delta_E = left_E - right_E

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["w","lambda","lock_rate","locked","total","p","E"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    summary = {
        "in_json": args.in_json,
        "definition": {
            "p_xstar_given_w_lambda": "lock_rate(w,lambda) = locked/total",
            "E_w_lambda": "E(w,lambda) = -log(p_xstar_given_w_lambda)"
        },
        "boundary": {"lock_rate_thr": float(thr)},
        "means": {
            "mean_E_below_or_equal_thr": float(left_E),
            "mean_E_above_thr": float(right_E),
            "delta_E": float(delta_E)
        },
        "claim": "E decreases past the same lock_rate boundary where lock_rate rises"
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("WROTE:", args.out_csv)
    print("WROTE:", args.out_summary)
    print("lock_rate boundary =", thr)
    print("mean(E) below boundary =", left_E)
    print("mean(E) above boundary =", right_E)
    print("delta(E) =", delta_E)

if __name__ == "__main__":
    main()
