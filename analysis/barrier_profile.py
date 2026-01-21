import json, math, argparse, os

def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")

def linfit(x, y):
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    varx = sum((x[i] - mx) ** 2 for i in range(n))
    if varx <= 0:
        return float("nan"), float("nan"), float("nan")
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    a = cov / varx
    b = my - a * mx
    yhat = [a * x[i] + b for i in range(n)]
    sse = sum((y[i] - yhat[i]) ** 2 for i in range(n))
    sst = sum((y[i] - my) ** 2 for i in range(n))
    r2 = 1.0 - (sse / sst) if sst > 0 else float("nan")
    return a, b, r2

def unique_sorted(xs):
    seen = {}
    for v in xs:
        seen[float(v)] = 1
    return sorted(seen.keys())

def get_w(p):
    return float(p.get("weight", p.get("w", 0.0)))

def get_lam(p):
    return float(p.get("constraint_lambda", p.get("lambda", 0.0)))

def get_lock(p):
    return float(p.get("lock_rate", 0.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    d = json.load(open(args.in_json, "r", encoding="utf-8"))
    pts = d.get("points", [])
    if not pts:
        raise SystemExit("no points found in JSON")

    lambdas = unique_sorted(d.get("lambdas", [get_lam(p) for p in pts]))
    weights = unique_sorted(d.get("weights", [get_w(p) for p in pts]))

    by_lam = {}
    for p in pts:
        lam = get_lam(p)
        by_lam.setdefault(lam, []).append(p)

    rows = []
    for lam in lambdas:
        slice_pts = by_lam.get(lam, [])
        if not slice_pts:
            rows.append({
                "lambda": lam,
                "n": 0,
                "a": None,
                "w0": None,
                "width": None,
                "r2": None,
                "note": "no points for this lambda"
            })
            continue

        slice_pts = sorted(slice_pts, key=get_w)

        xs = []
        ys = []
        locks = []
        for p in slice_pts:
            w = get_w(p)
            lr = get_lock(p)
            lr = clamp(lr, args.eps, 1.0 - args.eps)
            logit = math.log(lr / (1.0 - lr))
            xs.append(w)
            ys.append(logit)
            locks.append(lr)

        a, b, r2 = linfit(xs, ys)

        if a == a and a != 0.0:
            w0 = -b / a
            width = 1.0 / abs(a)
        else:
            w0 = float("nan")
            width = float("nan")

        lock_min = min(locks) if locks else float("nan")
        lock_max = max(locks) if locks else float("nan")
        lock_span = lock_max - lock_min if lock_min == lock_min and lock_max == lock_max else float("nan")

        note = ""
        if lock_span == lock_span and lock_span < 0.10:
            note = "weak transition: lock_rate span < 0.10"
        if not (a == a) or a == 0.0:
            note = "degenerate fit: slope undefined/zero"

        rows.append({
            "lambda": lam,
            "n": len(xs),
            "a": float(a) if a == a else None,
            "w0": float(w0) if w0 == w0 else None,
            "width": float(width) if width == width else None,
            "r2": float(r2) if r2 == r2 else None,
            "lock_min": float(lock_min) if lock_min == lock_min else None,
            "lock_max": float(lock_max) if lock_max == lock_max else None,
            "lock_span": float(lock_span) if lock_span == lock_span else None,
            "note": note
        })

    good = [r for r in rows if r.get("width") is not None]
    widths = [r["width"] for r in good if isinstance(r["width"], float)]
    width_min = min(widths) if widths else None
    width_max = max(widths) if widths else None
    width_mean = mean(widths) if widths else None

    out = {
        "in_json": args.in_json,
        "definition": {
            "lock_rate": "locked/total from points",
            "fit": "logit(lock_rate) = a*w + b per lambda slice",
            "ridge_center_w0": "w0 = -b/a",
            "ridge_width": "width = 1/|a|"
        },
        "eps": args.eps,
        "grid": {
            "num_points": len(pts),
            "num_lambdas": len(lambdas),
            "num_weights": len(weights)
        },
        "summary": {
            "width_min": width_min,
            "width_max": width_max,
            "width_mean": width_mean
        },
        "barrier_profile": rows
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("WROTE:", args.out_json)
    if width_mean is not None:
        print("width_mean =", width_mean, "width_min =", width_min, "width_max =", width_max)

if __name__ == "__main__":
    main()
