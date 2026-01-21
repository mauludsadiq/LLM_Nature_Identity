import json, math, csv, argparse, os
import numpy as np

def clamp_prob(p, pmin):
    p = float(p)
    if not (p == p):
        return pmin
    if p < pmin:
        return pmin
    if p > 1.0:
        return 1.0
    return p

def mean(xs):
    xs = [float(x) for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")

def unique_sorted(vals):
    out = sorted(set(float(v) for v in vals))
    return out

def grid_from_points(points, weights, lambdas):
    w_idx = {float(w): i for i, w in enumerate(weights)}
    l_idx = {float(l): i for i, l in enumerate(lambdas)}
    H = len(lambdas)
    W = len(weights)
    lock = np.zeros((H, W), dtype=float)
    total = np.zeros((H, W), dtype=float)
    for p in points:
        w = float(p.get("weight", p.get("w", 0.0)))
        lam = float(p.get("constraint_lambda", p.get("lambda", 0.0)))
        i = l_idx[lam]
        j = w_idx[w]
        lock[i, j] = float(p["lock_rate"])
        total[i, j] = float(p.get("total", 0))
    return lock, total

def logistic(x, k, x0):
    z = -k * (x - x0)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))

def fit_logistic_grid(x, y, k_grid=None, x0_grid=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4:
        return None
    if k_grid is None:
        k_grid = np.logspace(-2, 2.5, 140)
    if x0_grid is None:
        x0_grid = np.linspace(np.min(x), np.max(x), 140)
    best = None
    best_mse = 1e99
    for k in k_grid:
        for x0 in x0_grid:
            pred = logistic(x, k, x0)
            mse = float(np.mean((pred - y) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = (float(k), float(x0))
    if best is None:
        return None
    k, x0 = best
    return {"k": k, "x0": x0, "mse": best_mse, "width_1_over_k": (1.0 / k if k > 0 else float("inf"))}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--E_cap", type=float, default=20.0)
    ap.add_argument("--thresholds", default="0.1,0.5,0.9,0.99")
    ap.add_argument("--slice", default="")
    ap.add_argument("--slice_value", default="")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    d = json.load(open(args.in_json, "r", encoding="utf-8"))
    points = d["points"]
    weights = [float(x) for x in d["weights"]] if "weights" in d else unique_sorted([float(pp.get("weight", pp.get("w", 0.0))) for pp in d.get("points", [])])
    lambdas = [float(x) for x in d["lambdas"]] if "lambdas" in d else unique_sorted([float(pp.get("constraint_lambda", pp.get("lambda", 0.0))) for pp in d.get("points", [])])

    pmin = math.exp(-float(args.E_cap))

    rows = []
    lock_vals = []
    for p in points:
        w = float(p.get("weight", p.get("w", 0.0)))
        lam = float(p.get("constraint_lambda", p.get("lambda", 0.0)))
        lock_rate = float(p["lock_rate"])
        locked = int(p.get("locked", 0))
        total = int(p.get("total", 0))
        lock_vals.append(lock_rate)
        prob = clamp_prob(lock_rate, pmin)
        E = -math.log(prob)
        if E > args.E_cap:
            E = float(args.E_cap)
        rows.append({
            "w": w,
            "lambda": lam,
            "lock_rate": lock_rate,
            "locked": locked,
            "total": total,
            "p_clamped": prob,
            "E_cap": E,
        })

    out_csv = os.path.join(args.out_dir, "energy_landscape.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["w","lambda","lock_rate","locked","total","p_clamped","E_cap"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    thr_list = []
    for s in args.thresholds.split(","):
        s = s.strip()
        if s:
            thr_list.append(float(s))

    thr_stats = []
    for thr in thr_list:
        left_E = mean([r["E_cap"] for r in rows if r["lock_rate"] <= thr])
        right_E = mean([r["E_cap"] for r in rows if r["lock_rate"] > thr])
        delta_E = left_E - right_E
        thr_stats.append({
            "lock_rate_thr": float(thr),
            "mean_E_below_or_equal_thr": float(left_E),
            "mean_E_above_thr": float(right_E),
            "delta_E": float(delta_E),
        })

    barrier = None
    slice_meta = None

    if args.slice.strip() and args.slice_value.strip():
        mode = args.slice.strip().lower()
        val = float(args.slice_value.strip())

        if mode == "lambda":
            xs = []
            ys = []
            for r in rows:
                if float(r["lambda"]) == val:
                    xs.append(float(r["w"]))
                    ys.append(float(r["lock_rate"]))
            if len(xs) >= 4:
                barrier = fit_logistic_grid(xs, ys)
                slice_meta = {"mode": "lambda_fixed", "lambda": val, "x_axis": "w"}
        elif mode == "weight":
            xs = []
            ys = []
            for r in rows:
                if float(r["w"]) == val:
                    xs.append(float(r["lambda"]))
                    ys.append(float(r["lock_rate"]))
            if len(xs) >= 4:
                barrier = fit_logistic_grid(xs, ys)
                slice_meta = {"mode": "weight_fixed", "weight": val, "x_axis": "lambda"}

    summary = {
        "in_json": args.in_json,
        "out_dir": args.out_dir,
        "definition": {
            "p_xstar_given_w_lambda": "lock_rate(w,lambda) = locked/total",
            "E_w_lambda": "E(w,lambda) = -log(p_xstar_given_w_lambda)",
            "E_cap": float(args.E_cap),
            "pmin": float(pmin),
            "E_plot_value": "E_cap = min(-log(max(lock_rate,pmin)), E_cap)"
        },
        "threshold_sweep": thr_stats,
        "barrier_fit": barrier,
        "slice_meta": slice_meta
    }

    out_summary = os.path.join(args.out_dir, "energy_landscape_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("WROTE:", out_csv)
    print("WROTE:", out_summary)

    if args.plot:
        import matplotlib.pyplot as plt

        lock_grid, _ = grid_from_points(points, weights, lambdas)
        E_grid = np.zeros_like(lock_grid)
        for i in range(lock_grid.shape[0]):
            for j in range(lock_grid.shape[1]):
                p = clamp_prob(lock_grid[i, j], pmin)
                E = -math.log(p)
                if E > args.E_cap:
                    E = float(args.E_cap)
                E_grid[i, j] = E

        plt.figure()
        plt.imshow(lock_grid, aspect="auto", origin="lower")
        plt.xticks(range(len(weights)), [str(w) for w in weights], rotation=0)
        plt.yticks(range(len(lambdas)), [str(l) for l in lambdas], rotation=0)
        plt.xlabel("w")
        plt.ylabel("lambda")
        plt.title("lock_rate(w,lambda)")
        plt.colorbar()
        out_lock_png = os.path.join(args.out_dir, "lock_rate_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_lock_png, dpi=160)
        plt.close()

        plt.figure()
        plt.imshow(E_grid, aspect="auto", origin="lower")
        plt.xticks(range(len(weights)), [str(w) for w in weights], rotation=0)
        plt.yticks(range(len(lambdas)), [str(l) for l in lambdas], rotation=0)
        plt.xlabel("w")
        plt.ylabel("lambda")
        plt.title("E_cap(w,lambda)")
        plt.colorbar()
        out_E_png = os.path.join(args.out_dir, "energy_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_E_png, dpi=160)
        plt.close()

        if barrier is not None and slice_meta is not None:
            if slice_meta["mode"] == "lambda_fixed":
                lam = slice_meta["lambda"]
                xs = []
                ys = []
                for r in rows:
                    if float(r["lambda"]) == lam:
                        xs.append(float(r["w"]))
                        ys.append(float(r["lock_rate"]))
                x = np.array(xs, dtype=float)
                y = np.array(ys, dtype=float)
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                k = float(barrier["k"])
                x0 = float(barrier["x0"])
                pred = logistic(x, k, x0)
                plt.figure()
                plt.plot(x, y, marker="o")
                plt.plot(x, pred)
                plt.xlabel("w")
                plt.ylabel("lock_rate")
                plt.title("1D slice + logistic fit (lambda fixed)")
                out_slice_png = os.path.join(args.out_dir, "slice_logistic_fit.png")
                plt.tight_layout()
                plt.savefig(out_slice_png, dpi=160)
                plt.close()
            elif slice_meta["mode"] == "weight_fixed":
                w = slice_meta["weight"]
                xs = []
                ys = []
                for r in rows:
                    if float(r["w"]) == w:
                        xs.append(float(r["lambda"]))
                        ys.append(float(r["lock_rate"]))
                x = np.array(xs, dtype=float)
                y = np.array(ys, dtype=float)
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                k = float(barrier["k"])
                x0 = float(barrier["x0"])
                pred = logistic(x, k, x0)
                plt.figure()
                plt.plot(x, y, marker="o")
                plt.plot(x, pred)
                plt.xlabel("lambda")
                plt.ylabel("lock_rate")
                plt.title("1D slice + logistic fit (weight fixed)")
                out_slice_png = os.path.join(args.out_dir, "slice_logistic_fit.png")
                plt.tight_layout()
                plt.savefig(out_slice_png, dpi=160)
                plt.close()

        print("WROTE:", out_lock_png)
        print("WROTE:", out_E_png)
        if barrier is not None and slice_meta is not None:
            print("WROTE:", os.path.join(args.out_dir, "slice_logistic_fit.png"))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
