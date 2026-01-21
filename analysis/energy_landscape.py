import json
import math
import csv
import argparse
from typing import Dict, Any, List, Tuple

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clamp(p: float, eps: float = 1e-12) -> float:
    if p < eps:
        return eps
    if p > 1.0:
        return 1.0
    return p

def energy_from_probs(probs: List[float]) -> float:
    if len(probs) == 0:
        return float("nan")
    s = 0.0
    for p in probs:
        p = clamp(float(p))
        s += -math.log(p)
    return s / len(probs)

def best_boundary_by_energy_drop(rows: List[Dict[str, float]]) -> Dict[str, float]:
    clean = [r for r in rows if (not math.isnan(r["E"])) and (not math.isnan(r["lock_rate"]))]
    clean.sort(key=lambda r: r["lock_rate"])
    if len(clean) < 6:
        return {}

    prefixE = [0.0]
    for r in clean:
        prefixE.append(prefixE[-1] + r["E"])

    best = {
        "idx": -1,
        "lock_rate_thr": float("nan"),
        "E_left": float("nan"),
        "E_right": float("nan"),
        "drop": -1e18
    }

    n = len(clean)
    for i in range(2, n - 2):
        nL = i
        nR = n - i
        EL = prefixE[i] / nL
        ER = (prefixE[n] - prefixE[i]) / nR
        drop = EL - ER
        if drop > best["drop"]:
            best["idx"] = i
            best["lock_rate_thr"] = clean[i]["lock_rate"]
            best["E_left"] = EL
            best["E_right"] = ER
            best["drop"] = drop

    return best

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="JSON: probs[lambda][w] -> list of p(x*|w,lambda) per example")
    ap.add_argument("--lock", required=True, help="JSON: lock_rate[lambda][w] -> float")
    ap.add_argument("--out_csv", default="out/energy_landscape.csv")
    ap.add_argument("--out_json", default="out/energy_landscape_summary.json")
    args = ap.parse_args()

    probs_grid: Dict[str, Dict[str, List[float]]] = load_json(args.probs)
    lock_grid: Dict[str, Dict[str, float]] = load_json(args.lock)

    rows: List[Dict[str, float]] = []
    for lam_s, w_map in probs_grid.items():
        if lam_s not in lock_grid:
            continue
        for w_s, probs in w_map.items():
            if w_s not in lock_grid[lam_s]:
                continue
            E = energy_from_probs(probs)
            lr = float(lock_grid[lam_s][w_s])
            rows.append({
                "lambda": float(lam_s),
                "w": float(w_s),
                "E": float(E),
                "lock_rate": float(lr)
            })

    rows.sort(key=lambda r: (r["lambda"], r["w"]))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["lambda", "w", "E", "lock_rate"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    boundary = best_boundary_by_energy_drop(rows)

    summary = {
        "definition": "E(w,lambda) = -E[ log p(x* | w, lambda) ] computed as mean(-log p_i) over examples",
        "boundary_search": "sorted by lock_rate; pick split that maximizes energy drop EL-ER",
        "boundary": boundary,
        "n_points": len(rows)
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("WROTE:", args.out_csv)
    print("WROTE:", args.out_json)
    if boundary:
        print("BOUNDARY lock_rate≈", boundary["lock_rate_thr"])
        print("E_left (low lock_rate) =", boundary["E_left"])
        print("E_right (high lock_rate) =", boundary["E_right"])
        print("ENERGY DROP =", boundary["drop"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
