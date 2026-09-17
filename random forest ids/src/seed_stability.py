"""Measure how much of a result is the model and how much is the seed.

The search found single-tree forests that catch every attack window with zero
false alarms, which looks like the answer until you retrain the same shape with
a different random_state and get 136 false alarms. Detection is not the fragile
part; the false-alarm count is. So the useful question is not "what is the best
a shape can do" but "what does a shape do every time", and the only way to
answer it is to train it repeatedly.

This trains one shape across many seeds and reports the spread. A shape whose
zero comes back on one seed in three is luck and must not be shipped; a shape
whose zero comes back on every seed is a property of the shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_SETS                              # noqa: E402
from search_model import build, load_traces, score_trace       # noqa: E402
from select_model import predict_tables                        # noqa: E402
from sweep import majority_predict                             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/mixed/cache.npz")
    ap.add_argument("--evcache", default="results/evcache")
    ap.add_argument("--m", type=float, default=8.0)
    ap.add_argument("--set", default="no_payload")
    ap.add_argument("--trees", default="1,3,5,7,9")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-features", default="all", choices=["all", "sqrt"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/seed_stability.json")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    columns = list(z["columns"])
    cols = FEATURE_SETS[args.set]
    idx = [columns.index(c) for c in cols]
    Xtr, ytr = z["Xtr"][:, idx], z["ytr"]
    Xte, yte = z["Xte"][:, idx], z["yte"]
    mf = None if args.max_features == "all" else "sqrt"

    print(f"set={args.set}  depth={args.depth}  max_features={args.max_features}"
          f"  threshold={args.m:.0f}  {args.seeds} seeds each\n")
    traces = load_traces(args.evcache)
    print()

    rows = []
    for trees in [int(v) for v in args.trees.split(",")]:
        recs = []
        for seed in range(args.seeds):
            clf, model, n_cmp = build(Xtr, ytr, cols, trees, args.depth,
                                      mf, seed)
            ref, got = majority_predict(clf, Xte), predict_tables(model, Xte)
            assert (ref == got).all(), f"t={trees} s={seed} table mismatch"
            per = {t["tag"]: score_trace(t, model, args.m, cols) for t in traces}
            recs.append({
                "seed": seed, "comparators": n_cmp,
                "detected": sum(p["detected"] for p in per.values()),
                "windows": sum(p["windows"] for p in per.values()),
                "false_alarms": sum(p["false_alarms"] for p in per.values()),
                "worst_ms": max((p["worst_ms"] for p in per.values()
                                 if p["worst_ms"] == p["worst_ms"]), default=0.0),
            })

        fa = [r["false_alarms"] for r in recs]
        det = [r["detected"] for r in recs]
        cmps = [r["comparators"] for r in recs]
        clean = sum(1 for r in recs
                    if r["false_alarms"] == 0 and r["detected"] == r["windows"])
        row = {
            "trees": trees, "seeds": len(recs),
            "clean_seeds": clean,
            "comparators_min": min(cmps), "comparators_max": max(cmps),
            "comparators_median": float(np.median(cmps)),
            "detected_min": min(det), "windows": recs[0]["windows"],
            "fa_min": min(fa), "fa_max": max(fa),
            "fa_median": float(np.median(fa)),
            "worst_ms": max(r["worst_ms"] for r in recs),
            "runs": recs,
        }
        rows.append(row)
        print(f"{trees} tree{'s' if trees != 1 else ' '}  "
              f"comparators {min(cmps):2d}-{max(cmps):2d}  "
              f"detected {min(det):3d}-{max(det):3d}/{recs[0]['windows']}  "
              f"false alarms {min(fa):4d}-{max(fa):4d} "
              f"(median {np.median(fa):6.1f})  "
              f"clean on {clean}/{len(recs)} seeds")

    print()
    good = [r for r in rows if r["clean_seeds"] == r["seeds"]]
    if good:
        b = min(good, key=lambda r: r["comparators_median"])
        print(f"smallest shape that is clean on EVERY seed: {b['trees']} trees, "
              f"{b['comparators_min']}-{b['comparators_max']} comparators, "
              f"worst detect {b['worst_ms']:.1f} ms")
    else:
        print("no shape was clean on every seed")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
