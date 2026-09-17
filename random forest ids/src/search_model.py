"""Search model shapes against the metric that matters, not per-frame accuracy.

The sweep ranks candidates by per-frame test accuracy. That is the wrong
objective for the final choice, and the project has already shown why twice: a
model with 8000 per-frame false positives raises zero false alarms once the
leaky integrator is in front of it, and a model with excellent per-frame
numbers on one trace was blind to a second victim ID and to every slower attack
rate. Per-frame accuracy is a proxy; alarms across all seven traces are the
objective.

So this trains a grid of forest shapes and scores each one the way the detector
is actually used: window detection and false alarms per hour, summed over every
evaluation trace, at the deployed alarm threshold. Comparator count is the cost.
What comes out is the Pareto front of (comparators, windows detected), which is
the answer to "how small can this get without losing anything".

Candidates are trained on one truncation and scored on a disjoint one, and the
integer node tables are asserted to reproduce the float forest's majority vote
exactly, same as select_model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from eval_windows import (GRACE_S, alarm_intervals,            # noqa: E402
                          attack_windows, classify, merge_episodes)
from features import FEATURE_SETS                              # noqa: E402
from select_model import (forest_cost, predict_tables,          # noqa: E402
                          prune_equivalent, tree_to_tables)
from sweep import majority_predict                              # noqa: E402

TRACES = [
    ("dos", "real HCRL DoS"),
    ("zid", "0x000 flood"),
    ("vid", "valid-ID flood"),
    ("s2c0", "stealth 0x2c0"),
    ("s316", "stealth 0x316"),
    ("lowrate", "2x low-rate flood"),
    ("mixed", "mixed-rate floods"),
    # An hour of ordinary traffic with rare short bursts across all 27 victim
    # IDs. This is the trace that matters most and the one the other six
    # flatter: at 1.8 % attack traffic there is enough clean driving for a
    # false-alarm rate to mean something, and rotating the victim exposes
    # models whose thresholds only fit a 10 ms ID.
    ("soak", "1 hour soak, rare bursts"),
]


def load_traces(cache_dir: str):
    """Load every evaluation trace and pre-compute its attack windows."""
    out = []
    for tag, label in TRACES:
        path = os.path.join(cache_dir, f"{tag}.npz")
        if not os.path.exists(path):
            print(f"  (missing {path}, skipped)")
            continue
        z = np.load(path, allow_pickle=True)
        t, y = z["t"], z["y"]
        wins = attack_windows(t, y)
        span = float(t[-1] - t[0])
        clean = max(span - sum(e - s for s, e in wins), 1e-9)
        out.append({
            "tag": tag, "label": label,
            "X": z["X"], "columns": list(z["columns"]),
            "t": t, "can_id": z["can_id"],
            "wins": wins, "clean_h": clean / 3600.0,
        })
        print(f"  {label:22s} {len(t):8d} frames  {len(wins):3d} windows")
    return out


def score_trace(tr, model, m: float, cols):
    """Window detection and false alarms for one candidate on one trace."""
    idx = [tr["columns"].index(c) for c in cols]
    pred = predict_tables(model, tr["X"][:, idx])
    ev = merge_episodes(alarm_intervals(tr["t"], tr["can_id"], pred, m))
    wins = tr["wins"]
    inside, ring, false = classify(ev, wins)

    hit, lat = 0, []
    for ws, we in wins:
        over = [(a, b) for a, b, _ in inside if b >= ws and a <= we + GRACE_S]
        if over:
            hit += 1
            lat.append(max(0.0, min(a for a, _ in over) - ws) * 1000.0)
    fa = len(false)
    return {
        "windows": len(wins), "detected": hit, "false_alarms": fa,
        "ringing": len(ring), "per_hour": fa / tr["clean_h"],
        "worst_ms": float(np.max(lat)) if lat else float("nan"),
        "median_ms": float(np.median(lat)) if lat else float("nan"),
    }


def build(Xtr, ytr, cols, trees: int, depth: int, max_features, seed: int):
    """Train, freeze to integer tables, prune, and verify the tables."""
    clf = RandomForestClassifier(
        n_estimators=trees, max_depth=depth,
        max_features=max_features, min_samples_leaf=50,
        bootstrap=True, random_state=seed, n_jobs=-1,
    ).fit(Xtr, ytr)

    raw = [tree_to_tables(e, cols) for e in clf.estimators_]
    pruned = [prune_equivalent(t) for t in raw]
    model = {"feature_names": cols, "n_trees": trees, "trees": pruned,
             "cost": forest_cost(clf)}
    n_cmp = sum(1 for t in pruned for leaf in t["is_leaf"] if not leaf)
    model["cost"]["internal_nodes"] = n_cmp
    model["cost"]["max_depth"] = max(
        _depth(t) for t in pruned) if pruned else 0
    return clf, model, n_cmp


def _depth(tree, n: int = 0) -> int:
    if tree["is_leaf"][n]:
        return 0
    return 1 + max(_depth(tree, tree["left"][n]), _depth(tree, tree["right"][n]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/mixed/cache.npz")
    ap.add_argument("--evcache", default="results/evcache")
    ap.add_argument("--m", type=float, default=8.0)
    ap.add_argument("--sets", default="no_payload,timing")
    ap.add_argument("--trees", default="1,3,5")
    ap.add_argument("--depths", default="2,3,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/search.json")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    columns = list(z["columns"])

    print("evaluation traces:")
    traces = load_traces(args.evcache)
    total_windows = sum(len(x["wins"]) for x in traces)
    print(f"  total {total_windows} attack windows\n")

    results = []
    for sname in args.sets.split(","):
        cols = FEATURE_SETS[sname]
        idx = [columns.index(c) for c in cols]
        Xtr, ytr = z["Xtr"][:, idx], z["ytr"]
        Xte, yte = z["Xte"][:, idx], z["yte"]

        for trees in [int(v) for v in args.trees.split(",")]:
            for depth in [int(v) for v in args.depths.split(",")]:
                for mf in ("sqrt", None):
                    for seed in [int(v) for v in args.seeds.split(",")]:
                        clf, model, n_cmp = build(
                            Xtr, ytr, cols, trees, depth, mf, seed)
                        # the integer tables must match the float voter exactly
                        ref = majority_predict(clf, Xte)
                        got = predict_tables(model, Xte)
                        assert (ref == got).all(), (
                            f"{sname} t={trees} d={depth} mf={mf} s={seed}: "
                            f"{int((ref != got).sum())} table mismatches")

                        per = {tr["tag"]: score_trace(tr, model, args.m, cols)
                               for tr in traces}
                        det = sum(p["detected"] for p in per.values())
                        win = sum(p["windows"] for p in per.values())
                        fa = sum(p["false_alarms"] for p in per.values())
                        worst = max(
                            (p["worst_ms"] for p in per.values()
                             if p["worst_ms"] == p["worst_ms"]), default=0.0)
                        row = {
                            "set": sname, "trees": trees, "depth_cfg": depth,
                            "max_features": mf or "all", "seed": seed,
                            "comparators": n_cmp,
                            "real_depth": model["cost"]["max_depth"],
                            "detected": det, "windows": win,
                            "false_alarms": fa, "worst_ms": worst,
                            "per_trace": per,
                            "trees_tables": model["trees"],
                        }
                        results.append(row)
                        flag = "  <= ALL" if det == win and fa == 0 else ""
                        print(f"{sname:11s} t={trees} d={depth} "
                              f"mf={str(mf or 'all'):4s} s={seed}  "
                              f"cmp={n_cmp:3d}  {det:3d}/{win:3d}  "
                              f"FA={fa:3d}  worst={worst:6.1f} ms{flag}")

    # Pareto front: fewest comparators for each level of detection
    perfect = [r for r in results if r["detected"] == r["windows"]
               and r["false_alarms"] == 0]
    print("\n" + "=" * 70)
    if perfect:
        perfect.sort(key=lambda r: (r["comparators"], r["worst_ms"]))
        print("candidates detecting every window with zero false alarms, "
              "smallest first:")
        for r in perfect[:12]:
            print(f"  cmp={r['comparators']:3d}  depth={r['real_depth']}  "
                  f"{r['set']:11s} {r['trees']} trees  seed={r['seed']}  "
                  f"mf={r['max_features']:4s}  worst={r['worst_ms']:.1f} ms")
        b = perfect[0]
        print(f"\nsmallest: {b['comparators']} comparators, "
              f"{b['trees']} trees, depth {b['real_depth']}, {b['set']}")
    else:
        best = max(results, key=lambda r: (r["detected"], -r["false_alarms"]))
        print(f"nothing reached every window. best: {best['detected']}/"
              f"{best['windows']} with {best['comparators']} comparators")

    with open(args.out, "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "trees_tables"}
                   for r in results], fh, indent=1)
    print(f"\nwrote {args.out}  ({len(results)} candidates)")

    if perfect:
        b = perfect[0]
        best_model = {
            "feature_set": b["set"],
            "feature_names": FEATURE_SETS[b["set"]],
            "n_trees": b["trees"],
            "trees": b["trees_tables"],
            "cost": {"internal_nodes": b["comparators"],
                     "max_depth": b["real_depth"]},
            "search": {k: v for k, v in b.items()
                       if k not in ("trees_tables", "per_trace")},
            "per_trace": b["per_trace"],
        }
        path = os.path.join(os.path.dirname(args.out), "search_best.json")
        with open(path, "w") as fh:
            json.dump(best_model, fh, indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
