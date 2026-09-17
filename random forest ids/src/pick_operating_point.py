"""Choose the forest size and the alarm threshold together, across seeds.

They cannot be chosen separately. A low threshold buys real latency: the
deployed model detects the slowest flood in 36 ms at threshold 2 against 60 ms
at threshold 8. But that same model is clean at threshold 2 only because of its
random seed -- retrain the identical shape on nine other seeds and most of them
raise genuine false alarms there. So "threshold 2 works" is a statement about
one trained instance, not about the design, and shipping it would be shipping
luck.

The operating point is therefore a pair, and it is only usable if it holds for
every seed. This sweeps both, retraining each shape ten times, and keeps a
(trees, threshold) pair only when all ten instances catch every attack window
with no genuine false alarm. Post-attack ringing is excluded, because it is the
rate buckets draining after an attack the detector correctly caught, and the
attack-free capture is checked separately as the control.

Among the pairs that survive, the one with the lowest worst-case detection
latency wins, and comparator count breaks ties.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_windows import (GRACE_S, alarm_intervals, classify,     # noqa: E402
                          merge_episodes)
from features import FEATURE_SETS                                 # noqa: E402
from search_model import build, load_traces                        # noqa: E402
from select_model import predict_tables                            # noqa: E402
from sweep import majority_predict                                 # noqa: E402


def score(tr, pred, m):
    ev = merge_episodes(alarm_intervals(tr["t"], tr["can_id"], pred, m))
    inside, ring, false = classify(ev, tr["wins"])
    hit, lat = 0, []
    for ws, we in tr["wins"]:
        over = [(a, b) for a, b, _ in inside if b >= ws and a <= we + GRACE_S]
        if over:
            hit += 1
            lat.append(max(0.0, min(a for a, _ in over) - ws) * 1000.0)
    return (hit, len(tr["wins"]), len(false), len(ring),
            max(lat) if lat else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/mixed/cache.npz")
    ap.add_argument("--evcache", default="results/evcache")
    ap.add_argument("--set", default="no_payload")
    ap.add_argument("--trees", default="1,3,5,7,9")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--thresholds", default="2,3,4,6,8,12")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/operating_point.json")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    columns = list(z["columns"])
    cols = FEATURE_SETS[args.set]
    idx = [columns.index(c) for c in cols]
    Xtr, ytr = z["Xtr"][:, idx], z["ytr"]
    Xte = z["Xte"][:, idx]

    traces = load_traces(args.evcache)
    total = sum(len(t["wins"]) for t in traces)
    print(f"  total {total} attack windows\n")

    cz = np.load(os.path.join(args.evcache, "clean.npz"), allow_pickle=True)
    cc = list(cz["columns"])
    Xcl = cz["X"][:, [cc.index(c) for c in cols]]
    clean_h = float(cz["t"][-1] - cz["t"][0]) / 3600.0
    print(f"control: {len(cz['t'])} attack-free frames "
          f"({clean_h * 60:.1f} min)\n")

    ms = [float(v) for v in args.thresholds.split(",")]
    treelist = [int(v) for v in args.trees.split(",")]

    grid = {}
    for trees in treelist:
        preds, cpreds, cmps = [], [], []
        for seed in range(args.seeds):
            clf, model, n = build(Xtr, ytr, cols, trees, args.depth, None, seed)
            assert (majority_predict(clf, Xte)
                    == predict_tables(model, Xte)).all(), \
                f"t={trees} s={seed} integer tables disagree with the voter"
            cmps.append(n)
            preds.append([predict_tables(model, tr["X"][:, [
                tr["columns"].index(c) for c in cols]]) for tr in traces])
            cpreds.append(predict_tables(model, Xcl))

        for m in ms:
            worst_all, fa_all, det_all, ring_all, cl_all = 0.0, 0, [], 0, 0
            for s in range(args.seeds):
                det = fa = ring = 0
                worst = 0.0
                for tr, pr in zip(traces, preds[s]):
                    h, w, f, r, wo = score(tr, pr, m)
                    det += h
                    fa += f
                    ring += r
                    worst = max(worst, wo)
                det_all.append(det)
                fa_all += fa
                ring_all += ring
                worst_all = max(worst_all, worst)
                cl_all += len(merge_episodes(
                    alarm_intervals(cz["t"], cz["can_id"], cpreds[s], m)))
            clean_seeds = sum(1 for d in det_all if d == total)
            grid[(trees, m)] = {
                "trees": trees, "m": m, "seeds": args.seeds,
                "comparators_min": min(cmps), "comparators_max": max(cmps),
                "detected_min": min(det_all), "windows": total,
                "all_windows_every_seed": clean_seeds == args.seeds,
                "false_alarms_total": fa_all, "ringing_total": ring_all,
                "clean_capture_events": cl_all,
                "worst_ms": worst_all,
            }
            ok = (clean_seeds == args.seeds and fa_all == 0 and cl_all == 0)
            print(f"{trees} tree{'s' if trees != 1 else ' '} "
                  f"threshold {m:>4.0f}  cmp {min(cmps):2d}-{max(cmps):2d}  "
                  f"windows {min(det_all):3d}-{max(det_all):3d}/{total}  "
                  f"false alarms {fa_all:4d}  ringing {ring_all:3d}  "
                  f"clean-capture {cl_all:3d}  worst {worst_all:6.1f} ms"
                  f"{'   <= USABLE' if ok else ''}")
        print()

    rows = list(grid.values())
    usable = [r for r in rows if r["all_windows_every_seed"]
              and r["false_alarms_total"] == 0
              and r["clean_capture_events"] == 0]
    print("=" * 72)
    if usable:
        usable.sort(key=lambda r: (r["worst_ms"], r["comparators_max"]))
        print("usable operating points, lowest worst-case latency first:")
        for r in usable[:8]:
            print(f"  {r['trees']} trees at threshold {r['m']:.0f}: "
                  f"{r['comparators_min']}-{r['comparators_max']} comparators, "
                  f"worst {r['worst_ms']:.1f} ms")
        b = usable[0]
        print(f"\nchosen: {b['trees']} trees, alarm threshold {b['m']:.0f}, "
              f"worst-case detection {b['worst_ms']:.1f} ms, clean on all "
              f"{b['seeds']} seeds")
    else:
        print("no (trees, threshold) pair held for every seed")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
