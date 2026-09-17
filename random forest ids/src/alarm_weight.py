"""Does weighting the alarm by the forest's vote margin buy anything?

The alarm layer adds exactly 1 to an ID's score per flagged frame, whatever the
forest thought. But the vote count already exists in hardware: the frozen
forest produces 0..N votes and the verdict is the comparison against N/2. A
frame where all five trees agree is a stronger signal than one where three of
five do, and throwing that away is a choice, not a given.

So the increment becomes the vote margin, votes - floor(N/2), which is 1 for a
bare majority and (N+1)/2 for unanimity. Costs one small adder and the same
counter width. If confident floods then alarm sooner without costing false
alarms, it is free latency. If it does not, the flat +1 stays and this file is
the measurement that says so.

Nothing else changes: same trees, same thresholds, same decay, same
clear-on-silence. Only the increment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_windows import (CLEAR_SILENT_S, DECAY_PER_S, GRACE_S,  # noqa: E402
                          REARM_S, SCORE_CAP, attack_windows,
                          merge_episodes)
from features import FEATURE_SETS                                # noqa: E402
from search_model import TRACES, load_traces                     # noqa: E402


def predict_votes(model, X) -> np.ndarray:
    """Vote count per frame, 0..n_trees, before the majority comparison."""
    votes = np.zeros(len(X), dtype=np.int32)
    for tree in model["trees"]:
        f = np.asarray(tree["feature"])
        thr = np.asarray(tree["threshold"])
        lf, rt = np.asarray(tree["left"]), np.asarray(tree["right"])
        leaf = np.asarray(tree["is_leaf"], dtype=bool)
        val = np.asarray(tree["value"])
        node = np.zeros(len(X), dtype=np.int32)
        active = ~leaf[node]
        while active.any():
            idx = np.flatnonzero(active)
            nd = node[idx]
            node[idx] = np.where(X[idx, f[nd]] <= thr[nd], lf[nd], rt[nd])
            active = ~leaf[node]
        votes += val[node]
    return votes


def alarm_intervals_w(t, can_id, inc, m: float):
    """Same leaky integrator, but the per-frame increment is passed in.

    inc[i] is 0 for a frame the forest did not flag and the weight otherwise,
    so passing an array of 0/1 reproduces the original behaviour exactly.
    """
    score, last_t, since = {}, {}, {}
    out = []
    t_l, c_l, w_l = t.tolist(), can_id.tolist(), inc.tolist()
    end_t = t_l[-1] if t_l else 0.0

    for i in range(len(t_l)):
        now, cid = t_l[i], c_l[i]
        for fid in [k for k in since if since[k] is not None]:
            if now - last_t.get(fid, now) > CLEAR_SILENT_S:
                out.append((since[fid], now, fid))
                since[fid] = None
                score[fid] = 0.0

        sc = score.get(cid, 0.0) - DECAY_PER_S * (now - last_t.get(cid, now))
        if sc < 0.0:
            sc = 0.0
        if w_l[i]:
            sc += w_l[i]
            if sc > SCORE_CAP:
                sc = SCORE_CAP
        score[cid] = sc
        last_t[cid] = now

        hot, was = sc >= m, since.get(cid)
        if hot and was is None:
            since[cid] = now
        elif not hot and was is not None:
            out.append((was, now, cid))
            since[cid] = None

    for cid, st in since.items():
        if st is not None:
            out.append((st, end_t, cid))
    return sorted(out)


def score_one(tr, inc, m):
    ev = merge_episodes(alarm_intervals_w(tr["t"], tr["can_id"], inc, m))
    wins = tr["wins"]
    hit, lat = 0, []
    for ws, we in wins:
        over = [(a, b) for a, b, _ in ev if b >= ws and a <= we + GRACE_S]
        if over:
            hit += 1
            lat.append(max(0.0, min(a for a, _ in over) - ws) * 1000.0)
    fa = sum(1 for a, b, _ in ev
             if not any(b >= ws - GRACE_S and a <= we + GRACE_S
                        for ws, we in wins))
    return {"windows": len(wins), "detected": hit, "false_alarms": fa,
            "median_ms": float(np.median(lat)) if lat else float("nan"),
            "worst_ms": float(np.max(lat)) if lat else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/mixed/model.json")
    ap.add_argument("--evcache", default="results/evcache")
    ap.add_argument("--thresholds", default="4,8,12,16,24")
    ap.add_argument("--out", default="results/alarm_weight.json")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    n = model["n_trees"]
    maj = n // 2 + 1                      # votes needed for an attack verdict
    cols = FEATURE_SETS[model["feature_set"]]

    traces = load_traces(args.evcache)
    print()

    # cache the vote counts once; only the increment rule changes below
    for tr in traces:
        idx = [tr["columns"].index(c) for c in cols]
        v = predict_votes(model, tr["X"][:, idx])
        tr["flag"] = (v >= maj).astype(np.int32)
        tr["margin"] = np.where(v >= maj, v - (maj - 1), 0).astype(np.int32)

    # the flat rule must reproduce the shipped numbers, or the rest means nothing
    modes = {"flat +1": "flag", "vote margin": "margin"}
    rows = []
    for label, key in modes.items():
        print(f"--- {label}  (increment 1..{n - maj + 1 if key == 'margin' else 1})")
        hdr = (f"{'threshold':>10}{'windows':>14}{'median':>11}"
               f"{'worst':>10}{'false alarms':>14}")
        print(hdr)
        print("-" * len(hdr))
        for m in [float(v) for v in args.thresholds.split(",")]:
            per = {tr["tag"]: score_one(tr, tr[key], m) for tr in traces}
            det = sum(p["detected"] for p in per.values())
            win = sum(p["windows"] for p in per.values())
            fa = sum(p["false_alarms"] for p in per.values())
            lat = [p["worst_ms"] for p in per.values()
                   if p["worst_ms"] == p["worst_ms"]]
            med = [p["median_ms"] for p in per.values()
                   if p["median_ms"] == p["median_ms"]]
            worst = max(lat, default=float("nan"))
            rows.append({"mode": label, "m": m, "detected": det,
                         "windows": win, "false_alarms": fa,
                         "worst_ms": worst,
                         "median_ms": float(np.median(med)) if med else None,
                         "per_trace": per})
            print(f"{m:>10.0f}{det:>8} / {win:<5}"
                  f"{np.median(med):>8.1f} ms{worst:>7.1f} ms{fa:>14}")
        print()

    clean = [r for r in rows if r["detected"] == r["windows"]
             and r["false_alarms"] == 0]
    print("=" * 66)
    if clean:
        best = min(clean, key=lambda r: r["worst_ms"])
        print("every window, zero false alarms, best worst-case latency:")
        for r in sorted(clean, key=lambda r: r["worst_ms"])[:6]:
            print(f"  {r['mode']:12s} threshold {r['m']:>4.0f}  "
                  f"worst {r['worst_ms']:6.1f} ms  median {r['median_ms']:5.1f} ms")
        print(f"\nbest: {best['mode']} at threshold {best['m']:.0f}, "
              f"worst {best['worst_ms']:.1f} ms")
    else:
        print("no setting caught every window with zero false alarms")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
