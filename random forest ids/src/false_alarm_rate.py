"""Measure the false-alarm rate where it can actually be measured.

The attack traces are three to four minutes of clean traffic each, so one alarm
in one of them reads as 19 an hour, and a rate estimated from a single event
over three minutes is not a rate. The only honest denominator is the long
attack-free capture: HCRL's normal_run_data.txt, which has no injected frames
at all, so every alarm it produces is false by construction.

That is what decides the alarm threshold. A lower threshold is worth real
latency -- threshold 4 detects the slowest flood 15 ms sooner than 8 -- and the
only reason not to take it is the false alarms it costs. This prices them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_windows import alarm_intervals, merge_episodes         # noqa: E402
from features import FEATURE_SETS                                # noqa: E402
from select_model import predict_tables                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/mixed/model.json")
    ap.add_argument("--clean", default="results/evcache/clean.npz")
    ap.add_argument("--thresholds", default="1,2,3,4,5,6,8,10,12,16")
    ap.add_argument("--out", default="results/false_alarm_rate.json")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    cols = FEATURE_SETS[model["feature_set"]]

    z = np.load(args.clean, allow_pickle=True)
    columns = list(z["columns"])
    X = z["X"][:, [columns.index(c) for c in cols]]
    t, cid = z["t"], z["can_id"]
    assert int(z["y"].sum()) == 0, "this capture is supposed to be attack free"

    pred = predict_tables(model, X)
    span = float(t[-1] - t[0])
    hours = span / 3600.0

    print(f"clean capture  {len(t)} frames over {span:.1f} s "
          f"({hours * 60:.1f} min), {len(set(cid.tolist()))} IDs")
    print(f"model          {model['n_trees']} trees, "
          f"{model['cost']['internal_nodes']} comparators")
    print(f"per frame      {int(pred.sum())} frames flagged "
          f"({100.0 * pred.mean():.4f} %)\n")

    hdr = (f"{'threshold':>10}{'alarm events':>14}{'per hour':>11}"
           f"{'mean between':>15}{'IDs involved':>14}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for m in [float(v) for v in args.thresholds.split(",")]:
        ev = merge_episodes(alarm_intervals(t, cid, pred, m))
        n = len(ev)
        ids = sorted({c for _, _, c in ev})
        per_h = n / hours
        mtbf = (span / n / 60.0) if n else float("inf")
        rows.append({"m": m, "events": n, "per_hour": per_h,
                     "minutes_between": mtbf,
                     "ids": [f"0x{c:03x}" for c in ids],
                     "seconds_alarming": sum(b - a for a, b, _ in ev)})
        print(f"{m:>10.0f}{n:>14}{per_h:>11.2f}"
              f"{(f'{mtbf:.1f} min' if n else 'never'):>15}"
              f"{len(ids):>14}")

    print()
    for r in rows:
        if r["events"] and r["events"] <= 6:
            print(f"  threshold {r['m']:.0f}: {r['events']} event(s) on "
                  f"{', '.join(r['ids'])}, "
                  f"{r['seconds_alarming'] * 1000:.0f} ms alarming in total")

    zero = [r for r in rows if r["events"] == 0]
    if zero:
        best = min(zero, key=lambda r: r["m"])
        print(f"\nlowest threshold with no false alarm at all in "
              f"{hours * 60:.0f} minutes of clean traffic: {best['m']:.0f}")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
