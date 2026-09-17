"""Separate the detector's recovery transient from a real false alarm.

Both look the same in the alarm list: an episode on an ID with no injected
frames in it. They are not the same thing. Chased down, the only such episode
in these traces is ID 0x43f at 1479121898.52, and it starts 875 ms after the
last injected frame of the flood that ended at 1479121897.64. In the
attack-free capture the same ID is flagged 0 times in 50 641 frames.

The cause is the bus-level leaky bucket. bus_rate is still draining when the
flood stops, so for a few hundred milliseconds afterwards ordinary frames from
other IDs sit on an elevated bus-rate feature. That is the detector still
recovering from an attack it correctly caught, not a false positive on clean
traffic, and counting it as one makes the false-alarm rate look worse than it
is while hiding the real number, which is zero.

So episodes are classified by when they happen. What the window grace period
cannot do is cover this, because 875 ms of grace would swallow genuine alarms.
The classification is reported instead, and the deciding measurement stays the
attack-free capture, where a false alarm is false by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_windows import (GRACE_S, RING_S, alarm_intervals,      # noqa: E402
                          classify, merge_episodes)
from features import FEATURE_SETS                                  # noqa: E402
from search_model import load_traces                               # noqa: E402
from select_model import predict_tables                            # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/mixed/model.json")
    ap.add_argument("--evcache", default="results/evcache")
    ap.add_argument("--thresholds", default="2,3,4,5,6,8,10,12")
    ap.add_argument("--out", default="results/ringing.json")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    cols = FEATURE_SETS[model["feature_set"]]

    traces = load_traces(args.evcache)
    print()
    for tr in traces:
        idx = [tr["columns"].index(c) for c in cols]
        tr["pred"] = predict_tables(model, tr["X"][:, idx])

    z = np.load(os.path.join(args.evcache, "clean.npz"), allow_pickle=True)
    cc = list(z["columns"])
    clean_pred = predict_tables(model, z["X"][:, [cc.index(c) for c in cols]])
    clean_t, clean_id = z["t"], z["can_id"]
    clean_h = float(clean_t[-1] - clean_t[0]) / 3600.0

    hdr = (f"{'threshold':>10}{'windows':>13}{'worst detect':>14}"
           f"{'ringing':>10}{'false alarms':>14}{'clean capture':>15}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for m in [float(v) for v in args.thresholds.split(",")]:
        det = win = nring = nfalse = 0
        worst = 0.0
        per = {}
        for tr in traces:
            ev = merge_episodes(
                alarm_intervals(tr["t"], tr["can_id"], tr["pred"], m))
            inside, ring, false = classify(ev, tr["wins"])
            hit, lat = 0, []
            for ws, we in tr["wins"]:
                over = [(a, b) for a, b, _ in inside
                        if b >= ws and a <= we + GRACE_S]
                if over:
                    hit += 1
                    lat.append(max(0.0, min(a for a, _ in over) - ws) * 1000.0)
            det += hit
            win += len(tr["wins"])
            nring += len(ring)
            nfalse += len(false)
            if lat:
                worst = max(worst, max(lat))
            per[tr["tag"]] = {
                "windows": len(tr["wins"]), "detected": hit,
                "ringing": len(ring), "false_alarms": len(false),
                "worst_ms": max(lat) if lat else None,
            }

        cl = merge_episodes(alarm_intervals(clean_t, clean_id, clean_pred, m))
        rows.append({"m": m, "detected": det, "windows": win,
                     "worst_ms": worst, "ringing": nring,
                     "false_alarms": nfalse, "clean_events": len(cl),
                     "clean_per_hour": len(cl) / clean_h, "per_trace": per})
        print(f"{m:>10.0f}{det:>7} / {win:<4}{worst:>11.1f} ms"
              f"{nring:>10}{nfalse:>14}{len(cl):>15}")

    print()
    ok = [r for r in rows
          if r["detected"] == r["windows"] and r["false_alarms"] == 0
          and r["clean_events"] == 0]
    if ok:
        best = min(ok, key=lambda r: r["worst_ms"])
        print("every window, no false alarm, nothing on the clean capture:")
        for r in sorted(ok, key=lambda r: r["worst_ms"]):
            print(f"  threshold {r['m']:>3.0f}  worst {r['worst_ms']:6.1f} ms"
                  f"  ({r['ringing']} post-attack ringing episode"
                  f"{'' if r['ringing'] == 1 else 's'})")
        print(f"\nlowest worst-case latency: threshold {best['m']:.0f} at "
              f"{best['worst_ms']:.1f} ms, against "
              f"{max(r['worst_ms'] for r in ok):.1f} ms at the most "
              f"conservative setting")
    else:
        print("no threshold was clean on all three counts")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
