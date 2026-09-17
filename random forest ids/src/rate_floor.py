"""Detection rate against attack rate: where does the detector stop seeing it?

A flood's rate is the attacker's free parameter, and it is the parameter the
detector is most sensitive to. Reporting one detection number for a trace hides
that: a trace of mixed-rate bursts can score well overall while missing every
quiet burst in it.

So this measures each attack window separately, labels it with its own injection
rate as a multiple of the victim ID's normal rate, and reports detection per
rate band. The output is the honest answer to "how quiet can an attacker be and
still be caught".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from can_data import load_hcrl_csv, truncate                  # noqa: E402
from features import FEATURE_SETS, extract                    # noqa: E402
from cross_eval import load_baseline                          # noqa: E402
from eval_windows import alarm_intervals, attack_windows, GRACE_S  # noqa: E402
from prepare import TEST_SPAN                                 # noqa: E402
from select_model import predict_tables                       # noqa: E402

BANDS = [(0, 3), (3, 6), (6, 12), (12, 24), (24, 1e9)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--m", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    baseline = load_baseline(args.baseline)
    cols = FEATURE_SETS[model["feature_set"]]

    te = truncate(load_hcrl_csv(args.csv), *TEST_SPAN)
    X = extract(te, baseline)[cols].to_numpy(np.int32)
    pred = predict_tables(model, X)

    t = te.timestamp.to_numpy()
    y = te.label.to_numpy()
    cid = te.can_id.to_numpy()

    wins = attack_windows(t, y)
    ev = alarm_intervals(t, cid, pred, args.m)

    # each window's own rate: its injection period against the victim's period
    rows = []
    for s, e in wins:
        sel = (t >= s) & (t <= e) & (y == 1)
        if sel.sum() < 4:
            continue
        inj_period = float(np.median(np.diff(t[sel])))
        vid = int(np.bincount(cid[sel]).argmax())
        vsel = (cid == vid) & (y == 0)
        if vsel.sum() < 10:
            continue
        victim_period = float(np.median(np.diff(t[vsel])))
        rate = victim_period / inj_period if inj_period > 0 else np.inf
        over = [(a, b) for a, b, _ in ev if b >= s and a <= e + GRACE_S]
        hit = bool(over)
        lat = (max(0.0, min(a for a, _ in over) - s) * 1000.0
               if over else float("nan"))
        rows.append({"rate": rate, "detected": hit, "latency_ms": lat,
                     "victim": vid, "seconds": e - s})

    print(f"model      {model['n_trees']} trees, "
          f"{model['cost']['internal_nodes']} comparators")
    print(f"trace      {os.path.basename(args.csv)}")
    print(f"windows    {len(rows)} with a measurable rate, alarm threshold "
          f"{args.m:g}")
    print()
    hdr = (f"{'attack rate':>16}{'windows':>10}{'detected':>11}"
           f"{'median detect':>16}")
    print(hdr)
    print("-" * len(hdr))
    for lo, hi in BANDS:
        g = [r for r in rows if lo <= r["rate"] < hi]
        if not g:
            continue
        det = [r for r in g if r["detected"]]
        lats = [r["latency_ms"] for r in det if np.isfinite(r["latency_ms"])]
        label = f"{lo:g}x to {hi:g}x" if hi < 1e9 else f"above {lo:g}x"
        med = f"{np.median(lats):.1f} ms" if lats else "--"
        print(f"{label:>16}{len(g):>10}{len(det):>6} / {len(g):<4}{med:>16}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
