"""Score the detector the way an operator would: alarms, not frames.

Per-frame accuracy is the wrong final metric for this problem. The residual
error is the flooded ID's own frames arriving at the flood period, which are
genuinely ambiguous one frame at a time, and they are *scattered singles*. An
intrusion detection system does not act on one frame; it raises an alarm on an
ID, and a node exclusion system then acts on that ID.

So this adds the layer that turns verdicts into alarms and measures what
matters:

  * of the real attack windows, how many raised an alarm, and how fast
  * during attack-free traffic, how often an alarm fired anyway

The alarm rule is a leaky integrator per ID: a flagged frame adds one, the
score bleeds away with elapsed time, and the ID alarms while the score is at or
above M. That is a small counter and a timestamp per ID, and it is what kills
the scattered single false positives, because a stray flag decays before the
next one arrives.

It has to decay with *time*, not with frames of that ID. A shift register of the
last N verdicts looks equivalent and is not: an attacker's ID goes completely
silent between floods, so its register never gets a zero pushed in, the alarm
latches forever, and every window after the first is scored as already-alarming
rather than newly detected. That version reported 1 of 73 windows detected on a
trace where the model was perfect frame by frame, which is how the bug was
found. An ID that has been quiet longer than CLEAR_SILENT_S is cleared outright
for the same reason.
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
from prepare import TEST_SPAN                                 # noqa: E402
from select_model import predict_tables                       # noqa: E402
from syncan_data import load_syncan_csv                       # noqa: E402

WINDOW_GAP_S = 0.5      # injected frames this far apart start a new window
GRACE_S = 0.05          # an alarm this soon after a window ends still counts
DECAY_PER_S = 50.0      # score units bled off per second of elapsed time
CLEAR_SILENT_S = 0.2    # an ID quiet this long has its alarm cleared
SCORE_CAP = 64.0        # so a long flood cannot build an unclearable score


def attack_windows(t: np.ndarray, y: np.ndarray):
    """Group injected frames into (start, end) attack windows."""
    at = t[y == 1]
    if len(at) == 0:
        return []
    breaks = np.flatnonzero(np.diff(at) > WINDOW_GAP_S)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(at) - 1]])
    return [(float(at[s]), float(at[e])) for s, e in zip(starts, ends)]


def alarms(t, can_id, pred, m: float):
    """Per-ID leaky integrator. Returns the times at which an alarm starts.

    score bleeds at DECAY_PER_S per second and gains 1 per flagged frame, so a
    burst of flags crosses M while isolated flags never do. Alarms also clear
    when their ID falls silent, which is what stops an attacker's ID latching
    between floods.
    """
    score, last_t, firing = {}, {}, {}
    out = []
    t_l, c_l, p_l = t.tolist(), can_id.tolist(), pred.tolist()

    for i in range(len(t_l)):
        now = t_l[i]
        cid = c_l[i]

        # any firing ID that has gone quiet is cleared
        if firing:
            for fid in [k for k, v in firing.items() if v]:
                if now - last_t.get(fid, now) > CLEAR_SILENT_S:
                    firing[fid] = False
                    score[fid] = 0.0

        sc = score.get(cid, 0.0) - DECAY_PER_S * (now - last_t.get(cid, now))
        if sc < 0.0:
            sc = 0.0
        if p_l[i]:
            sc += 1.0
            if sc > SCORE_CAP:
                sc = SCORE_CAP
        score[cid] = sc
        last_t[cid] = now

        hot = sc >= m
        if hot and not firing.get(cid):
            out.append((now, cid))
        firing[cid] = hot

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--format", default="hcrl", choices=["hcrl", "syncan"])
    ap.add_argument("--m", default="1,2,3,4,6,8",
                    help="comma-separated alarm thresholds to sweep")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    baseline = load_baseline(args.baseline)
    cols = FEATURE_SETS[model["feature_set"]]

    loader = load_hcrl_csv if args.format == "hcrl" else load_syncan_csv
    te = truncate(loader(args.csv), *TEST_SPAN)
    X = extract(te, baseline)[cols].to_numpy(np.int32)
    pred = predict_tables(model, X)

    t = te.timestamp.to_numpy()
    y = te.label.to_numpy()
    cid = te.can_id.to_numpy()

    wins = attack_windows(t, y)
    span = float(t[-1] - t[0])
    attack_time = sum(e - s for s, e in wins)
    clean_time = max(span - attack_time, 1e-9)

    print(f"trace      {os.path.basename(args.csv)}")
    print(f"model      {model['n_trees']} trees, "
          f"{model['cost']['internal_nodes']} comparators")
    print(f"test slice {len(te)} frames over {span:.1f} s")
    print(f"attacks    {len(wins)} windows, {attack_time:.1f} s of attack, "
          f"{clean_time:.1f} s clean")
    print(f"per frame  {int(((pred == 1) & (y == 0)).sum())} false positives, "
          f"{int(((pred == 0) & (y == 1)).sum())} missed")
    print()

    hdr = (f"{'threshold':>10}{'windows detected':>19}{'median detect':>15}"
           f"{'worst detect':>14}{'false alarms':>14}{'per hour':>10}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for m in [int(v) for v in args.m.split(",")]:
        ev = alarms(t, cid, pred, float(m))
        hit, lat = 0, []
        for s, e in wins:
            first = next((at for at, _ in ev if s <= at <= e + GRACE_S), None)
            if first is not None:
                hit += 1
                lat.append((first - s) * 1000.0)
        # an alarm outside every window, with the grace margin, is false
        fa = sum(1 for at, _ in ev
                 if not any(s - GRACE_S <= at <= e + GRACE_S for s, e in wins))
        med = float(np.median(lat)) if lat else float("nan")
        worst = float(np.max(lat)) if lat else float("nan")
        per_h = fa / (clean_time / 3600.0)
        print(f"{m:>10}{hit:>10} / {len(wins):<7}"
              f"{med:>12.1f} ms{worst:>11.1f} ms{fa:>14}{per_h:>10.2f}")
        rows.append({"m": m, "windows": len(wins),
                     "detected": hit, "median_ms": med, "worst_ms": worst,
                     "false_alarms": fa, "per_hour": per_h})

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
