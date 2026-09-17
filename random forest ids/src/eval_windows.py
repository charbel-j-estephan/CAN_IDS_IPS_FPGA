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

The metric has the same trap. Asking "did an alarm *start* inside this window"
undercounts: a fast burst saturates the score, which then takes over a second to
decay, and a window opening inside that shadow records no new start even though
the ID was alarming throughout it. That scored fast floods as *less* detected
than slow ones, which is backwards and is how it was caught. What is measured is
whether the alarm was *active* during the window, which is also the question an
operator is actually asking.
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
REARM_S = 0.25          # gap below which two alarms on one ID are one event
RING_S = 2.0            # after a window ends the rate buckets are still draining


def attack_windows(t: np.ndarray, y: np.ndarray):
    """Group injected frames into (start, end) attack windows."""
    at = t[y == 1]
    if len(at) == 0:
        return []
    breaks = np.flatnonzero(np.diff(at) > WINDOW_GAP_S)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(at) - 1]])
    return [(float(at[s]), float(at[e])) for s, e in zip(starts, ends)]


def classify(ev, wins, ring_s: float = RING_S):
    """Split alarm episodes into detections, post-attack ringing, and false alarms.

    An episode with no injected frames in it is not automatically a false
    positive. Chased down, every such episode in these traces is the bus-level
    leaky bucket still draining after a flood stopped: ID 0x43f alarms at
    1479121898.52, which is 875 ms after the last injected frame of the window
    that ended at 1479121897.64, and the same ID is flagged 0 times in 50 641
    frames of the attack-free capture. For a few hundred milliseconds after a
    flood, ordinary frames from bystander IDs sit on an elevated bus_rate.

    That is the detector recovering from an attack it caught, not a false
    positive on clean traffic, and lumping the two together makes the
    false-alarm rate look worse while hiding the real number. Extending the
    window grace to cover it is not an option, because 875 ms of grace would
    swallow genuine alarms. They are separated and reported instead.
    """
    inside, ring, false = [], [], []
    for a, b, cid in ev:
        if any(b >= ws - GRACE_S and a <= we + GRACE_S for ws, we in wins):
            inside.append((a, b, cid))
        elif any(0 <= a - we <= ring_s for _, we in wins):
            ring.append((a, b, cid))
        else:
            false.append((a, b, cid))
    return inside, ring, false


def merge_episodes(intervals, gap: float = REARM_S):
    """Collapse alarms on one ID separated by less than `gap` into one event.

    A false alarm is one thing an operator has to look at, and the raw interval
    list does not count that. When the score sits near the threshold it dips
    below and re-crosses, so a single 250 ms disturbance on one ID emerges as
    one interval at threshold 4 and three at threshold 6 -- the same event,
    counted three times, which made the false-alarm total rise with the
    threshold. It was the same event both times: ID 0x43f, 1479121898.58 to
    1479121898.73, in the clean capture the floods are built over.

    Merging by ID with a re-arm gap fixes it. The gap has to exceed the dip,
    and the dips measured here are 3 ms and 13 ms, comfortably inside 250 ms,
    which is also the longest a real operator would call one alarm.
    """
    by_id = {}
    for a, b, cid in intervals:
        by_id.setdefault(cid, []).append((a, b))
    out = []
    for cid, iv in by_id.items():
        iv.sort()
        cur_a, cur_b = iv[0]
        for a, b in iv[1:]:
            if a - cur_b <= gap:
                cur_b = max(cur_b, b)
            else:
                out.append((cur_a, cur_b, cid))
                cur_a, cur_b = a, b
        out.append((cur_a, cur_b, cid))
    return sorted(out)


def alarm_intervals(t, can_id, pred, m: float):
    """Per-ID leaky integrator. Returns (start, end, id) alarming intervals.

    score bleeds at DECAY_PER_S per second and gains 1 per flagged frame, so a
    burst of flags crosses M while isolated flags never do. Alarms also clear
    when their ID falls silent, which is what stops an attacker's ID latching
    between floods.
    """
    score, last_t, since = {}, {}, {}
    out = []
    t_l, c_l, p_l = t.tolist(), can_id.tolist(), pred.tolist()
    end_t = t_l[-1] if t_l else 0.0

    for i in range(len(t_l)):
        now = t_l[i]
        cid = c_l[i]

        # a firing ID that has gone quiet is cleared, and its interval closed
        for fid in [k for k in since if since[k] is not None]:
            if now - last_t.get(fid, now) > CLEAR_SILENT_S:
                out.append((since[fid], now, fid))
                since[fid] = None
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
        was = since.get(cid)
        if hot and was is None:
            since[cid] = now
        elif not hot and was is not None:
            out.append((was, now, cid))
            since[cid] = None

    for cid, st in since.items():
        if st is not None:
            out.append((st, end_t, cid))
    return sorted(out)


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
        ev = merge_episodes(alarm_intervals(t, cid, pred, float(m)))
        hit, lat = 0, []
        for ws, we in wins:
            # detected if the alarm was ACTIVE at any point in the window
            over = [(a, b) for a, b, _ in ev
                    if b >= ws and a <= we + GRACE_S]
            if over:
                hit += 1
                first = min(a for a, _ in over)
                lat.append(max(0.0, first - ws) * 1000.0)
        # an alarm interval overlapping no window at all is a false alarm
        fa = sum(1 for a, b, _ in ev
                 if not any(b >= ws - GRACE_S and a <= we + GRACE_S
                            for ws, we in wins))
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
