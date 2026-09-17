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
# For the period-scaled variant in alarm_period.py: how many of an ID's own
# nominal periods of silence clear its alarm. 0.2 s is 20 periods for a 10 ms
# ID and a fifth of one for a 1 s ID, which is the same absolute-threshold bug
# the features had.
CLEAR_SILENT_PERIODS_DEFAULT = 20.0
SCORE_CAP = 64.0        # so a long flood cannot build an unclearable score
REARM_S = 0.25          # gap below which two alarms on one ID are one event
# Measured, not chosen: after a flood ends, bus_rate takes 1.4 to 4.6 s to
# fall back below the 63 that a model trained on it roots on, so anything
# inside 5 s of a window is the bucket draining rather than clean traffic.
RING_S = 5.0
# ...and in periods, which is the form that is actually right. A rate bucket
# leaks once per frame of its ID, so its time constant is a number of that
# ID's periods, not a number of seconds. Classifying the drain in seconds
# undercounts it for slow IDs by the ratio of their periods: with a 5 s
# window, 35 of 36 alarms called "false" turned out to sit within 10 of the
# affected ID's own periods of a window ending, and the 36th within 50.
# Every absolute threshold in this project has been wrong for the same
# reason. The attack-free control is what the false-alarm claim actually
# rests on, because there no classification choice can flatter it: with no
# attack anywhere in the capture, every alarm is false by construction.
RING_PERIODS = 20.0


def attack_windows(t: np.ndarray, y: np.ndarray):
    """Group injected frames into (start, end) attack windows."""
    at = t[y == 1]
    if len(at) == 0:
        return []
    breaks = np.flatnonzero(np.diff(at) > WINDOW_GAP_S)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(at) - 1]])
    return [(float(at[s]), float(at[e])) for s, e in zip(starts, ends)]


def _period_table(can_id, baseline):
    """Per-frame nominal period for its ID, 0 where the ID has no baseline."""
    size = max(2048, int(can_id.max()) + 1)
    table = np.zeros(size, dtype=np.int64)
    for c, us in baseline.mean_interval.items():
        if 0 <= int(c) < size:
            table[int(c)] = int(us)
    return table[can_id]


def _bus_ratio_q6(dt_bus, baseline):
    """Elapsed time in 64ths of the BUS's nominal interval.

    Always defined, unlike the per-ID ratio, because the bus has a nominal
    frame interval even when the ID carrying the frame is brand new.
    """
    from features import CAP_RATIO, RECIP_SHIFT
    recip = (int(round((1 << RECIP_SHIFT) * 64.0 / baseline.bus_interval_us))
             if baseline.bus_interval_us > 0 else 0)
    return np.clip((dt_bus * recip) >> RECIP_SHIFT, 0, CAP_RATIO)


def classify(ev, wins, ring_s: float = RING_S, id_period_us=None,
             ring_periods: float = RING_PERIODS):
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
        # the drain window for this ID: whichever of the two is longer, since
        # a fast ID's buckets are still governed by the bus-level transient
        win = ring_s
        if id_period_us is not None:
            per = id_period_us.get(int(cid), 0) if hasattr(id_period_us, "get") \
                else 0
            if per > 0:
                win = max(win, ring_periods * per / 1e6)
        if any(b >= ws - GRACE_S and a <= we + GRACE_S for ws, we in wins):
            inside.append((a, b, cid))
        elif any(0 <= a - we <= win for _, we in wins):
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


# The decay, expressed the way everything else in this system had to be: in
# the ID's own periods rather than in seconds. A fixed 50/s means a flag is
# gone in 20 ms, so two flags 33 ms apart never combine, and every ID running
# at 100 ms or slower is undetectable at any rate an attacker would bother
# with. DECAY_PERIODS costs one score unit per that many of the ID's nominal
# periods instead, driven by dt_ratio_q6, which is already the elapsed gap in
# 64ths of that period and already computed for every frame: a subtract and a
# shift, no new state. Worst-case clear time is CAP_SCORED * DECAY_PERIODS of
# the ID's period, which is why the cap is small -- at 64 a 1 s ID that took
# 26 flags stayed alarming for 111 s and marked every window inside two
# minutes as detected.
DECAY_PERIODS = 2.0
# The cap is not a free constant: it has to exceed the alarm threshold or the
# score can never reach it, and every unit above the threshold is time the
# alarm cannot clear in. So it is derived from the threshold rather than set,
# and a fixed 3 silently reported 0 of 73 windows at threshold 4 before this
# was derived.
CAP_HEADROOM = 1.0


def alarm_intervals(t, can_id, pred, m: float, dt_ratio=None,
                    id_period_us=None, decay_periods: float = DECAY_PERIODS,
                    cap: float = 0.0, bus_ratio=None):
    """Per-ID leaky integrator. Returns (start, end, id) alarming intervals.

    A flagged frame adds 1 to that ID's score, the score bleeds away with
    elapsed time, and the ID alarms while the score is at or above M. So a
    burst of flags crosses M while isolated flags never do.

    With `dt_ratio` supplied the decay is scaled to the ID's own period, which
    is what the detector actually ships; without it the decay is the fixed
    DECAY_PER_S, kept so the two can be compared. `bus_ratio` covers the ID
    that has no baseline at all, whose dt_ratio_q6 is 0 on every frame -- 100 %
    of ID 0x000's -- and would otherwise never decay.

    Clear-on-silence stays in real time in both. It has to: an ID has gone
    quiet after a certain amount of elapsed time, and an unbaselined ID has no
    period to express that in. Scaling it by a ratio instead made 0x000, which
    appears only inside floods, never register as silent between them, and one
    alarm ran 215 s across four separate windows.
    """
    score, last_t, since = {}, {}, {}
    out = []
    t_l, c_l, p_l = t.tolist(), can_id.tolist(), pred.tolist()
    r_l = dt_ratio.tolist() if dt_ratio is not None else None
    b_l = bus_ratio.tolist() if bus_ratio is not None else None
    per_l = id_period_us.tolist() if id_period_us is not None else None
    step = 64.0 * decay_periods
    if cap <= 0.0:
        cap = m + CAP_HEADROOM
    assert cap > m, f"score cap {cap} cannot reach threshold {m}"
    end_t = t_l[-1] if t_l else 0.0

    for i in range(len(t_l)):
        now = t_l[i]
        cid = c_l[i]

        # a firing ID that has gone quiet is cleared, and its interval closed
        quiet = CLEAR_SILENT_S
        if per_l is not None and per_l[i] > 0:
            quiet = max(quiet, CLEAR_SILENT_PERIODS_DEFAULT * per_l[i] / 1e6)
        for fid in [k for k in since if since[k] is not None]:
            if now - last_t.get(fid, now) > quiet:
                out.append((since[fid], now, fid))
                since[fid] = None
                score[fid] = 0.0

        if r_l is None:
            sc = score.get(cid, 0.0) - DECAY_PER_S * (now - last_t.get(cid, now))
            ceiling = SCORE_CAP
        else:
            ratio = r_l[i]
            if ratio == 0 and b_l is not None:
                ratio = b_l[i]
            sc = score.get(cid, 0.0) - ratio / step
            ceiling = cap
        if sc < 0.0:
            sc = 0.0
        if p_l[i]:
            sc += 1.0
            if sc > ceiling:
                sc = ceiling
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
    feats = extract(te, baseline)
    X = feats[cols].to_numpy(np.int32)
    pred = predict_tables(model, X)

    # the period-scaled decay needs the elapsed gap in 64ths of the ID's own
    # period, which the extractor already produced, plus the bus-level ratio
    # for any ID with no baseline
    dt_ratio = feats["dt_ratio_q6"].to_numpy(np.int32)
    periods = _period_table(te.can_id.to_numpy(np.int64), baseline)
    bratio = _bus_ratio_q6(feats["dt_bus"].to_numpy(np.int64), baseline)

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
        ev = merge_episodes(alarm_intervals(
            t, cid, pred, float(m), dt_ratio=dt_ratio,
            id_period_us=periods, bus_ratio=bratio))
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
