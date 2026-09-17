"""Scale the alarm's decay to the ID's own period, the way the features are.

The model was fixed by removing every feature that encoded an absolute
microsecond threshold, because a threshold learned on a 10 ms ID means nothing
on a 100 ms one. The alarm layer above it still has exactly that bug.

DECAY_PER_S is 50, so a flagged frame's contribution is gone in 20 ms. Two
flags 33 ms apart therefore never combine, whatever the model thought of them
individually. That is the binding constraint on the four bursts still missed
after the model was fixed: 0x0a1 at 3x and 0x690 at 3x inject every 33 ms,
0x5a0 at 16x every 62 ms, 0x5a2 at 5x every 200 ms. Every one of those IDs runs
at 100 ms or slower, so a flood fast enough to matter for them is still far too
slow for a 20 ms integrator. No model can fix this; the integrator throws the
evidence away before it can be added up.

So the decay becomes one score unit per N of the ID's own nominal periods. A
10 ms ID keeps roughly the behaviour it has now (N=2 is 20 ms, exactly the
current constant); a 1 s ID gets 2 s to accumulate two flags.

It needs no new state and no new arithmetic. dt_ratio_q6 is already the elapsed
gap expressed in 64ths of that ID's period, computed for every frame to feed
the model, so the decay is that value shifted right -- a subtract and a shift
against the stored reciprocal multiply that already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_windows import (CLEAR_SILENT_PERIODS_DEFAULT,           # noqa: E402
                          CLEAR_SILENT_S, GRACE_S, SCORE_CAP,
                          alarm_intervals, classify, merge_episodes)
from features import FEATURE_SETS                                 # noqa: E402
from search_model import load_traces                              # noqa: E402
from select_model import predict_tables                           # noqa: E402


def alarm_intervals_period(t, can_id, pred, dt_ratio, m: float,
                           periods: float = 2.0,
                           clear_periods: float = CLEAR_SILENT_PERIODS_DEFAULT,
                           cap: float = SCORE_CAP,
                           bus_ratio=None, id_period_us=None):
    """Leaky integrator whose decay is measured in the ID's own periods.

    dt_ratio[i] is dt_ratio_q6: 64 means the frame arrived exactly one nominal
    period after the previous frame of its ID. So dt_ratio/64 is the elapsed
    time in periods, and the score loses 1 per `periods` of them.

    Clear-on-silence cannot be scaled the same way, and trying to was a
    mistake worth recording. An ID has gone quiet after a certain amount of
    real time, and that question has to be asked in real time: the elapsed
    per-ID gap. Driving it from a ratio instead means an unbaselined ID falls
    back to the bus's elapsed time, and dt_bus measures the gap since *any*
    frame, which stays at a few hundred microseconds while the other 26 IDs
    keep talking. So ID 0x000, which appears only inside floods, never
    registered as silent between them and its alarm ran 215 s across four
    separate windows.

    It is therefore absolute, with a floor: an ID is cleared after
    CLEAR_SILENT_S of silence, or after clear_periods of its own nominal
    period, whichever is longer. The floor is what a 1 s ID needs, since 0.2 s
    of quiet is less than one of its periods and would clear it constantly.

    `cap` matters more here than it did with a fixed decay, and getting it
    wrong reintroduces latching. The score only ever needs enough headroom to
    cross `m`; everything above that is time the alarm cannot clear in. At
    cap 64 and 1 unit per 4 periods, a 1 s ID that took 26 flagged frames
    stayed alarming for 111 seconds afterwards, which then marked every window
    inside those two minutes as detected. Worst-case clear time is
    cap * periods of the ID's own period, so the cap has to be small.

    `bus_ratio` is not optional in practice, and leaving it out is the flaw
    that makes this whole rule unshippable. An ID with no baseline entry has no
    stored reciprocal, so its dt_ratio_q6 is 0 on every frame -- exactly 100 %
    of them for ID 0x000 in the real DoS capture. Scaling the decay by that
    subtracts nothing, so the score for an unbaselined ID only ever rises and
    the alarm latches for the whole trace: 1110 seconds of one 1111-second
    capture. The unbaselined ID is the case a flood detector most needs to
    handle, so the decay falls back to the bus's own elapsed-time ratio there,
    which always exists because the bus always has a nominal frame interval.
    In hardware it is the reciprocal multiply bus_rate already performs.
    """
    score, since, last_t = {}, {}, {}
    out = []
    c_l, p_l, r_l = can_id.tolist(), pred.tolist(), dt_ratio.tolist()
    b_l = bus_ratio.tolist() if bus_ratio is not None else None
    per_l = id_period_us.tolist() if id_period_us is not None else None
    t_l = t.tolist()
    end_t = t_l[-1] if t_l else 0.0
    step = 64.0 * periods

    for i in range(len(t_l)):
        now, cid, ratio = t_l[i], c_l[i], r_l[i]
        # an ID with no baseline has no period to scale by, so use the bus's
        if ratio == 0 and b_l is not None:
            ratio = b_l[i]

        # silence is real time, and an unbaselined ID has no period, so the
        # threshold is absolute with a floor of clear_periods of the ID's own
        # period where one is known
        quiet = CLEAR_SILENT_S
        if per_l is not None and per_l[i] > 0:
            quiet = max(quiet, clear_periods * per_l[i] / 1e6)
        for fid in [k for k in since if since[k] is not None]:
            if now - last_t.get(fid, now) > quiet:
                out.append((since[fid], now, fid))
                since[fid] = None
                score[fid] = 0.0
        last_t[cid] = now

        sc = score.get(cid, 0.0) - ratio / step
        if sc < 0.0:
            sc = 0.0
        if p_l[i]:
            sc += 1.0
            if sc > cap:
                sc = cap
        score[cid] = sc

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


def id_period_us(can_id, baseline):
    """The stored nominal period for each frame's ID, 0 where none is known.

    Same per-ID table the feature extractor already keeps for the reciprocal
    multiply, so it is not new state. An ID absent from it reads 0, which is
    exactly the "no baseline" case the caller has to handle.
    """
    size = max(2048, int(can_id.max()) + 1)
    table = np.zeros(size, dtype=np.int64)
    for cid, us in baseline.mean_interval.items():
        if 0 <= int(cid) < size:
            table[int(cid)] = int(us)
    return table[can_id.astype(np.int64)]


def bus_ratio_q6(dt_bus, baseline):
    """Elapsed time in 64ths of the bus's own nominal frame interval.

    Always defined, unlike the per-ID ratio, because the bus always has a
    nominal interval even when the ID carrying the frame is brand new. Same
    stored-reciprocal multiply the feature extractor already does.
    """
    from features import CAP_RATIO, RECIP_SHIFT
    recip = int(round((1 << RECIP_SHIFT) * 64.0 / baseline.bus_interval_us)) \
        if baseline.bus_interval_us > 0 else 0
    return np.clip((dt_bus.astype(np.int64) * recip) >> RECIP_SHIFT,
                   0, CAP_RATIO)


def run(tr, ev, m):
    """Score one trace, and measure latching rather than trusting it away.

    `stale` counts windows that opened while an alarm was already running.
    Those are scored as detected by the "alarm active during the window" rule,
    which is the right rule only as long as alarms actually clear. A stuck
    alarm satisfies it for free, so the count is reported next to the
    detections instead of being folded into them.
    """
    inside, ring, false = classify(ev, tr["wins"])
    hit, lat, stale = 0, [], 0
    longest = 0.0
    for a, b, _ in inside:
        longest = max(longest, b - a)
    for ws, we in tr["wins"]:
        over = [(a, b) for a, b, _ in inside if b >= ws and a <= we + GRACE_S]
        if over:
            hit += 1
            lat.append(max(0.0, min(a for a, _ in over) - ws) * 1000.0)
        if any(a < ws - 1e-9 and b >= ws for a, b, _ in inside):
            stale += 1
    return hit, len(tr["wins"]), len(false), len(ring), lat, stale, longest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/soak/model_perid.json")
    ap.add_argument("--evcache", default="results/evcache")
    ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--periods", default="1,2,4,8")
    ap.add_argument("--caps", default="",
                    help="score caps to try with the best period rule")
    ap.add_argument("--baseline", default="results/soak/baseline.json")
    ap.add_argument("--out", default="results/alarm_period.json")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    cols = FEATURE_SETS[model["feature_set"]]
    from cross_eval import load_baseline
    baseline = load_baseline(args.baseline)
    traces = load_traces(args.evcache)
    print()

    for tr in traces:
        idx = [tr["columns"].index(c) for c in cols]
        tr["pred"] = predict_tables(model, tr["X"][:, idx])
        tr["ratio"] = tr["X"][:, tr["columns"].index("dt_ratio_q6")]
        tr["bratio"] = bus_ratio_q6(
            tr["X"][:, tr["columns"].index("dt_bus")], baseline)
        tr["period"] = id_period_us(tr["can_id"], baseline)

    cz = np.load(os.path.join(args.evcache, "clean.npz"), allow_pickle=True)
    cc = list(cz["columns"])
    cpred = predict_tables(model, cz["X"][:, [cc.index(c) for c in cols]])
    cratio = cz["X"][:, cc.index("dt_ratio_q6")]
    cbratio = bus_ratio_q6(cz["X"][:, cc.index("dt_bus")], baseline)
    cperiod = id_period_us(cz["can_id"], baseline)
    clean_h = float(cz["t"][-1] - cz["t"][0]) / 3600.0

    print(f"model      {model['n_trees']} trees, "
          f"{model['cost']['internal_nodes']} comparators, "
          f"{model['feature_set']}")
    print(f"threshold  {args.m:.0f}\n")

    hdr = (f"{'decay rule':>26}{'cap':>5}{'windows':>12}{'median':>9}"
           f"{'worst':>10}{'false':>7}{'ring':>6}{'clean':>7}"
           f"{'stale':>7}{'longest alarm':>15}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    variants = [("fixed 50/s (current)", None, SCORE_CAP)]
    for p in args.periods.split(","):
        variants.append((f"1 unit per {p} periods", float(p), SCORE_CAP))
    for c in [v for v in args.caps.split(",") if v]:
        for p in args.periods.split(","):
            variants.append((f"1 unit per {p} periods", float(p), float(c)))

    for label, per, cap in variants:
        det = win = fa = ring = stale = 0
        lats = []
        longest = 0.0
        for tr in traces:
            if per is None:
                ev = alarm_intervals(tr["t"], tr["can_id"], tr["pred"], args.m)
            else:
                ev = alarm_intervals_period(
                    tr["t"], tr["can_id"], tr["pred"], tr["ratio"], args.m,
                    per, cap=cap, bus_ratio=tr["bratio"],
                    id_period_us=tr["period"])
            h, w, f, r, lat, st, lg = run(tr, merge_episodes(ev), args.m)
            det += h
            win += w
            fa += f
            ring += r
            stale += st
            longest = max(longest, lg)
            lats += lat
        if per is None:
            cev = alarm_intervals(cz["t"], cz["can_id"], cpred, args.m)
        else:
            cev = alarm_intervals_period(cz["t"], cz["can_id"], cpred, cratio,
                                         args.m, per, cap=cap,
                                         bus_ratio=cbratio,
                                         id_period_us=cperiod)
        ncl = len(merge_episodes(cev))
        rows.append({"rule": label, "periods": per, "cap": cap,
                     "detected": det, "windows": win, "false_alarms": fa,
                     "ringing": ring, "clean_events": ncl,
                     "stale_windows": stale, "longest_alarm_s": longest,
                     "clean_per_hour": ncl / clean_h,
                     "median_ms": float(np.median(lats)) if lats else None,
                     "worst_ms": float(np.max(lats)) if lats else None})
        print(f"{label:>26}{cap:>5.0f}{det:>6} / {win:<4}"
              f"{np.median(lats):>6.1f} ms{np.max(lats):>7.1f} ms"
              f"{fa:>7}{ring:>6}{ncl:>7}{stale:>7}{longest:>12.1f} s")

    print()
    # a rule only counts if its alarms clear: no more than a handful of windows
    # may open on an already-running alarm, and no alarm may outlast a window
    # by an order of magnitude
    ok = [r for r in rows if r["false_alarms"] == 0 and r["clean_events"] == 0
          and r["stale_windows"] <= 0.1 * r["windows"]
          and r["longest_alarm_s"] <= 30.0]
    if ok:
        best = max(ok, key=lambda r: (r["detected"], -r["worst_ms"]))
        print(f"best rule that detects without latching: {best['rule']}, "
              f"cap {best['cap']:.0f} -- {best['detected']}/{best['windows']} "
              f"windows, worst {best['worst_ms']:.1f} ms, longest alarm "
              f"{best['longest_alarm_s']:.1f} s, {best['stale_windows']} "
              f"window(s) opening on a running alarm")
        base = rows[0]
        d = best["detected"] - base["detected"]
        if d:
            print(f"  {d:+d} window{'' if abs(d) == 1 else 's'} against the "
                  f"fixed decay, at no cost in false alarms")
    else:
        print("no rule both detected everything and cleared its alarms:")
        for r in sorted(rows, key=lambda r: -r["detected"])[:4]:
            why = []
            if r["false_alarms"]:
                why.append(f"{r['false_alarms']} false alarms")
            if r["clean_events"]:
                why.append(f"{r['clean_events']} on the clean capture")
            if r["stale_windows"] > 0.1 * r["windows"]:
                why.append(f"{r['stale_windows']} windows opened on a "
                           f"running alarm")
            if r["longest_alarm_s"] > 30.0:
                why.append(f"one alarm ran {r['longest_alarm_s']:.0f} s")
            print(f"  {r['rule']} cap {r['cap']:.0f}: "
                  f"{r['detected']}/{r['windows']}, " + "; ".join(why))

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
