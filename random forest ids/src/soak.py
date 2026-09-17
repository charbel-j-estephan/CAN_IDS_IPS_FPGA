"""An hour of ordinary driving with rare attacks, and the worst attacks I can build.

Every result so far is measured on traces that are roughly a third attack
traffic. That flatters the false-alarm rate badly: there is so little clean
traffic that being wrong about it barely costs anything, and "zero false
alarms" over three minutes of clean driving says almost nothing about an hour.
A real vehicle is attacked never, and then briefly.

So this builds the duty cycle that actually matters. The attack-free capture is
tiled to reach about an hour, short bursts are dropped in at wide intervals, and
the whole thing is scored. Attack traffic ends up near 1 %, against 33 % before.

Tiling is an artifact and is treated as one. Repeating the capture makes each
ID's inter-arrival jump once per seam, which is a real discontinuity that the
detector is right to notice and that a real vehicle would never produce. Alarms
inside a guard band around each seam are counted and reported separately rather
than quietly dropped or quietly counted; both totals are printed.

The adversarial modes are the other half. `phase` is the one to beat: it places
each injected frame exactly halfway between two of the victim's real frames, so
every gap is precisely half the normal period and the arrival pattern stays
perfectly regular. There is no burst, no jitter and no rate spike beyond 2x. If
anything gets through, it is this.

  soak     rare short bursts, rates and victims drawn across the whole range
  phase    injected at the exact midpoint of the victim's period, 2x rate
  creep    1.2x the victim's rate, the quietest flood that still adds frames
  rampup   each burst ramps from 1.2x to 8x, so it starts below any threshold
  single   one injected frame per burst, which is the limit case
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from attack_on_real import FRAME_TIME                             # noqa: E402
from can_data import load_hcrl_normal_txt                         # noqa: E402
from cross_eval import load_baseline                              # noqa: E402
from eval_windows import (GRACE_S, alarm_intervals,               # noqa: E402
                          attack_windows, classify, merge_episodes)
from features import FEATURE_SETS, extract                        # noqa: E402
from select_model import predict_tables                           # noqa: E402

SEAM_GUARD_S = 1.0        # alarms this close to a tile seam are tiling, not signal
SOAK_RATES = (1.5, 2.0, 3.0, 5.0, 8.0, 16.0, 33.0)
BURST_S = (0.3, 2.0)      # a real spoof needs a gauge to move, not a minute
GAP_S = (30.0, 120.0)     # and then the attacker stops


def tile(clean, times: int):
    """Repeat the capture end to end, returning the seam timestamps."""
    t = clean["timestamp"].to_numpy(np.float64)
    t = t - t[0]
    dur = float(t[-1])
    # advance by the capture length plus one median bus gap, so the seam is at
    # least a plausible inter-frame time rather than two frames at once
    step = dur + float(np.median(np.diff(t)))
    ts = np.concatenate([t + k * step for k in range(times)])
    seams = [k * step for k in range(1, times)]
    rep = lambda a: np.tile(a, times)                            # noqa: E731
    return (ts,
            rep(clean["can_id"].to_numpy(np.int64)),
            rep(clean["dlc"].to_numpy(np.int64)),
            rep(clean["payload"].to_numpy(np.uint64)),
            seams, dur)


def burst_times(mode, start, rate, victim_period, victim_t, rng,
                burst_s=BURST_S):
    """Injection timestamps for one burst."""
    dur = rng.uniform(*burst_s)
    if mode == "single":
        return np.array([start])
    if mode == "phase":
        # exactly halfway between the victim's own frames: every gap is
        # precisely half the normal period, with no jitter and no burst
        v = victim_t[(victim_t >= start) & (victim_t <= start + dur)]
        if len(v) < 2:
            return np.array([])
        return (v[:-1] + v[1:]) / 2.0
    if mode == "rampup":
        # rate climbs through the burst, so it opens below any fixed threshold
        out, now, r = [], start, 1.2
        while now < start + dur:
            out.append(now)
            now += victim_period / r
            r = min(8.0, r * 1.06)
        return np.array(out)
    period = victim_period / rate
    n = max(1, int(dur / period))
    return start + np.arange(n) * period


def build(clean, mode, times, seed, ids, burst_s=BURST_S, gap_s=GAP_S):
    rng = np.random.default_rng(seed)
    t, can_id, dlc, payload, seams, dur = tile(clean, times)
    span = float(t[-1])

    # per-ID median period, needed to pace every mode off the victim's own rate
    periods, pls = {}, {}
    for cid in ids:
        vt = t[can_id == cid]
        if len(vt) < 100:
            continue
        periods[cid] = float(np.median(np.diff(vt)))
        pls[cid] = payload[can_id == cid]
    victims = sorted(periods)
    if not victims:
        raise SystemExit("no ID in the capture is frequent enough to flood")

    bursts, a_t, a_id, a_pl = [], [], [], []
    now = 60.0                     # leave clean traffic for the baseline
    k = 0
    while now < span - 10.0:
        cid = victims[k % len(victims)]
        rate = (2.0 if mode == "phase" else
                1.2 if mode == "creep" else
                float(rng.choice(SOAK_RATES)))
        vt = t[can_id == cid]
        bt = burst_times(mode, now, rate, periods[cid], vt, rng, burst_s)
        if len(bt):
            a_t.append(bt)
            a_id.append(np.full(len(bt), cid, dtype=np.int64))
            a_pl.append(pls[cid][rng.integers(0, len(pls[cid]), size=len(bt))])
            bursts.append({"start": float(bt[0]), "end": float(bt[-1]),
                           "victim": int(cid), "rate": rate,
                           "frames": int(len(bt))})
        now += rng.uniform(*gap_s)
        k += 1

    a_t = np.concatenate(a_t) if a_t else np.array([])
    a_id = np.concatenate(a_id) if a_id else np.array([], dtype=np.int64)
    a_pl = np.concatenate(a_pl) if a_pl else np.array([], dtype=np.uint64)
    n_a = len(a_t)

    all_t = np.concatenate([t, a_t])
    all_id = np.concatenate([can_id, a_id])
    all_dlc = np.concatenate([dlc, np.full(n_a, 8, dtype=np.int64)])
    all_pl = np.concatenate([payload, a_pl])
    lab = np.concatenate([np.zeros(len(t), np.int8), np.ones(n_a, np.int8)])

    # the flooded ID wins arbitration ties, then the bus serialises everything
    desired = all_t - np.where(lab == 1, 1e-9, 0.0)
    order = np.argsort(desired, kind="stable")
    all_t, all_id, all_dlc, all_pl, lab = (all_t[order], all_id[order],
                                           all_dlc[order], all_pl[order],
                                           lab[order])
    idx = np.arange(len(all_t), dtype=np.float64) * FRAME_TIME
    all_t = np.maximum.accumulate(all_t - idx) + idx
    return all_t, all_id, all_dlc, all_pl, lab, bursts, seams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/normal_run_data.txt")
    ap.add_argument("--model", default="results/mixed/model.json")
    ap.add_argument("--baseline", default="results/mixed/baseline.json")
    ap.add_argument("--mode", default="soak",
                    choices=["soak", "phase", "creep", "rampup", "single"])
    ap.add_argument("--times", type=int, default=7,
                    help="how many times to repeat the capture")
    ap.add_argument("--thresholds", default="2,3,4,6,8,12")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--cache-out", default="",
                    help="save the extracted features as an evcache npz, so "
                         "the search and stability harnesses can score "
                         "candidates against this trace without rebuilding it")
    ap.add_argument("--csv-out", default="",
                    help="write the trace as an HCRL-format CSV so the normal "
                         "prepare/sweep/select pipeline can train on it")
    ap.add_argument("--burst", default="0.3,2.0",
                    help="burst length range in seconds")
    ap.add_argument("--gap", default="30,120",
                    help="seconds of clean traffic between bursts")
    args = ap.parse_args()
    burst_s = tuple(float(v) for v in args.burst.split(","))
    gap_s = tuple(float(v) for v in args.gap.split(","))

    with open(args.model) as fh:
        model = json.load(fh)
    baseline = load_baseline(args.baseline)
    cols = FEATURE_SETS[model["feature_set"]]

    clean = load_hcrl_normal_txt(args.clean)
    ids = sorted(clean["can_id"].unique().tolist())
    t, can_id, dlc, payload, lab, bursts, seams = build(
        clean, args.mode, args.times, args.seed, ids, burst_s, gap_s)

    if args.csv_out:
        from attack_on_real import write_csv
        write_csv(args.csv_out, t, can_id, dlc, payload, lab)
        mb = os.path.getsize(args.csv_out) / 1e6
        print(f"wrote {args.csv_out} ({mb:.0f} MB, {len(t)} frames, "
              f"{int(lab.sum())} injected, {len(bursts)} bursts)")

    import pandas as pd
    df = pd.DataFrame({"timestamp": t, "can_id": can_id, "dlc": dlc,
                       "payload": payload, "label": lab})
    feats = extract(df, baseline)
    X = feats[cols].to_numpy(np.int32)
    pred = predict_tables(model, X)

    if args.cache_out:
        from features import FEATURE_NAMES
        np.savez_compressed(
            args.cache_out, X=feats[FEATURE_NAMES].to_numpy(np.int32),
            columns=np.array(FEATURE_NAMES), t=t,
            y=lab.astype(np.int8), can_id=can_id.astype(np.int32))
        print(f"wrote {args.cache_out} "
              f"({os.path.getsize(args.cache_out) / 1e6:.0f} MB)\n")

    span = float(t[-1] - t[0])
    wins = attack_windows(t, lab)
    atk = sum(e - s for s, e in wins)
    clean_h = max(span - atk, 1e-9) / 3600.0

    print(f"mode        {args.mode}")
    print(f"trace       {len(t)} frames over {span / 60:.1f} min "
          f"({args.times} repeats of the capture, {len(seams)} seams)")
    print(f"attacks     {len(bursts)} bursts, {len(wins)} windows, "
          f"{atk:.1f} s of attack = {100 * atk / span:.2f} % of the trace")
    print(f"clean       {clean_h * 60:.1f} min")
    if bursts:
        fr = [b["frames"] for b in bursts]
        print(f"bursts      {min(fr)}-{max(fr)} injected frames each, "
              f"{len(set(b['victim'] for b in bursts))} distinct victim IDs")
    print(f"per frame   {int(((pred == 1) & (lab == 0)).sum())} clean frames "
          f"flagged, {int(((pred == 0) & (lab == 1)).sum())} injected missed")
    print()

    hdr = (f"{'threshold':>10}{'bursts caught':>16}{'median':>10}{'worst':>10}"
           f"{'false alarms':>14}{'per hour':>10}{'at seams':>10}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for m in [float(v) for v in args.thresholds.split(",")]:
        ev = merge_episodes(alarm_intervals(t, can_id, pred, m))
        inside, ring, false = classify(ev, wins)
        hit, lat = 0, []
        for ws, we in wins:
            over = [(a, b) for a, b, _ in inside
                    if b >= ws and a <= we + GRACE_S]
            if over:
                hit += 1
                lat.append(max(0.0, min(a for a, _ in over) - ws) * 1000.0)
        # a seam is a tiling artifact, not traffic a vehicle would produce
        seam_fa = [e for e in false
                   if any(abs(e[0] - s) <= SEAM_GUARD_S for s in seams)]
        real_fa = [e for e in false if e not in seam_fa]
        per_h = len(real_fa) / clean_h
        # which bursts got through, so a miss can be diagnosed rather than
        # reported as a bare count
        missed = []
        for i, (ws, we) in enumerate(wins):
            if not any(b >= ws and a <= we + GRACE_S for a, b, _ in inside):
                near = min(bursts, key=lambda x: abs(x["start"] - ws)) \
                    if bursts else {}
                missed.append({"window": i, "start": ws,
                               "seconds": we - ws,
                               "victim": near.get("victim"),
                               "rate": near.get("rate"),
                               "frames": near.get("frames")})
        rows.append({"m": m, "missed": missed,
                     "windows": len(wins), "detected": hit,
                     "median_ms": float(np.median(lat)) if lat else None,
                     "worst_ms": float(np.max(lat)) if lat else None,
                     "false_alarms": len(real_fa), "per_hour": per_h,
                     "seam_alarms": len(seam_fa), "ringing": len(ring)})
        print(f"{m:>10.0f}{hit:>9} / {len(wins):<4}"
              f"{(np.median(lat) if lat else float('nan')):>7.1f} ms"
              f"{(np.max(lat) if lat else float('nan')):>7.1f} ms"
              f"{len(real_fa):>14}{per_h:>10.2f}{len(seam_fa):>10}")

    print()
    ok = [r for r in rows if r["detected"] == r["windows"]
          and r["false_alarms"] == 0]
    if ok:
        b = min(ok, key=lambda r: r["worst_ms"])
        print(f"every burst caught with no false alarm in "
              f"{clean_h * 60:.0f} min of clean traffic at threshold "
              f"{b['m']:.0f}, worst case {b['worst_ms']:.1f} ms")
    else:
        best = max(rows, key=lambda r: (r["detected"], -r["false_alarms"]))
        print(f"nothing caught every burst cleanly. best: "
              f"{best['detected']}/{best['windows']} at threshold "
              f"{best['m']:.0f} with {best['false_alarms']} false alarm"
              f"{'' if best['false_alarms'] == 1 else 's'} "
              f"({best['per_hour']:.2f}/hour)")
        miss = [r for r in rows if r["false_alarms"] == 0]
        if miss:
            w = max(miss, key=lambda r: r["detected"])
            print(f"cleanest setting that raises no false alarm: threshold "
                  f"{w['m']:.0f}, catching {w['detected']}/{w['windows']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"mode": args.mode, "frames": len(t),
                       "minutes": span / 60.0, "clean_minutes": clean_h * 60,
                       "attack_percent": 100 * atk / span,
                       "bursts": bursts, "seams": seams, "rows": rows}, fh,
                      indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
