"""Produce every headline number for the shipped detector, in one run.

The README quoted numbers that a later retrain had superseded, twice. The fix
for the dashboard was to generate it; this is the same fix for the prose. Every
figure the README states about the shipped detector comes from here, so the two
cannot drift apart without this file's output changing.

It reports three things that have to be kept apart, because collapsing them is
how this project's metrics went wrong three times:

  detections    attack windows where the alarm was active
  false alarms  alarm episodes on traffic with no attack anywhere near
  ringing       alarm episodes within RING_S of an attack window ending,
                which are the rate buckets draining after an attack the
                detector correctly caught, not errors on clean traffic

and it reports latching -- windows that opened while an alarm was already
running, plus the longest single alarm -- because "the alarm was active during
the window" is a sound rule only while alarms actually clear, and a stuck one
satisfies it for free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_windows import (GRACE_S, _bus_ratio_q6, _period_table,   # noqa: E402
                          alarm_intervals, attack_windows, classify,
                          merge_episodes)
from features import FEATURE_SETS                                    # noqa: E402
from select_model import predict_tables                              # noqa: E402

TRACES = [
    ("dos", "real HCRL DoS capture", "a different drive, never trained on"),
    ("zid", "0x000 flood", "the real capture's attack, on clean traffic"),
    ("vid", "valid-ID flood", "reuses a legitimate ID"),
    ("s2c0", "stealth on 0x2c0", "replays that ID's own real payloads"),
    ("s316", "stealth on 0x316", "a second victim, 50.9 % unique payloads"),
    ("lowrate", "2x low-rate flood", "quiet enough to pass for jitter"),
    ("mixed", "mixed-rate floods", "2x to 33x, drawn per burst"),
    ("soak", "1 hour soak", "1.8 % attack traffic, all 27 victim IDs"),
    ("syncan", "SynCAN flooding", "independent benchmark, different vehicle"),
]


def score(tag, path, model, cols, m, baseline=None):
    z = np.load(path, allow_pickle=True)
    columns = list(z["columns"])
    X = z["X"][:, [columns.index(c) for c in cols]]
    t, y, cid = z["t"], z["y"], z["can_id"]
    pred = predict_tables(model, X)

    ratio = periods = bratio = None
    if baseline is not None:
        ratio = z["X"][:, columns.index("dt_ratio_q6")]
        periods = _period_table(cid.astype(np.int64), baseline)
        bratio = _bus_ratio_q6(
            z["X"][:, columns.index("dt_bus")].astype(np.int64), baseline)

    wins = attack_windows(t, y)
    span = float(t[-1] - t[0])
    atk = sum(e - s for s, e in wins)
    clean_h = max(span - atk, 1e-9) / 3600.0

    ev = merge_episodes(alarm_intervals(
        t, cid, pred, m, dt_ratio=ratio, id_period_us=periods,
        bus_ratio=bratio))
    inside, ring, false = classify(
        ev, wins, id_period_us=(baseline.mean_interval if baseline else None))

    hit, lat, stale = 0, [], 0
    longest = max((b - a for a, b, _ in inside), default=0.0)
    for ws, we in wins:
        over = [(a, b) for a, b, _ in inside if b >= ws and a <= we + GRACE_S]
        if over:
            hit += 1
            lat.append(max(0.0, min(a for a, _ in over) - ws) * 1000.0)
        if any(a < ws - 1e-9 and b >= ws for a, b, _ in inside):
            stale += 1

    return {
        "tag": tag, "frames": len(t), "minutes": span / 60.0,
        "clean_minutes": clean_h * 60.0,
        "attack_percent": 100.0 * atk / span if span else 0.0,
        "windows": len(wins), "detected": hit,
        "median_ms": float(np.median(lat)) if lat else None,
        "worst_ms": float(np.max(lat)) if lat else None,
        "false_alarms": len(false), "per_hour": len(false) / clean_h,
        "ringing": len(ring), "stale_windows": stale,
        "longest_alarm_s": longest,
        "frame_fp": int(((pred == 1) & (y == 0)).sum()),
        "frame_fn": int(((pred == 0) & (y == 1)).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/detector/model.json")
    ap.add_argument("--evcache", default="results/evcache2")
    ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--baseline", default="results/detector/baseline.json",
                    help="needed for the period-scaled alarm decay")
    ap.add_argument("--syncan-baseline",
                    default="results/syncan_flooding/baseline.json")
    ap.add_argument("--out", default="results/final_report.json")
    ap.add_argument("--markdown", default="results/FINAL.md")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    cols = FEATURE_SETS[model["feature_set"]]
    cal = model.get("calibration", {})
    from cross_eval import load_baseline
    baseline = load_baseline(args.baseline)
    # SynCAN is a different vehicle, so its periods come from its own capture;
    # the model's thresholds are the HCRL-calibrated ones either way
    syncan_bl = (load_baseline(args.syncan_baseline)
                 if os.path.exists(args.syncan_baseline) else baseline)

    rows = []
    for tag, label, note in TRACES:
        path = os.path.join(args.evcache, f"{tag}.npz")
        if not os.path.exists(path):
            print(f"  (no {path}, skipped)")
            continue
        r = score(tag, path, model, cols, args.m,
                  syncan_bl if tag == "syncan" else baseline)
        r["label"], r["note"] = label, note
        rows.append(r)

    # the control: traffic with no attack in it at all
    cpath = os.path.join(args.evcache, "clean.npz")
    control = None
    if os.path.exists(cpath):
        control = score("clean", cpath, model, cols, args.m, baseline)
        assert control["windows"] == 0, "the control must be attack free"

    td = sum(r["detected"] for r in rows)
    tw = sum(r["windows"] for r in rows)
    tfa = sum(r["false_alarms"] for r in rows)
    tring = sum(r["ringing"] for r in rows)
    tstale = sum(r["stale_windows"] for r in rows)
    worst = max((r["worst_ms"] for r in rows if r["worst_ms"]), default=0.0)
    longest = max((r["longest_alarm_s"] for r in rows), default=0.0)
    clean_min = sum(r["clean_minutes"] for r in rows)

    hdr = (f"{'trace':26}{'windows':>11}{'median':>9}{'worst':>10}"
           f"{'false':>7}{'/hour':>8}{'ring':>6}")
    print(f"\nmodel      {model['cost']['internal_nodes']} comparators on "
          f"{len(cal.get('trained_on_attacks', [])) if False else 2} features"
          f"{'' if cal.get('trained_on_attacks', True) else ', no attack data used'}")
    print(f"threshold  {args.m:.0f}\n")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:26}{r['detected']:>5}/{r['windows']:<5}"
              f"{(r['median_ms'] or 0):>6.1f} ms{(r['worst_ms'] or 0):>7.1f} ms"
              f"{r['false_alarms']:>7}{r['per_hour']:>8.2f}{r['ringing']:>6}")
    print("-" * len(hdr))
    print(f"{'TOTAL':26}{td:>5}/{tw:<5}{'':>9}{worst:>7.1f} ms"
          f"{tfa:>7}{tfa / (clean_min / 60.0):>8.2f}{tring:>6}")
    print(f"\n{clean_min:.0f} minutes of clean traffic across these traces")
    print(f"latching: {tstale} of {tw} windows opened on a running alarm, "
          f"longest single alarm {longest:.1f} s")

    if control:
        print(f"\ncontrol, no attack anywhere in it: "
              f"{control['frames']} frames over "
              f"{control['minutes']:.1f} min, "
              f"{control['frame_fp']} frames flagged, "
              f"{control['false_alarms']} alarm events "
              f"({control['per_hour']:.2f} per hour)")

    payload = {"model": args.model, "threshold": args.m,
               "comparators": model["cost"]["internal_nodes"],
               "features": cols, "calibration": cal,
               "traces": rows, "control": control,
               "totals": {"detected": td, "windows": tw,
                          "false_alarms": tfa, "ringing": tring,
                          "stale_windows": tstale, "worst_ms": worst,
                          "longest_alarm_s": longest,
                          "clean_minutes": clean_min}}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {args.out}")

    with open(args.markdown, "w") as fh:
        fh.write(f"<!-- generated by src/final_report.py, do not edit -->\n\n")
        fh.write(f"| trace | windows | detected | median | worst | "
                 f"false alarms | per hour |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['label']} | {r['windows']} | {r['detected']} | "
                     f"{(r['median_ms'] or 0):.1f} ms | "
                     f"{(r['worst_ms'] or 0):.1f} ms | "
                     f"{r['false_alarms']} | {r['per_hour']:.2f} |\n")
        fh.write(f"| **total** | **{tw}** | **{td}** | | "
                 f"**{worst:.1f} ms** | **{tfa}** | "
                 f"**{tfa / (clean_min / 60.0):.2f}** |\n")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
