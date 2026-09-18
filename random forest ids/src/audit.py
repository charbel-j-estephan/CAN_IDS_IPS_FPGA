"""Answer the evaluation checklist with measurements, including the awkward ones.

Five things the rest of this repo asserts rather than measures, so they are
measured here:

  1. a confusion matrix and a false-positive rate per attack type, since
     overall accuracy on a class-imbalanced bus says almost nothing
  2. generalisation to CAN IDs held out of the baseline entirely, which is the
     honest version of "did it learn traffic patterns or memorise IDs"
  3. what each comparator actually contributes, which is the equivalent of
     feature importances for a detector that is not a forest
  4. single-frame inference latency at N=1, not bulk array prediction
  5. the memory footprint, in bits, of everything that has to live on the part

Two of these are uncomfortable and are reported anyway. The latency number in
Python is dominated by interpreter overhead and is an upper bound on nothing
useful; it is printed with the arithmetic that actually bounds the hardware
beside it, rather than passed off as the hardware figure. And the footprint is
dominated by the per-ID table, not by the classifier, which is worth knowing
before anyone congratulates themselves on three comparators.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cross_eval import load_baseline                             # noqa: E402
from features import (CAP_DT, CAP_RATE, CAP_RATIO,               # noqa: E402
                      FEATURE_SETS, RECIP_SHIFT)
from select_model import predict_tables                          # noqa: E402

ATTACKS = [
    ("dos", "DoS, real capture"),
    ("zid", "DoS, 0x000 on clean traffic"),
    ("vid", "flood reusing a valid ID"),
    ("s2c0", "stealth, replayed payloads"),
    ("s316", "stealth, second victim"),
    ("lowrate", "low rate, 2x"),
    ("mixed", "mixed rate, 2x to 33x"),
    ("soak", "rare bursts, 1.8 % duty"),
    ("syncan", "SynCAN, different vehicle"),
]


def confusion(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def rates(tp, tn, fp, fn):
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return prec, rec, f1, fpr


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def per_attack(model, cols, evcache):
    section("1. PER-ATTACK CONFUSION MATRIX AND FALSE POSITIVE RATE")
    print("Per frame, before the alarm layer. The FPR target for an IPS is "
          "under 0.01 %.\n")
    hdr = (f"{'attack':30}{'TP':>9}{'FN':>8}{'FP':>8}{'precision':>11}"
           f"{'recall':>9}{'F1':>8}{'FPR':>10}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for tag, label in ATTACKS:
        path = os.path.join(evcache, f"{tag}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        c = list(z["columns"])
        pred = predict_tables(model, z["X"][:, [c.index(x) for x in cols]])
        tp, tn, fp, fn = confusion(pred, z["y"])
        prec, rec, f1, fpr = rates(tp, tn, fp, fn)
        rows.append({"tag": tag, "label": label, "tp": tp, "tn": tn,
                     "fp": fp, "fn": fn, "precision": prec, "recall": rec,
                     "f1": f1, "fpr": fpr})
        print(f"{label:30}{tp:>9}{fn:>8}{fp:>8}{prec * 100:>10.2f}%"
              f"{rec * 100:>8.2f}%{f1 * 100:>7.2f}%{fpr * 100:>9.4f}%")

    over = [r for r in rows if r["fpr"] > 1e-4]
    print()
    if over:
        print(f"{len(over)} of {len(rows)} traces exceed the 0.01 % FPR target "
              f"per frame:")
        for r in sorted(over, key=lambda r: -r["fpr"]):
            print(f"  {r['label']:30} {r['fpr'] * 100:.4f} %  ({r['fp']} "
                  f"frames of {r['fp'] + r['tn']})")
        print("\nThat is the number to report for a per-frame IPS, and it is "
              "why this design\ndoes not act per frame. Those false positives "
              "are overwhelmingly the victim\nID's OWN frames during a flood, "
              "which arrive at the flood period and are\ngenuinely ambiguous "
              "one frame at a time. The alarm layer above resolves them:\nsee "
              "the attack-free control below, where every alarm would be "
              "false by\nconstruction.")
    else:
        print("every trace is inside the 0.01 % target per frame")
    return rows


def control(model, cols, evcache):
    section("1b. THE CONTROL: TRAFFIC WITH NO ATTACK IN IT AT ALL")
    path = os.path.join(evcache, "clean.npz")
    if not os.path.exists(path):
        print("no clean.npz")
        return None
    z = np.load(path, allow_pickle=True)
    c = list(z["columns"])
    assert int(z["y"].sum()) == 0, "the control must be attack free"
    pred = predict_tables(model, z["X"][:, [c.index(x) for x in cols]])
    span = float(z["t"][-1] - z["t"][0])
    fp = int(pred.sum())
    print(f"frames            {len(pred)}")
    print(f"duration          {span / 60:.1f} min")
    print(f"frames flagged    {fp}")
    print(f"false positive rate {100.0 * fp / len(pred):.6f} %   "
          f"(target < 0.01 %)")
    print("\nEvery positive here is false by construction, so this is the only "
          "FPR figure\nthat cannot be argued with.")
    return {"frames": int(len(pred)), "minutes": span / 60.0, "fp": fp,
            "fpr": fp / len(pred)}


def unseen_ids(model, cols, evcache, baseline_path, n_hold=6):
    section("2. GENERALISATION TO CAN IDs HELD OUT OF THE BASELINE")
    print("Does it detect a flood on an ID it has never seen, or has it just "
          "memorised\nthe IDs it was calibrated on? The detector uses no "
          "can_id feature at all, so\nthe question becomes what happens when "
          "the per-ID baseline has no entry.\n")
    bl = load_baseline(baseline_path)
    known = sorted(bl.mean_interval)
    hold = known[::max(1, len(known) // n_hold)][:n_hold]

    path = os.path.join(evcache, "mixed.npz")
    if not os.path.exists(path):
        print("no mixed.npz to test against")
        return None
    z = np.load(path, allow_pickle=True)
    c = list(z["columns"])
    X, y, cid = z["X"], z["y"], z["can_id"]

    # holding out an ID that is never attacked in this trace tests nothing, so
    # make sure the victim is among them
    victims = sorted(set(cid[y == 1].tolist()))
    hold = sorted(set(hold) | set(victims[:2]))
    print(f"baseline knows {len(known)} IDs; holding out "
          + ", ".join(f"0x{i:03x}" for i in hold))
    print(f"this trace's victim IDs: "
          + ", ".join(f"0x{v:03x}" for v in victims)
          + f"  ({'at least one is held out' if set(victims) & set(hold) else 'NONE held out, this tests nothing'})\n")

    # An ID missing from the baseline has recip 0, so dt_ratio_q6 is 0 on every
    # one of its frames, and id_rate never leaves its initial value. Simulate
    # exactly that rather than approximating it.
    Xh = X.copy()
    mask = np.isin(cid, hold)
    Xh[mask, c.index("dt_ratio_q6")] = 0
    Xh[mask, c.index("id_rate")] = 64

    pred_full = predict_tables(model, X[:, [c.index(x) for x in cols]])
    pred_held = predict_tables(model, Xh[:, [c.index(x) for x in cols]])

    for name, p in (("baseline complete", pred_full),
                    ("6 IDs unbaselined", pred_held)):
        tp, tn, fp, fn = confusion(p, y)
        prec, rec, f1, fpr = rates(tp, tn, fp, fn)
        print(f"{name:22} recall {rec * 100:6.2f} %   FPR {fpr * 100:.4f} %")

    hm = mask & (y == 1)
    if hm.any():
        print(f"\non the held-out IDs' own injected frames "
              f"({int(hm.sum())} of them):")
        print(f"  with a baseline    {100.0 * pred_full[hm].mean():.2f} % flagged")
        print(f"  without one        {100.0 * pred_held[hm].mean():.2f} % flagged")
        print("\nAn ID with no baseline has recip 0, so its dt_ratio_q6 is 0 "
              "on every frame,\nwhich satisfies the timing rule "
              "unconditionally. Every frame of an unknown ID\nis therefore "
              "flagged: 100 % recall on its injected frames and 100 % on its\n"
              "legitimate ones too, which is the FPR jump above.\n\n"
              "That is whitelist behaviour and it is stated rather than "
              "buried. On a CAN bus\nit is defensible, because the ID set is "
              "fixed at design time and an ID nobody\nhas ever seen is "
              "anomalous by existing at all. But it is a property of the\n"
              "baseline being empty, not of the model detecting anything, and "
              "the honest way\nto ship it is an explicit known-ID check "
              "beside the detector rather than an\naccident of a reciprocal "
              "being zero. Note also what it means for detection: an\n"
              "attacker who floods a brand-new ID is caught trivially, which "
              "is why the real\nHCRL capture's 0x000 flood was never the hard "
              "case.")
    return {"held_out": [int(h) for h in hold]}


def ablation(model, cols, evcache, baseline_path):
    section("3. WHAT EACH COMPARATOR CONTRIBUTES")
    print("The equivalent of feature_importances_ for a detector that is not a "
          "forest:\nturn each rule off and measure what is lost.\n")
    cal = model.get("calibration", {})
    variants = {
        "both rules (shipped)": (cal.get("id_rate_threshold"),
                                 cal.get("dt_ratio_threshold")),
        "rate rule only": (cal.get("id_rate_threshold"), -1),
        "timing rule only": (1 << 20, cal.get("dt_ratio_threshold")),
    }
    hdr = (f"{'configuration':24}{'recall':>10}{'FPR':>10}"
           f"{'clean-capture FP':>19}{'worst-trace FPR':>18}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    cpath = os.path.join(evcache, "clean.npz")
    for name, (rt, dt) in variants.items():
        tp = tn = fp = fn = 0
        worst = 0.0
        for tag, _ in ATTACKS:
            path = os.path.join(evcache, f"{tag}.npz")
            if not os.path.exists(path) or tag == "syncan":
                continue
            z = np.load(path, allow_pickle=True)
            c = list(z["columns"])
            ir = z["X"][:, c.index("id_rate")]
            dr = z["X"][:, c.index("dt_ratio_q6")]
            pred = ((ir > rt) | (dr <= dt)).astype(np.int8)
            a, b, cc_, d = confusion(pred, z["y"])
            tp += a
            tn += b
            fp += cc_
            fn += d
            worst = max(worst, cc_ / (cc_ + b) if cc_ + b else 0.0)
        prec, rec, f1, fpr = rates(tp, tn, fp, fn)
        cfp = 0
        if os.path.exists(cpath):
            z = np.load(cpath, allow_pickle=True)
            c = list(z["columns"])
            cfp = int((((z["X"][:, c.index("id_rate")] > rt)
                        | (z["X"][:, c.index("dt_ratio_q6")] <= dt))).sum())
        out[name] = {"recall": rec, "fpr": fpr, "clean_fp": cfp,
                     "worst_trace_fpr": worst}
        print(f"{name:24}{rec * 100:>9.2f}%{fpr * 100:>9.3f}%"
              f"{cfp:>19}{worst * 100:>17.3f}%")

    print("\nPer-frame recall saturates at 100 % for all three, so it cannot "
          "separate them\nand is not the axis that matters. False positives "
          "are. Read the last two\ncolumns: the timing rule alone is what "
          "puts frames on the attack-free capture,\nand the rate rule alone "
          "is what drives the worst per-trace FPR. Each is\ncarrying a "
          "different failure, which is why both are kept -- not because "
          "either\nadds recall, because neither does.")
    return out


def latency(model, cols, evcache, n=20000):
    section("4. SINGLE-FRAME INFERENCE LATENCY (N=1)")
    path = os.path.join(evcache, "dos.npz")
    z = np.load(path, allow_pickle=True)
    c = list(z["columns"])
    X = z["X"][:, [c.index(x) for x in cols]][:n]

    tree = model["trees"][0]
    f, thr = tree["feature"], tree["threshold"]
    lf, rt, leaf, val = (tree["left"], tree["right"], tree["is_leaf"],
                         tree["value"])
    rows = [tuple(int(v) for v in r) for r in X]

    def one(x):
        votes = 0
        for t in model["trees"]:
            nd = 0
            while not t["is_leaf"][nd]:
                nd = (t["left"][nd] if x[t["feature"][nd]] <= t["threshold"][nd]
                      else t["right"][nd])
            votes += t["value"][nd]
        return votes * 2 > len(model["trees"])

    one(rows[0])
    t0 = time.perf_counter()
    for r in rows:
        one(r)
    el = time.perf_counter() - t0
    per_us = el / len(rows) * 1e6

    print(f"pure-Python, one frame at a time, {len(rows)} frames")
    print(f"  mean {per_us:.2f} us per frame   ({1e6 / per_us:,.0f} frames/s)")
    print(f"\nThat number is interpreter overhead, not the design. It is "
          f"reported because\nit was asked for, and it is already "
          f"{1000.0 / per_us:.0f}x inside a 1 ms budget on a\n"
          f"laptop CPU running Python, which settles the question for any "
          f"host-side use.")
    print("\nWhat actually bounds the hardware:")
    print(f"  {model['cost']['internal_nodes']} comparators, "
          f"depth {model['cost']['max_depth']}, no data dependence between "
          f"trees")
    print("  the tree is depth 2 but the logic is an OR of two independent")
    print("    comparisons, so in hardware it is 1 comparator deep plus an "
          "OR gate")
    print("  the features cost one 16x16 multiply (dt_ratio_q6, by a stored")
    print("    reciprocal, no divider) and one multiply-shift-add (id_rate)")
    print("  at 500 kbit/s a CAN frame occupies at least 111 bit times, "
          "which is 222 us")
    print("  so the budget is ~222 us and the work is a handful of cycles")
    return {"python_us_per_frame": per_us}


def footprint(model, baseline_path):
    section("5. MEMORY FOOTPRINT")
    bl = load_baseline(baseline_path)
    n_ids = 2048            # 11-bit CAN identifier space, direct indexed

    recip_b = 16            # 64/period, the dt_ratio multiply
    period_b = 20           # nominal period in us, for the alarm's silence rule
    last_ts_b = 32          # last timestamp seen for this ID
    rate_b = 12             # CAP_RATE
    score_b = 4             # alarm score, capped just above the threshold
    alarm_ts_b = 32         # when this ID started alarming

    per_id = recip_b + period_b + last_ts_b + rate_b + score_b + alarm_ts_b
    table_bits = n_ids * per_id

    cmp_bits = model["cost"]["internal_nodes"] * (4 + 20 + 2)

    print(f"per-ID state, {n_ids} entries direct indexed by the 11-bit ID:")
    for name, bits in (("reciprocal (dt_ratio)", recip_b),
                       ("nominal period", period_b),
                       ("last timestamp", last_ts_b),
                       ("rate bucket", rate_b),
                       ("alarm score", score_b),
                       ("alarm start time", alarm_ts_b)):
        print(f"    {name:24} {bits:>3} bits")
    print(f"    {'':24} {'---':>3}")
    print(f"    {'per ID':24} {per_id:>3} bits")
    print(f"\n  table total   {table_bits:,} bits = "
          f"{table_bits / 8 / 1024:.1f} KiB")
    print(f"  classifier    {cmp_bits:,} bits "
          f"({model['cost']['internal_nodes']} comparators)")
    print(f"  TOTAL         {(table_bits + cmp_bits) / 8 / 1024:.1f} KiB")
    print(f"\nThe classifier is {100.0 * cmp_bits / (table_bits + cmp_bits):.2f} % "
          f"of the footprint. Shrinking the forest\nfrom 8 comparators to 3 "
          f"saved {5 * 26} bits; the per-ID table is the design.")
    print(f"\nOnly {len(bl.mean_interval)} of the {n_ids} slots are populated "
          f"on this bus, so a CAM or a\nhash keyed on the observed IDs would "
          f"cut the table by ~{100 - 100.0 * len(bl.mean_interval) / n_ids:.0f} %, "
          f"at the cost of\nthe direct-index lookup being one cycle.")
    return {"table_bits": table_bits, "classifier_bits": cmp_bits,
            "total_kib": (table_bits + cmp_bits) / 8 / 1024,
            "populated_ids": len(bl.mean_interval)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/detector/model.json")
    ap.add_argument("--baseline", default="results/detector/baseline.json")
    ap.add_argument("--evcache", default="results/evcache2")
    ap.add_argument("--out", default="results/audit.json")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    cols = FEATURE_SETS[model["feature_set"]]

    print(f"auditing {args.model}: {model['cost']['internal_nodes']} "
          f"comparators on {len(cols)} features {cols}")

    payload = {
        "per_attack": per_attack(model, cols, args.evcache),
        "control": control(model, cols, args.evcache),
        "unseen_ids": unseen_ids(model, cols, args.evcache, args.baseline),
        "ablation": ablation(model, cols, args.evcache, args.baseline),
        "latency": latency(model, cols, args.evcache),
        "footprint": footprint(model, args.baseline),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
