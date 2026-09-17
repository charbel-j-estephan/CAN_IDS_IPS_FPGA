"""Collect every sweep, model and cross-evaluation into one JSON for charting."""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

# Human labels for the result directories and the feature sets.
DATASETS = {
    "real": "HCRL DoS (real capture)",
    "realatk_zero-id": "real traffic + 0x000 flood",
    "realatk_valid-id": "real traffic + valid-ID flood",
    "realatk_stealth": "real traffic + stealth flood",
    "syncan_flooding": "SynCAN flooding (independent)",
    "mixed": "real traffic + mixed-rate floods",
    "lowrate": "real traffic + 2x low-rate flood",
    "stealth316": "real traffic + stealth on 0x316",
}

# The seven traces the deployed model is scored against, and the two candidate
# models compared over them. "fast" is the model trained only on 33 %-duty
# floods of one victim ID; "mixed" is trained across attack rates and victims.
FIRMWARE_TRACES = [
    ("dos", "real HCRL DoS"),
    ("zid", "0x000 flood"),
    ("vid", "valid-ID flood"),
    ("s2c0", "stealth on 0x2c0"),
    ("s316", "stealth on 0x316"),
    ("lowrate", "2x low-rate flood"),
    ("mixed", "mixed-rate floods"),
]
FIRMWARE_MODELS = {
    "fast": "trained on 33 % floods of 0x2c0 only",
    "mixed": "trained across rates and victims",
}

# Attack-rate bands for the rate-floor chart, as multiples of the victim ID's
# own normal frame rate.
BANDS = [(0.0, 3.0, "under 3x"), (3.0, 6.0, "3 to 6x"),
         (6.0, 12.0, "6 to 12x"), (12.0, 24.0, "12 to 24x"),
         (24.0, 1e9, "above 24x")]
SETS = {
    "timing": "all features",
    "no_rate": "without rate buckets",
    "timing_only": "without payload invariant",
    "no_payload": "timing and rate only",
    "paper": "reference paper, 2 features",
}


def front(x: pd.DataFrame) -> list:
    """Pareto front: keep a config only if nothing smaller is as accurate."""
    x = x.sort_values(["internal_nodes", "test_accuracy"],
                      ascending=[True, False])
    out, best = [], -1.0
    for _, r in x.iterrows():
        if r.test_accuracy > best:
            out.append({
                "nodes": int(r.internal_nodes),
                "trees": int(r.n_trees),
                "depth": int(r.max_depth),
                "accuracy": round(float(r.test_accuracy) * 100, 4),
                "recall": round(float(r.test_recall) * 100, 4),
                "precision": round(float(r.test_precision) * 100, 4),
                "fp": int(r.test_fp),
                "fn": int(r.test_fn),
            })
            best = r.test_accuracy
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/viz_data.json")
    args = ap.parse_args()

    payload = {"fronts": {}, "models": {}, "cross_eval": {}, "meta": {}}

    for path in sorted(glob.glob(os.path.join(args.results, "*", "sweep.csv"))):
        tag = os.path.basename(os.path.dirname(path))
        if tag not in DATASETS:
            continue
        d = pd.read_csv(path)
        payload["fronts"][tag] = {
            "label": DATASETS[tag],
            "sets": {
                s: {"label": SETS.get(s, s), "points": front(d[d.feature_set == s])}
                for s in d.feature_set.unique() if s in SETS
                and not d[d.feature_set == s].empty
            },
        }
        # what the training/test split looked like
        log = os.path.join(os.path.dirname(path), "prepare.log")
        if os.path.exists(log):
            payload["meta"][tag] = open(log).read().strip().splitlines()

    for path in sorted(glob.glob(os.path.join(args.results, "*", "model.json"))):
        tag = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            m = json.load(fh)
        payload["models"][tag] = {
            "label": DATASETS.get(tag, tag),
            "trees": m["n_trees"],
            "nodes": m["cost"]["internal_nodes"],
            "depth": m["cost"]["max_depth"],
            "features": [m["feature_names"][i]
                         for i in m["cost"]["features_used"]],
            "metrics": m.get("test_metrics", {}),
            "tree_tables": m["trees"],
            "feature_names": m["feature_names"],
        }

    WINDOW_LABEL = {
        "realdos": "real HCRL DoS capture",
        "zeroid": "real traffic + 0x000 flood",
        "validid": "real traffic + valid-ID flood",
        "stealth2c0": "real traffic + stealth on 0x2c0",
        "stealth316": "real traffic + stealth on 0x316",
        "syncan": "SynCAN flooding (independent)",
    }
    order = []
    for tag in ["realdos", "zeroid", "validid", "stealth2c0", "stealth316",
                "syncan"]:
        f = os.path.join(args.results, f"window_{tag}.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            rows = json.load(fh)
        payload.setdefault("windows", {})[tag] = {
            "label": WINDOW_LABEL[tag], "rows": rows,
        }
        order.append(tag)
    payload["window_order"] = order

    # the two candidate models scored over every trace, at each threshold
    fw = {}
    for mdl, mlabel in FIRMWARE_MODELS.items():
        rows = {}
        for tag, tlabel in FIRMWARE_TRACES:
            f = os.path.join(args.results, f"fw_{mdl}_{tag}.json")
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                rows[tag] = {"label": tlabel, "thresholds": json.load(fh)}
        if rows:
            fw[mdl] = {"label": mlabel, "traces": rows}
    if fw:
        payload["firmware"] = fw
        payload["firmware_order"] = [t for t, _ in FIRMWARE_TRACES]

    # detection against how fast the flood actually was
    for mdl in FIRMWARE_MODELS:
        f = os.path.join(args.results, f"ratefloor_{mdl}.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            wins = json.load(fh)
        bands = []
        for lo, hi, label in BANDS:
            sel = [w for w in wins if lo <= w["rate"] < hi]
            if not sel:
                continue
            got = [w for w in sel if w["detected"]]
            bands.append({
                "label": label,
                "windows": len(sel),
                "detected": len(got),
                "median_ms": round(float(pd.Series(
                    [w["latency_ms"] for w in got]).median()), 2) if got else None,
                "worst_ms": round(max((w["latency_ms"] for w in got),
                                      default=0.0), 2) if got else None,
            })
        payload.setdefault("rate_floor", {})[mdl] = bands

    # the shipped detector's own numbers, straight from final_report.py, so
    # the page cannot show a model the repo no longer ships
    fr = os.path.join(args.results, "final_report.json")
    if os.path.exists(fr):
        with open(fr) as fh:
            payload["final"] = json.load(fh)

    ADV = {"phase": "exact-midpoint phasing, 2x",
           "creep": "creep at 1.2x the victim's rate",
           "rampup": "ramp from 1.2x to 8x",
           "single": "one injected frame"}
    adv = []
    for tag, label in ADV.items():
        f = os.path.join(args.results, f"final_adv_{tag}.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            d = json.load(fh)
        r = [x for x in d["rows"] if x["m"] == 2]
        if not r:
            continue
        r = r[0]
        adv.append({"tag": tag, "label": label,
                    "windows": r["windows"], "detected": r["detected"],
                    "median_ms": r["median_ms"], "worst_ms": r["worst_ms"],
                    "false_alarms": r["false_alarms"],
                    "per_hour": r["per_hour"],
                    "clean_minutes": d["clean_minutes"],
                    "attack_percent": d["attack_percent"]})
    if adv:
        payload["adversarial"] = adv

    for path in sorted(glob.glob(os.path.join(args.results, "cross_eval*.json"))):
        name = os.path.basename(path).replace(".json", "")
        with open(path) as fh:
            payload["cross_eval"][name] = json.load(fh)

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({kb:.0f} kB)")
    print(f"  fronts      {list(payload['fronts'])}")
    print(f"  models      {list(payload['models'])}")
    print(f"  cross_eval  {list(payload['cross_eval'])}")
    print(f"  windows     {payload.get('window_order', [])}")
    print(f"  firmware    {list(payload.get('firmware', {}))}")
    print(f"  rate_floor  {list(payload.get('rate_floor', {}))}")
    print(f"  final       {'yes' if 'final' in payload else 'no'}"
          f", adversarial {len(payload.get('adversarial', []))} modes")


if __name__ == "__main__":
    main()
