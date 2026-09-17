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
}
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


if __name__ == "__main__":
    main()
