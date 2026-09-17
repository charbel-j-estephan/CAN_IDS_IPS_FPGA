"""Score one frozen model against traces it was not trained on.

This is the deployment question. A deployed detector carries one forest and one
set of per-ID baseline tables. If an attacker switches from flooding 0x000 to
flooding a legitimate ID, or starts replaying valid payloads, that same frozen
model has to cope. Retraining per attack style is not an option in the field.

So the features here are deliberately extracted with the *training* dataset's
baseline, not each trace's own, because that baseline is part of the frozen
artefact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from can_data import load_hcrl_csv, truncate                  # noqa: E402
from features import Baseline, FEATURE_SETS, extract          # noqa: E402
from prepare import TEST_SPAN                                 # noqa: E402
from select_model import predict_tables                       # noqa: E402
from sweep import score                                       # noqa: E402


def load_baseline(path: str) -> Baseline:
    with open(path) as fh:
        d = json.load(fh)
    return Baseline(
        {int(k): int(v) for k, v in d["mean_interval_us"].items()},
        {int(k): int(v) for k, v in d["mean_hamming"].items()},
        {int(k): int(v, 16) for k, v in d.get("const_mask", {}).items()},
        {int(k): int(v, 16) for k, v in d.get("const_val", {}).items()},
        int(d.get("bus_interval_us", 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", required=True,
                    help="the baseline the model was trained with, i.e. the "
                         "one that ships with it")
    ap.add_argument("--traces", nargs="+", required=True,
                    help="name=path.csv pairs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)
    baseline = load_baseline(args.baseline)
    cols = FEATURE_SETS[model["feature_set"]]

    print(f"model: {model['n_trees']} trees, "
          f"{model['cost']['internal_nodes']} comparator nodes, "
          f"depth {model['cost']['max_depth']}")
    print(f"baseline: {os.path.basename(args.baseline)} "
          f"({len(baseline.mean_interval)} known IDs)\n")

    header = (f"{'trace':<12} {'accuracy':>11} {'F1':>10} {'recall':>10} "
              f"{'precision':>11} {'FP':>8} {'FN':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for spec in args.traces:
        name, path = spec.split("=", 1)
        df = load_hcrl_csv(path)
        te = truncate(df, *TEST_SPAN)
        X = extract(te, baseline)[cols].to_numpy(np.int32)
        m = score(te.label.to_numpy(np.int8), predict_tables(model, X))
        rows.append({"trace": name, **m})
        print(f"{name:<12} {m['accuracy'] * 100:10.4f}% "
              f"{m['f1'] * 100:9.4f}% {m['recall'] * 100:9.4f}% "
              f"{m['precision'] * 100:10.4f}% {m['fp']:8d} {m['fn']:8d}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
