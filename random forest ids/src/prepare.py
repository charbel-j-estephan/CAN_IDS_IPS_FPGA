"""Load the trace, cut two disjoint truncations, extract features, cache them.

Train truncation  : frames [0 %, 45 %)
Held-out gap      : frames [45 %, 55 %)   discarded
Test truncation   : frames [55 %, 100 %)

The gap matters. Every feature is sequential, so if the two truncations touched,
the last training frames would be the immediate predecessors of the first test
frames and the per-ID state would carry straight across the boundary.

The per-ID baseline (mean interval, mean Hamming distance) is fitted on clean
frames from the TRAIN truncation only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from can_data import load_hcrl_csv, truncate          # noqa: E402
from features import extract, fit_baseline            # noqa: E402

TRAIN_SPAN = (0.00, 0.45)
TEST_SPAN = (0.55, 1.00)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/cache.npz")
    ap.add_argument("--baseline-out", default="results/baseline.json")
    ap.add_argument("--raw-out", default="",
                    help="optional npz of the raw test frames, used by the "
                         "end-to-end RTL testbench")
    args = ap.parse_args()

    df = load_hcrl_csv(args.csv)
    print(f"loaded {len(df)} frames, {df.label.mean() * 100:.2f} % injected")

    tr = truncate(df, *TRAIN_SPAN)
    te = truncate(df, *TEST_SPAN)
    print(f"train truncation {len(tr)} frames, {tr.label.mean() * 100:.2f} % injected")
    print(f"test  truncation {len(te)} frames, {te.label.mean() * 100:.2f} % injected")

    # A truncation with no attacks, or no normal traffic, trains or scores a
    # model that is meaningless while still reporting a plausible accuracy.
    # Catch it here rather than letting it through to the sweep.
    for name, part in (("train", tr), ("test", te)):
        n_attack = int(part.label.sum())
        n_normal = int(len(part) - n_attack)
        if n_attack == 0 or n_normal == 0:
            raise SystemExit(
                f"the {name} truncation holds {n_attack} attack and "
                f"{n_normal} normal frames, so it cannot be used. The attack "
                f"bursts are not spread across the trace. Use a longer capture, "
                f"or change TRAIN_SPAN / TEST_SPAN in this file so both "
                f"truncations straddle attack activity."
            )

    baseline = fit_baseline(tr)
    print(f"baseline learned for {len(baseline.known_ids)} CAN IDs")

    Xtr = extract(tr, baseline)
    Xte = extract(te, baseline)

    unseen = set(te.can_id.unique()) - baseline.known_ids
    print(f"IDs in test that were never seen clean in train: "
          f"{sorted(hex(i) for i in unseen)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        Xtr=Xtr.to_numpy(np.int32), ytr=tr.label.to_numpy(np.int8),
        Xte=Xte.to_numpy(np.int32), yte=te.label.to_numpy(np.int8),
        columns=np.array(list(Xtr.columns)),
    )
    if args.raw_out:
        ts_us = np.rint(
            (te["timestamp"].to_numpy() - te["timestamp"].iloc[0]) * 1e6
        ).astype(np.int64)
        np.savez_compressed(
            args.raw_out,
            can_id=te["can_id"].to_numpy(np.int64),
            dlc=te["dlc"].to_numpy(np.int64),
            payload=te["payload"].to_numpy(np.uint64),
            ts_us=ts_us,
            label=te["label"].to_numpy(np.int8),
        )
        print(f"raw test frames -> {args.raw_out}")

    with open(args.baseline_out, "w") as fh:
        json.dump(
            {
                "mean_interval_us": baseline.mean_interval,
                "mean_hamming": baseline.mean_hamming,
                "const_mask": {k: f"{v:016x}" for k, v in
                               baseline.const_mask.items()},
                "const_val": {k: f"{v:016x}" for k, v in
                              baseline.const_val.items()},
                "bus_interval_us": baseline.bus_interval_us,
            },
            fh,
            indent=2,
            sort_keys=True,
        )
    print(f"cached -> {args.out}")


if __name__ == "__main__":
    main()
