"""Build the shipped detector from attack-free traffic, in one step.

This is the deployment story, and until now the code did not express it. The
final detector uses no attack data at all: the per-ID baseline is fitted on
clean traffic, and both thresholds are calibrated from what that same clean
traffic does. A vehicle can supply an hour of normal driving; it cannot supply
a labelled flood at every rate against every ID, and the whole reason the
thresholds are calibrated rather than trained is that the ones which were
trained left a corridor an attacker could drive through.

So everything the detector needs comes from one file, and this produces it:

    python3 src/build_detector.py --clean data/normal_run_data.txt

writes the baseline and the model, and prints the rule. Nothing else is
required to run the detector on a trace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from can_data import load_hcrl_csv, load_hcrl_normal_txt         # noqa: E402
from features import extract, fit_baseline                       # noqa: E402


def write_baseline(baseline, path: str) -> None:
    """Same on-disk form prepare.py writes, so every tool reads it."""
    with open(path, "w") as fh:
        json.dump({
            "mean_interval_us": baseline.mean_interval,
            "mean_hamming": baseline.mean_hamming,
            "const_mask": {k: f"{v:016x}" for k, v in
                           baseline.const_mask.items()},
            "const_val": {k: f"{v:016x}" for k, v in
                          baseline.const_val.items()},
            "bus_interval_us": baseline.bus_interval_us,
        }, fh, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/normal_run_data.txt",
                    help="attack-free capture. .txt is HCRL's "
                         "normal_run_data format, .csv is the HCRL CSV "
                         "format with every row marked R")
    ap.add_argument("--out-dir", default="results/detector")
    ap.add_argument("--rate-margin", type=float, default=1.12)
    ap.add_argument("--ratio-quantile", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    loader = (load_hcrl_normal_txt if args.clean.endswith(".txt")
              else load_hcrl_csv)
    clean = loader(args.clean)

    if int(clean["label"].sum()) != 0:
        raise SystemExit(
            f"{args.clean} contains {int(clean['label'].sum())} frames marked "
            f"as injected. Calibration has to run on attack-free traffic, "
            f"because the thresholds are set from what normal traffic does."
        )

    span = float(clean["timestamp"].iloc[-1] - clean["timestamp"].iloc[0])
    print(f"clean capture  {len(clean)} frames over {span / 60:.1f} min, "
          f"{clean['can_id'].nunique()} CAN IDs")

    baseline = fit_baseline(clean)
    bpath = os.path.join(args.out_dir, "baseline.json")
    write_baseline(baseline, bpath)
    print(f"baseline       {len(baseline.known_ids)} IDs, bus interval "
          f"{baseline.bus_interval_us:.0f} us  ->  {bpath}")

    # the calibration set is the same clean traffic, as features
    from features import FEATURE_NAMES
    feats = extract(clean, baseline)
    cache = os.path.join(args.out_dir, "clean.npz")
    np.savez_compressed(
        cache, X=feats[FEATURE_NAMES].to_numpy(np.int32),
        columns=np.array(FEATURE_NAMES),
        t=clean["timestamp"].to_numpy(np.float64),
        y=clean["label"].to_numpy(np.int8),
        can_id=clean["can_id"].to_numpy(np.int32))
    print(f"features       -> {cache}\n")

    model = os.path.join(args.out_dir, "model.json")
    subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__),
                                      "calibrate.py"),
         "--clean", cache, "--out", model,
         "--rate-margin", str(args.rate_margin),
         "--ratio-quantile", str(args.ratio_quantile)],
        check=True)

    print(f"\nrun it on a trace with:")
    print(f"  python3 src/eval_windows.py --model {model} \\")
    print(f"      --baseline {bpath} --csv <trace.csv> --m 1,2,4,8")


if __name__ == "__main__":
    main()
