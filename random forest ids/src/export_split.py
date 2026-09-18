"""Write the train and test truncations out as CSVs, on demand.

The split is already deterministic and reproducible from constants:
prepare.TRAIN_SPAN is the first 45 % of a capture, TEST_SPAN the last 45 %,
and the 10 % between them is discarded so no feature can span the boundary.
Nothing needs these files to exist. They are written when something outside
this repo needs to read the exact rows -- an HDL testbench, a different
toolchain, a reviewer who wants to diff the split rather than trust it.

They are large and derived, so data/ is gitignored and they stay out of the
repository. Regenerate rather than store.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from attack_on_real import write_csv                            # noqa: E402
from can_data import load_hcrl_csv, load_hcrl_normal_txt, truncate  # noqa: E402
from prepare import TEST_SPAN, TRAIN_SPAN                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/DoS_dataset.csv")
    ap.add_argument("--prefix", default="data/dos")
    ap.add_argument("--format", default="hcrl", choices=["hcrl", "hcrl_txt"])
    args = ap.parse_args()

    loader = load_hcrl_csv if args.format == "hcrl" else load_hcrl_normal_txt
    df = loader(args.csv)
    total = len(df)

    for name, span in (("train", TRAIN_SPAN), ("test", TEST_SPAN)):
        part = truncate(df, *span)
        out = f"{args.prefix}_{name}_trunc.csv"
        write_csv(out, part.timestamp.to_numpy(np.float64),
                  part.can_id.to_numpy(np.int64),
                  part.dlc.to_numpy(np.int64),
                  part.payload.to_numpy(np.uint64),
                  part.label.to_numpy(np.int8))
        atk = int(part.label.sum())
        secs = float(part.timestamp.iloc[-1] - part.timestamp.iloc[0])
        print(f"{name:5}  frames {len(part):>9}  "
              f"({span[0]:.0%} to {span[1]:.0%} of {total})  "
              f"{secs:7.1f} s  attack {atk:>8} ({100.0 * atk / len(part):.2f} %)"
              f"  -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)")

    gap = TEST_SPAN[0] - TRAIN_SPAN[1]
    print(f"\n{gap:.0%} discarded between them, so no feature window spans the "
          f"boundary.\nThe split is contiguous rather than shuffled because "
          f"every timing feature\nleaks under a random split: a frame's "
          f"inter-arrival is computed from its\nneighbours, which a shuffle "
          f"puts on both sides of the divide.")


if __name__ == "__main__":
    main()
