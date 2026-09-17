"""Pre-extract the test-slice features of an evaluation trace, once.

Scoring one candidate model against the seven evaluation traces means parsing
half a gigabyte of CSV. That is fine for a final report and hopeless for a
search over model shapes, because the parse dominates and it produces the same
numbers every time. So the parse and the feature extraction are done once per
trace and stored; the search then only re-runs the part that actually depends
on the candidate, which is the forest and the alarm layer.

The cache stores every feature column, not one feature set, so a candidate may
use any set without invalidating it. It is keyed by the baseline file as well
as the trace, because the features are relative to a learned baseline.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from can_data import load_hcrl_csv, truncate                  # noqa: E402
from cross_eval import load_baseline                          # noqa: E402
from features import FEATURE_NAMES, extract                   # noqa: E402
from prepare import TEST_SPAN                                 # noqa: E402
from syncan_data import load_syncan_csv                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--format", default="hcrl", choices=["hcrl", "syncan"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loader = load_hcrl_csv if args.format == "hcrl" else load_syncan_csv
    te = truncate(loader(args.csv), *TEST_SPAN)
    feats = extract(te, load_baseline(args.baseline))
    X = feats[FEATURE_NAMES].to_numpy(np.int32)

    np.savez_compressed(
        args.out,
        X=X,
        columns=np.array(FEATURE_NAMES),
        t=te.timestamp.to_numpy(np.float64),
        y=te.label.to_numpy(np.int8),
        can_id=te.can_id.to_numpy(np.int32),
    )
    mb = os.path.getsize(args.out) / 1e6
    print(f"{os.path.basename(args.csv):24s} {len(te):8d} frames  "
          f"{int(te.label.sum()):7d} injected  ->  {args.out} ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
