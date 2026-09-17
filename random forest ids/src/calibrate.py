"""Set the thresholds from clean traffic, because that is the data you have.

A decision tree picks the split that best separates its training set, so the
threshold it learns encodes the attack rates that happened to be in that set.
That is the failure this project kept rediscovering at increasing depth: a
model trained on 33 % floods missed 2x floods, a model trained on 10 ms victims
missed 100 ms victims, and a model trained across both still missed a flood
paced at exactly half the victim's period.

That last one is worth stating precisely, because it shows the problem is not
coverage. Across 988 871 frames of genuinely attack-free traffic, id_rate never
once exceeds 71. Under a 2x flood placed at the exact midpoint of the victim's
period it sits at 131 to 140. The two are cleanly separated with a gap of 60
units, and the trained forest still missed every one of those attacks, because
it had learned id_rate <= 141 -- a threshold that separates its training data
perfectly while leaving a 70-unit corridor an attacker can drive straight
through.

So the threshold is calibrated on the clean capture instead: take what normal
traffic actually does, add a margin, and flag anything past it. No attack data
is involved, which is both more honest about what a vehicle can supply and
immune to the rate, the victim and the phasing of an attack nobody thought to
generate. It is one-class calibration, and for this problem it dominates.

The margin is expressed in the same units and reported, so the cost of moving
it is visible rather than buried in a fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_SETS                                # noqa: E402


def leaf(value: int) -> dict:
    return {"feature": 0, "threshold": 0, "left": 0, "right": 0,
            "is_leaf": 1, "value": value}


def one_comparator(fidx: int, thr: int, cols, attack_when_le: bool) -> dict:
    """A single-comparator tree: feature <= thr decides, one way or the other."""
    lo, hi = (1, 0) if attack_when_le else (0, 1)
    return {
        "feature": [fidx, 0, 0], "threshold": [int(thr), 0, 0],
        "left": [1, 0, 0], "right": [2, 0, 0],
        "is_leaf": [0, 1, 1], "value": [0, lo, hi],
        "feature_names": list(cols),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="results/evcache/clean.npz")
    ap.add_argument("--set", default="per_id")
    ap.add_argument("--rate-margin", type=float, default=1.12,
                    help="id_rate threshold as a multiple of the clean maximum")
    ap.add_argument("--ratio-quantile", type=float, default=0.01,
                    help="percent of clean frames allowed below the "
                         "dt_ratio_q6 threshold")
    ap.add_argument("--out", default="results/calibrated/model.json")
    args = ap.parse_args()

    z = np.load(args.clean, allow_pickle=True)
    cols_all = list(z["columns"])
    assert int(z["y"].sum()) == 0, "calibration needs attack-free traffic"
    n = len(z["y"])

    ir = z["X"][:, cols_all.index("id_rate")]
    dr = z["X"][:, cols_all.index("dt_ratio_q6")]

    rate_thr = int(np.ceil(ir.max() * args.rate_margin))
    ratio_thr = int(np.floor(np.percentile(dr, args.ratio_quantile)))

    print(f"calibrated on {n} attack-free frames\n")
    print(f"  id_rate      clean max {ir.max()}, "
          f"threshold > {rate_thr}  "
          f"({int((ir > rate_thr).sum())} clean frames past it)")
    print(f"  dt_ratio_q6  clean min {dr.min()}, p{args.ratio_quantile} "
          f"{np.percentile(dr, args.ratio_quantile):.0f}, "
          f"threshold <= {ratio_thr}  "
          f"({int((dr <= ratio_thr).sum())} clean frames past it, "
          f"{100.0 * (dr <= ratio_thr).mean():.5f} %)")

    cols = FEATURE_SETS[args.set]
    ri, di = cols.index("id_rate"), cols.index("dt_ratio_q6")

    # Three trees so the majority vote is the OR of the two rules: either
    # comparator alone gives two votes out of three. Same cost as the rules.
    trees = [
        one_comparator(ri, rate_thr, cols, attack_when_le=False),
        one_comparator(di, ratio_thr, cols, attack_when_le=True),
        one_comparator(ri, rate_thr, cols, attack_when_le=False),
    ]
    model = {
        "feature_set": args.set,
        "feature_names": cols,
        "n_trees": 3,
        "trees": trees,
        "cost": {"internal_nodes": 3, "leaves": 6, "max_depth": 1,
                 "features_used": sorted({ri, di}), "n_features_used": 2},
        "calibration": {
            "clean_frames": n,
            "id_rate_clean_max": int(ir.max()),
            "id_rate_threshold": rate_thr,
            "dt_ratio_clean_min": int(dr.min()),
            "dt_ratio_threshold": ratio_thr,
            "dt_ratio_clean_below_threshold": int((dr <= ratio_thr).sum()),
            "rate_margin": args.rate_margin,
            "ratio_quantile": args.ratio_quantile,
            "trained_on_attacks": False,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(model, fh, indent=1)

    # the rule, stated plainly, since that is the whole point of it being small
    print(f"\nthe whole detector:")
    print(f"  flag a frame when  id_rate > {rate_thr}"
          f"  OR  dt_ratio_q6 <= {ratio_thr}")
    print(f"  that is 2 comparators on 2 features, no attack data used")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
