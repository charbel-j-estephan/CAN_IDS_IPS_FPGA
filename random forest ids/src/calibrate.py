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


def or_tree(rate_idx: int, rate_thr: int,
            ratio_idx: int, ratio_thr: int, cols) -> dict:
    """One depth-2 tree computing `rate > R OR ratio <= T`.

    It has to be one tree, and getting that wrong is worth recording. The
    first version used three single-comparator trees -- the rate rule twice
    and the timing rule once -- on the reasoning that a majority of three
    would give the OR. It does not. The timing rule alone collects one vote
    out of three and loses, so the forest reduced to the rate rule with two
    dead comparators bolted on, and a frame with dt_ratio_q6 = 0 (an ID with
    no baseline at all, the most anomalous timing possible) was classified
    normal. Majority voting cannot express OR over two distinct rules: two
    copies of A give A, two copies of B give B.

    A single tree can, in the same two comparators:

        if rate > R            -> ATTACK
        else if ratio <= T     -> ATTACK
        else                   -> normal

    and a one-tree forest's majority is just that tree.
    """
    return {
        # node 0: rate <= R ?  yes -> node 1 (check timing), no -> ATTACK leaf
        # node 1: ratio <= T ? yes -> ATTACK leaf,           no -> normal leaf
        "feature": [rate_idx, ratio_idx, 0, 0, 0],
        "threshold": [int(rate_thr), int(ratio_thr), 0, 0, 0],
        "left": [1, 3, 0, 0, 0],
        "right": [2, 4, 0, 0, 0],
        "is_leaf": [0, 0, 1, 1, 1],
        "value": [0, 0, 1, 1, 0],
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

    trees = [or_tree(ri, rate_thr, di, ratio_thr, cols)]
    model = {
        "feature_set": args.set,
        "feature_names": cols,
        "n_trees": 1,
        "trees": trees,
        "cost": {"internal_nodes": 2, "leaves": 3, "max_depth": 2,
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

    # assert the tables really compute the OR, rather than trusting the shape
    from select_model import predict_tables
    probe = np.array([[0, 64], [ratio_thr, 64], [ratio_thr + 1, 64],
                      [64, rate_thr], [64, rate_thr + 1], [64, 64]],
                     dtype=np.int32)
    order = [cols.index("dt_ratio_q6"), cols.index("id_rate")]
    grid = np.zeros((len(probe), len(cols)), dtype=np.int32)
    grid[:, order[0]] = probe[:, 0]
    grid[:, order[1]] = probe[:, 1]
    got = predict_tables(model, grid)
    want = ((probe[:, 1] > rate_thr) | (probe[:, 0] <= ratio_thr)).astype(np.int8)
    bad = [(int(a), int(b), int(g), int(w))
           for (a, b), g, w in zip(probe, got, want) if g != w]
    assert not bad, (f"the frozen tables do not compute the stated rule: "
                     f"{bad} (dt_ratio_q6, id_rate, got, want)")
    print(f"  verified: the frozen tables reproduce that rule on "
          f"{len(probe)} boundary cases")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
