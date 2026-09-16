"""Pick the smallest forest that meets the accuracy target, then freeze it.

Two selection policies.

  --policy smallest   accuracy >= target, then fewest comparator nodes, then
                      shallowest, then fewest trees. Gives the absolute minimum
                      hardware.

  --policy robust     accuracy >= target, then the most distinct features split
                      on, then fewest nodes, and at least 3 trees. This is the
                      default, because the smallest model on this data collapses
                      onto a single comparator. That scores perfectly and is a
                      bad thing to deploy: one feature carries the whole verdict,
                      so an attacker who defeats that one feature defeats the
                      IDS outright, and a single tree has no vote to lose. A few
                      extra comparators buy graceful degradation for free at
                      this scale.

The frozen model is written as integer node tables. Because every feature is an
integer, an sklearn split "x <= 2.5" is exactly "x <= 2", so the thresholds
convert to integers with zero loss. That equivalence is asserted here, not
assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_SETS                          # noqa: E402
from sweep import forest_cost, score                       # noqa: E402

# Bit widths the RTL uses for each feature, matching features.py saturation.
FEATURE_WIDTH = {
    "dt_id": 20, "dt_id_dev": 20, "dt_ratio_q6": 12, "hd": 7, "hd_dev": 7,
    "dt_bus": 16, "burst": 4, "id_known": 1, "can_id": 11, "dlc": 4,
    "pl_violation": 7, "pl_popcount": 7,
}


def tree_to_tables(est, feature_names):
    """Flatten an sklearn tree into integer node tables."""
    t = est.tree_
    n = t.node_count
    feature, threshold, left, right, is_leaf, value = [], [], [], [], [], []
    for i in range(n):
        leaf = t.children_left[i] == -1
        is_leaf.append(int(leaf))
        if leaf:
            feature.append(0)
            threshold.append(0)
            left.append(0)
            right.append(0)
            value.append(int(np.argmax(t.value[i][0])))
        else:
            feature.append(int(t.feature[i]))
            # integer-exact: x <= thr  <=>  x <= floor(thr) for integer x
            threshold.append(int(np.floor(t.threshold[i])))
            left.append(int(t.children_left[i]))
            right.append(int(t.children_right[i]))
            value.append(0)
    return {
        "feature": feature, "threshold": threshold, "left": left,
        "right": right, "is_leaf": is_leaf, "value": value,
        "feature_names": list(feature_names),
    }


def predict_tables(model, X) -> np.ndarray:
    """Reference integer implementation, the thing the RTL must match."""
    votes = np.zeros(len(X), dtype=np.int32)
    for tree in model["trees"]:
        f = np.asarray(tree["feature"])
        thr = np.asarray(tree["threshold"])
        lf = np.asarray(tree["left"])
        rt = np.asarray(tree["right"])
        leaf = np.asarray(tree["is_leaf"], dtype=bool)
        val = np.asarray(tree["value"])

        node = np.zeros(len(X), dtype=np.int32)
        active = ~leaf[node]
        while active.any():
            idx = np.flatnonzero(active)
            nd = node[idx]
            go_left = X[idx, f[nd]] <= thr[nd]
            node[idx] = np.where(go_left, lf[nd], rt[nd])
            active = ~leaf[node]
        votes += val[node]
    return (votes * 2 > len(model["trees"])).astype(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache.npz")
    ap.add_argument("--sweep", default="results/sweep.csv")
    ap.add_argument("--out", default="results/model.json")
    ap.add_argument("--target", type=float, default=0.999)
    ap.add_argument("--policy", default="robust",
                    choices=["robust", "smallest"])
    ap.add_argument(
        "--feature-set", default="timing",
        help="timing excludes can_id and id_known, so the model cannot fall "
             "back on an ID whitelist and still works when a flood reuses a "
             "legitimate ID.",
    )
    args = ap.parse_args()

    import pandas as pd

    z = np.load(args.cache, allow_pickle=True)
    columns = list(z["columns"])
    cols = FEATURE_SETS[args.feature_set]
    idx = [columns.index(c) for c in cols]
    Xtr, ytr = z["Xtr"][:, idx], z["ytr"]
    Xte, yte = z["Xte"][:, idx], z["yte"]

    sweep = pd.read_csv(args.sweep)
    cand = sweep[
        (sweep.feature_set == args.feature_set)
        & (sweep.test_accuracy >= args.target)
    ]
    if cand.empty:
        best_acc = sweep[sweep.feature_set == args.feature_set].test_accuracy.max()
        raise SystemExit(
            f"no {args.feature_set} configuration reaches {args.target:.4%}; "
            f"best was {best_acc:.4%}. Lower --target or add features."
        )
    if args.policy == "smallest":
        pick = cand.sort_values(
            ["internal_nodes", "max_depth", "n_trees"]
        ).iloc[0]
    else:
        multi = cand[cand.n_trees >= 3]
        if multi.empty:
            multi = cand
        pick = multi.sort_values(
            ["n_features_used", "internal_nodes", "max_depth", "n_trees"],
            ascending=[False, True, True, True],
        ).iloc[0]
    print(f"policy={args.policy}  picked: trees={pick.n_trees} "
          f"depth_cfg={pick.max_depth_cfg} max_features={pick.max_features} "
          f"nodes={pick.internal_nodes} features={pick.n_features_used} "
          f"acc={pick.test_accuracy:.6%}")

    clf = RandomForestClassifier(
        n_estimators=int(pick.n_trees),
        max_depth=int(pick.max_depth_cfg),
        max_features="sqrt" if pick.max_features == "sqrt" else None,
        min_samples_leaf=50,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    ).fit(Xtr, ytr)

    model = {
        "feature_set": args.feature_set,
        "feature_names": cols,
        "feature_widths": [FEATURE_WIDTH[c] for c in cols],
        "n_trees": int(pick.n_trees),
        "trees": [tree_to_tables(e, cols) for e in clf.estimators_],
        "cost": forest_cost(clf),
    }

    # the integer tables must reproduce sklearn exactly, on both truncations
    for name, X in (("train", Xtr), ("test", Xte)):
        a = clf.predict(X).astype(np.int8)
        b = predict_tables(model, X)
        mismatch = int((a != b).sum())
        print(f"integer tables vs sklearn on {name}: {mismatch} mismatches")
        assert mismatch == 0, "integer threshold conversion is not exact"

    m = score(yte, predict_tables(model, Xte))
    model["test_metrics"] = m
    print(f"test accuracy  {m['accuracy']:.6%}")
    print(f"test f1        {m['f1']:.6%}")
    print(f"test recall    {m['recall']:.6%}   fn={m['fn']}")
    print(f"test precision {m['precision']:.6%}   fp={m['fp']}")
    print(f"test fpr       {m['fpr']:.8%}")
    print(f"internal nodes {model['cost']['internal_nodes']}  "
          f"depth {model['cost']['max_depth']}")

    with open(args.out, "w") as fh:
        json.dump(model, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
