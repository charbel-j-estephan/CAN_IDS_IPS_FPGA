"""Sweep forest geometry and report the accuracy versus model-size front.

Size is measured as the total number of internal (comparator) nodes summed over
all trees, not as tree count, because that is what actually grows: a forest of
many shallow trees can be smaller than one deep tree. Depth is tracked
separately since it is close to free.

Tree counts are kept odd so the majority vote cannot tie.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_SETS                      # noqa: E402

def majority_predict(clf, X) -> np.ndarray:
    """Hard majority vote over the trees, which is what the hardware does.

    RandomForestClassifier.predict averages per-tree class *probabilities* and
    then takes the argmax. This project scores a hard majority vote instead:
    each tree gives one bit and the bits are counted. The two agree on shallow,
    cleanly separated trees and diverge once leaves are mixed, so the rule used
    to score has to be the rule the frozen model uses.
    """
    votes = np.zeros(len(X), dtype=np.int32)
    for est in clf.estimators_:
        votes += est.predict(X).astype(np.int32)
    return (votes * 2 > len(clf.estimators_)).astype(np.int8)


def forest_cost(clf) -> dict:
    internal = 0
    leaves = 0
    depth = 0
    used = set()
    for est in clf.estimators_:
        t = est.tree_
        is_leaf = t.children_left == -1
        internal += int((~is_leaf).sum())
        leaves += int(is_leaf.sum())
        depth = max(depth, int(t.max_depth))
        used.update(int(f) for f in t.feature[~is_leaf])
    return {
        "internal_nodes": internal,
        "leaves": leaves,
        "max_depth": depth,
        "features_used": sorted(used),
        "n_features_used": len(used),
    }


def score(y_true, y_pred) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache.npz")
    ap.add_argument("--out", default="results/sweep.csv")
    ap.add_argument("--trees", default="1,3,5,7,9")
    ap.add_argument("--depths", default="2,3,4,5,6,7,8")
    ap.add_argument("--sets", default="full,timing,paper")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    columns = list(z["columns"])
    Xtr_all, ytr = z["Xtr"], z["ytr"]
    Xte_all, yte = z["Xte"], z["yte"]

    trees = [int(v) for v in args.trees.split(",")]
    depths = [int(v) for v in args.depths.split(",")]
    sets = args.sets.split(",")

    rows = []
    for set_name in sets:
        cols = FEATURE_SETS[set_name]
        idx = [columns.index(c) for c in cols]
        Xtr, Xte = Xtr_all[:, idx], Xte_all[:, idx]

        for n_trees, depth, max_feat in itertools.product(
            trees, depths, ["sqrt", None]
        ):
            clf = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=depth,
                max_features=max_feat,
                min_samples_leaf=50,
                bootstrap=True,
                random_state=0,
                n_jobs=-1,
            )
            clf.fit(Xtr, ytr)
            cost = forest_cost(clf)
            m_te = score(yte, majority_predict(clf, Xte))
            m_tr = score(ytr, majority_predict(clf, Xtr))
            rows.append(
                {
                    "feature_set": set_name,
                    "n_trees": n_trees,
                    "max_depth_cfg": depth,
                    "max_features": "sqrt" if max_feat == "sqrt" else "all",
                    **cost,
                    "features_used": ",".join(cols[i] for i in cost["features_used"]),
                    "test_accuracy": m_te["accuracy"],
                    "test_f1": m_te["f1"],
                    "test_recall": m_te["recall"],
                    "test_precision": m_te["precision"],
                    "test_fpr": m_te["fpr"],
                    "test_fp": m_te["fp"],
                    "test_fn": m_te["fn"],
                    "train_accuracy": m_tr["accuracy"],
                }
            )
            print(
                f"{set_name:7s} trees={n_trees} depth={depth} feat={str(max_feat):5s} "
                f"nodes={cost['internal_nodes']:4d} "
                f"acc={m_te['accuracy'] * 100:7.4f}% f1={m_te['f1'] * 100:7.4f}% "
                f"fp={m_te['fp']:6d} fn={m_te['fn']:6d}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} configurations)")

    print("\nsmallest configuration reaching each accuracy target:")
    for target in [0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]:
        ok = out[out.test_accuracy >= target]
        if ok.empty:
            print(f"  >= {target * 100:8.4f}%  none")
            continue
        best = ok.sort_values(
            ["internal_nodes", "max_depth", "n_trees"]
        ).iloc[0]
        print(
            f"  >= {target * 100:8.4f}%  {best.feature_set:7s} "
            f"trees={best.n_trees} depth={best.max_depth} "
            f"nodes={best.internal_nodes:4d} acc={best.test_accuracy * 100:.4f}% "
            f"fp={best.test_fp} fn={best.test_fn}"
        )


if __name__ == "__main__":
    main()
