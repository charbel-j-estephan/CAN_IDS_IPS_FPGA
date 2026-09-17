"""Collect every sweep into one markdown report with the accuracy-cost fronts."""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

SET_NOTE = {
    "full": "every feature, including `can_id` and `id_known`",
    "timing": "no identity features, payload-content features included",
    "timing_only": "no identity features, no payload-content features",
    "paper": "the reference paper's two features only",
}


def front(x: pd.DataFrame) -> pd.DataFrame:
    """Pareto front: configs where nothing smaller is at least as accurate."""
    x = x.sort_values(["internal_nodes", "test_accuracy"],
                      ascending=[True, False])
    rows, best = [], -1.0
    for _, r in x.iterrows():
        if r.test_accuracy > best:
            rows.append(r)
            best = r.test_accuracy
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/REPORT.md")
    args = ap.parse_args()

    lines = ["# Random forest CAN IDS: accuracy versus hardware cost", ""]
    lines += [
        "`nodes` counts internal comparator nodes summed over the forest. Depth",
        "is tracked separately because it costs very little: every comparison",
        "depends only on the feature vector, so nothing about evaluating a tree",
        "is inherently sequential.",
        "",
    ]

    for path in sorted(glob.glob(os.path.join(args.results, "*", "sweep.csv"))):
        tag = os.path.basename(os.path.dirname(path))
        d = pd.read_csv(path)
        lines += [f"## dataset: `{tag}`", ""]
        for s in ["timing", "timing_only", "paper", "full"]:
            x = d[d.feature_set == s]
            if x.empty:
                continue
            f = front(x)
            lines += [f"### feature set `{s}` — {SET_NOTE.get(s, '')}", ""]
            lines += ["| trees | depth | nodes | accuracy | F1 | FP | FN | split on |",
                      "|---|---|---|---|---|---|---|---|"]
            for _, r in f.iterrows():
                lines.append(
                    f"| {int(r.n_trees)} | {int(r.max_depth)} | "
                    f"{int(r.internal_nodes)} | {r.test_accuracy * 100:.4f}% | "
                    f"{r.test_f1 * 100:.4f}% | {int(r.test_fp)} | "
                    f"{int(r.test_fn)} | {r.features_used} |"
                )
            lines.append("")

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
