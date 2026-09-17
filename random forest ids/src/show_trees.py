"""Print a frozen forest as readable decision trees.

`model.json` holds the trees as integer node tables, which is what the
reference predictor consumes but is not something you can read. This renders the
same thing as a tree you can follow by eye, and translates each threshold into
physical units so the numbers mean something.
"""

from __future__ import annotations

import argparse
import json

# What each feature counts, so a threshold reads as a statement about the bus
# rather than as a bare integer.
UNITS = {
    "dt_id": ("us", "microseconds since the previous frame of this ID"),
    "dt_id_dev": ("us", "how far this gap is from the ID's normal gap"),
    "dt_ratio_q6": ("1/64 of period", "64 means exactly on schedule"),
    "hd": ("bits", "payload bits that changed since this ID's last frame"),
    "hd_dev": ("bits", "how unusual that amount of change is for this ID"),
    "dt_bus": ("us", "microseconds since the previous frame on the bus"),
    "burst": ("frames", "consecutive frames in a row carrying this ID"),
    "dlc": ("bytes", "data length code"),
    "pl_violation": ("bits", "payload bits that break this ID's fixed pattern"),
    "pl_popcount": ("bits", "how many bits of the payload are set at all"),
    "id_rate": ("64 = normal", "sustained rate for this ID, 4095 means 64x"),
    "bus_rate": ("64 = normal", "sustained rate for the whole bus"),
    "can_id": ("id", "the raw 11-bit identifier"),
    "id_known": ("0 or 1", "1 if this ID appears in clean traffic"),
}


def interpret(feature: str, thr: int) -> str:
    """One plain sentence for what passing this test means."""
    if feature == "dt_ratio_q6":
        pct = 100.0 * (thr + 1) / 64.0
        return (f"arrived at under {pct:.0f} % of this ID's normal gap, "
                f"so far too early")
    if feature == "id_rate":
        mult = (thr + 1) / 64.0
        return f"sustained rate at or below {mult:.1f}x this ID's normal rate"
    if feature == "pl_popcount":
        return ("payload has no more than "
                f"{thr} bits set" + (" at all, i.e. all zeros" if thr == 0 else ""))
    if feature == "pl_violation":
        return f"no more than {thr} bits break this ID's fixed pattern"
    if feature in ("dt_id", "dt_id_dev", "dt_bus"):
        return f"at or below {thr} us ({thr / 1000.0:.2f} ms)"
    if feature == "burst":
        return f"at most {thr} frames of this ID back to back"
    return f"at or below {thr}"


def render(tree: dict, node: int, names: list, depth: int, lines: list,
           prefix: str = "") -> None:
    pad = "    " * depth
    if tree["is_leaf"][node]:
        verdict = "ATTACK" if tree["value"][node] == 1 else "normal"
        lines.append(f"{pad}{prefix}=> {verdict}")
        return
    f = names[tree["feature"][node]]
    thr = tree["threshold"][node]
    unit = UNITS.get(f, ("", ""))[0]
    lines.append(f"{pad}{prefix}if {f} <= {thr}   [{unit}]")
    lines.append(f"{pad}    ({interpret(f, thr)})")
    render(tree, tree["left"][node], names, depth + 1, lines, "yes: ")
    render(tree, tree["right"][node], names, depth + 1, lines, "no:  ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    with open(args.model) as fh:
        m = json.load(fh)
    names = m["feature_names"]
    n = m["n_trees"]
    majority = n // 2 + 1

    print(f"forest from {args.model}")
    print(f"  {n} trees, {m['cost']['internal_nodes']} comparator nodes, "
          f"depth {m['cost']['max_depth']}")
    print(f"  feature set: {m['feature_set']}")
    print()
    print("HOW A VERDICT IS REACHED")
    print("  Every tree sees the same features and votes ATTACK or normal.")
    print(f"  The frame is flagged when at least {majority} of the {n} trees "
          f"vote ATTACK.")
    print("  Tree counts are always odd so the vote cannot tie.")
    print("  Every comparison depends only on the feature vector, so nothing")
    print("  about the evaluation is sequential and depth costs very little.")
    print()
    print("THE FEATURES THESE TREES TEST")
    used = sorted({names[tree["feature"][i]]
                   for tree in m["trees"]
                   for i in range(len(tree["is_leaf"]))
                   if not tree["is_leaf"][i]})
    for f in used:
        unit, desc = UNITS.get(f, ("", ""))
        print(f"  {f:<14} {desc}")
    print()

    for i, tree in enumerate(m["trees"]):
        n_cmp = sum(1 for x in tree["is_leaf"] if not x)
        print("=" * 66)
        print(f"TREE {i}   ({n_cmp} comparators)")
        print("=" * 66)
        lines = []
        render(tree, 0, names, 0, lines)
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
