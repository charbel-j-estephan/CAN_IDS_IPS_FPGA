"""Generate the dashboard page from measured results, never by hand.

The page went stale twice while it was hand-edited: each time the model was
retrained the HTML kept quoting the superseded numbers, which is worse than no
page at all because it reads as current. So the numbers now come from
viz_data.json every time the page is built, and the HTML holds only prose and
layout, with one `__DATA__` slot.

Anything the page states as a figure has to arrive through here. If a claim
cannot be produced from the results files, it does not belong on the page.
"""

from __future__ import annotations

import argparse
import json
import os

# the compact keys the page's renderer reads
def pt(p: dict) -> dict:
    return {"n": p["nodes"], "a": p["accuracy"], "r": p["recall"],
            "fp": p["fp"], "fn": p["fn"], "t": p["trees"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", default="results/viz_data.json")
    ap.add_argument("--template", default="viz/dashboard.template.html")
    ap.add_argument("--out", default="results/dashboard.html")
    ap.add_argument("--model", default="mixed",
                    help="which trained model the page presents as deployed")
    args = ap.parse_args()

    with open(args.viz) as fh:
        v = json.load(fh)

    D = {"fronts": {}, "models": {}, "cross": {}, "windows": {}}

    for tag, f in v["fronts"].items():
        D["fronts"][tag] = {
            "label": f["label"],
            "sets": {s: {"label": d["label"], "points": [pt(p) for p in d["points"]]}
                     for s, d in f["sets"].items()},
        }

    for tag, m in v["models"].items():
        D["models"][tag] = {
            "label": m["label"], "trees": m["trees"], "nodes": m["nodes"],
            "depth": m["depth"], "features": m["features"],
            "metrics": m["metrics"], "tree_tables": m["tree_tables"],
            "feature_names": m["feature_names"],
        }

    for k in ("cross_eval_from_real", "cross_eval_real", "cross_eval"):
        if k in v["cross_eval"]:
            D["cross"][k] = v["cross_eval"][k]

    # The alarm table shows the DEPLOYED model over every trace, so it cannot
    # drift away from what is actually shipped.
    # The shipped detector's results come straight from final_report.py. The
    # trained-forest tables below it are kept as the comparison that justifies
    # not shipping one.
    if "final" in v:
        D["final"] = v["final"]
    if "adversarial" in v:
        D["adversarial"] = v["adversarial"]

    fw = v.get("firmware", {})
    dep = fw.get(args.model)
    if dep is None:
        raise SystemExit(f"no firmware results for model {args.model!r}; "
                         f"have {list(fw)}")
    for tag in v.get("firmware_order", list(dep["traces"])):
        if tag not in dep["traces"]:
            continue
        tr = dep["traces"][tag]
        D["windows"][tag] = {"label": tr["label"], "rows": tr["thresholds"]}
    D["window_order"] = [t for t in v.get("firmware_order", [])
                         if t in D["windows"]]
    D["deployed"] = args.model

    # the superseded model over the same traces, which is the overfitting story
    other = [k for k in fw if k != args.model]
    if other:
        o = other[0]
        D["compare"] = {
            "label": fw[o]["label"], "key": o,
            "traces": {t: fw[o]["traces"][t]["thresholds"]
                       for t in fw[o]["traces"]},
        }
        D["compare_deployed_label"] = dep["label"]

    if "rate_floor" in v:
        D["rate_floor"] = v["rate_floor"].get(args.model, [])
        D["rate_floor_other"] = (v["rate_floor"].get(other[0], [])
                                 if other else [])

    # the independent-dataset alarm result keeps its own thresholds
    if "windows" in v and "syncan" in v["windows"]:
        D["syncan_windows"] = v["windows"]["syncan"]

    html = open(args.template).read()
    if "__DATA__" not in html:
        raise SystemExit(f"{args.template} has no __DATA__ slot")
    out = html.replace("__DATA__", json.dumps(D, separators=(",", ":")))
    with open(args.out, "w") as fh:
        fh.write(out)

    kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({kb:.0f} kB)")
    print(f"  deployed model  {args.model}: {D['models'][args.model]['trees']} "
          f"trees, {D['models'][args.model]['nodes']} comparators")
    print(f"  alarm traces    {D['window_order']}")
    if "final" in D:
        t = D["final"]["totals"]
        print(f"  shipped         {D['final']['comparators']} comparators, "
              f"{t['detected']}/{t['windows']} windows, "
              f"{t['false_alarms']} false alarms")
    for a in D.get("adversarial", []):
        print(f"    {a['label']:34s} {a['detected']}/{a['windows']}")
    tot = {}
    for t in D["window_order"]:
        for r in D["windows"][t]["rows"]:
            a = tot.setdefault(r["m"], [0, 0, 0])
            a[0] += r["detected"]; a[1] += r["windows"]; a[2] += r["false_alarms"]
    for m in sorted(tot):
        d, w, f = tot[m]
        print(f"    threshold {m}: {d}/{w} windows, {f} false alarms")


if __name__ == "__main__":
    main()
