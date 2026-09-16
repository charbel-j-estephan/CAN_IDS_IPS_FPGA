"""Turn the frozen model into an FPGA area and timing budget.

Latency is counted in clock cycles, not estimated. The feature extractor is a
fixed 3-cycle sequence and the forest registers its verdict one cycle later, so
every frame costs exactly 4 cycles regardless of how the trees branch. That
determinism is the point: there is no data-dependent worst case to argue about.
"""

from __future__ import annotations

import argparse
import json

CLK_MHZ = 100.0
FEATURE_CYCLES = 3      # S_IDLE latch, S_READ fetch, S_CALC compute
FOREST_CYCLES = 1       # combinational forest, registered output
N_IDS = 2048

# per-ID memory, bits
STATE_BITS = 32 + 64 + 1            # last_ts, last_payload, seen
ROM_BITS = 20 + 16 + 7 + 1 + 64 + 64  # mean_int, recip, mean_hd, known, mask, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model.json")
    ap.add_argument("--clk", type=float, default=CLK_MHZ)
    ap.add_argument("--budget-us", type=float, default=1000.0)
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)

    nodes = model["cost"]["internal_nodes"]
    depth = model["cost"]["max_depth"]
    trees = model["n_trees"]
    widths = dict(zip(model["feature_names"], model["feature_widths"]))

    cycles = FEATURE_CYCLES + FOREST_CYCLES
    latency_us = cycles / args.clk
    margin = args.budget_us / latency_us

    # one magnitude comparator per internal node, sized by its feature
    cmp_bits = 0
    for tree in model["trees"]:
        for n, leaf in enumerate(tree["is_leaf"]):
            if not leaf:
                cmp_bits += widths[tree["feature_names"][tree["feature"][n]]]

    state_kbit = N_IDS * STATE_BITS / 1024.0
    rom_kbit = N_IDS * ROM_BITS / 1024.0

    print("FOREST")
    print(f"  trees                 {trees}")
    print(f"  comparator nodes      {nodes}")
    print(f"  leaves                {model['cost']['leaves']}")
    print(f"  max depth             {depth}")
    print(f"  features split on     {model['cost']['n_features_used']} of "
          f"{len(model['feature_names'])}")
    print()
    print("LATENCY  (fixed, not data dependent)")
    print(f"  feature extraction    {FEATURE_CYCLES} cycles")
    print(f"  forest + voter        {FOREST_CYCLES} cycle")
    print(f"  total per frame       {cycles} cycles = "
          f"{latency_us * 1000:.0f} ns at {args.clk:.0f} MHz")
    print(f"  budget                {args.budget_us:.0f} us per frame")
    print(f"  margin                {margin:,.0f}x")
    print()
    print("THROUGHPUT")
    peak = 1e6 / latency_us
    print(f"  detector ceiling      {peak:,.0f} frames/s")
    print(f"  CAN 1 Mbit/s ceiling  {1e6 / 47:,.0f} frames/s "
          f"(47 bit minimum frame)")
    print(f"  headroom vs the bus   {peak / (1e6 / 47):,.0f}x")
    print()
    print("AREA  (pre synthesis estimate)")
    print(f"  comparator bits       {cmp_bits}  "
          f"(~{cmp_bits} LUT6 for the compares)")
    print(f"  mux tree              ~{model['cost']['leaves']} LUT6")
    print(f"  popcount units        2 x 64-bit adder tree, ~120 LUT6")
    print(f"  reciprocal multiply   1 DSP48 (20x16)")
    print(f"  per-ID state RAM      {state_kbit:.1f} kbit "
          f"({N_IDS} IDs x {STATE_BITS} bit)")
    print(f"  baseline ROM          {rom_kbit:.1f} kbit "
          f"({N_IDS} IDs x {ROM_BITS} bit)")
    print(f"  total memory          {(state_kbit + rom_kbit) / 8:.1f} kB, "
          f"fits in {int((state_kbit + rom_kbit) / 18) + 1} x 18 kbit BRAM")
    print()
    m = model.get("test_metrics", {})
    if m:
        print("ACCURACY  (held-out test truncation)")
        print(f"  accuracy              {m['accuracy'] * 100:.4f} %")
        print(f"  f1                    {m['f1'] * 100:.4f} %")
        print(f"  recall                {m['recall'] * 100:.4f} %  "
              f"({m['fn']} missed attacks)")
        print(f"  precision             {m['precision'] * 100:.4f} %  "
              f"({m['fp']} false alarms)")
        print(f"  false positive rate   {m['fpr'] * 100:.6f} %")


if __name__ == "__main__":
    main()
