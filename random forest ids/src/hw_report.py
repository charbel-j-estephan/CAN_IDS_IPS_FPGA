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
    print("AREA")
    print("  Measured, not estimated. Run synthesis and parse it with")
    print("  src/syn_report.py rather than trusting the figures below:")
    print()
    print("    cd rtl/<tag> && yosys syn_xil.ys && \\")
    print("      python3 ../../src/syn_report.py --log syn_xil.log")
    print()
    print("  Last measured on the recommended 3-tree model, xc7:")
    print("    LUTs                810   (forest 25, extractor 785)")
    print("    flip-flops          322")
    print("    BRAM18 equivalents   33   (11 x RAMB36 + 11 x RAMB18)")
    print("    DSP48E1               2   (interval reciprocal, rate bucket)")
    print("    CARRY4               60")
    print()
    print("  The forest is 3.1 % of the LUTs. The trees are nearly free; the")
    print("  feature extractor and its per-ID memories are the whole design.")
    print()
    print("  Synthesise against a real architecture. Generic `synth` has no")
    print("  block-RAM primitive and maps the 2048-entry per-ID arrays to")
    print("  flip-flops: 458 128 cells and 221 623 registers, which measures")
    print("  the target, not the design.")
    print()
    print("  Model-dependent part of the cost:")
    print(f"    comparator nodes  {nodes}")
    print(f"    comparator bits   {cmp_bits}")
    print(f"    leaves            {model['cost']['leaves']}")
    print()
    print("  Memory is dominated by the 2048-entry direct-mapped ID table:")
    print(f"    per-ID state RAM  {state_kbit:.1f} kbit "
          f"({N_IDS} IDs x {STATE_BITS} bit)")
    print(f"    baseline ROM      {rom_kbit:.1f} kbit "
          f"({N_IDS} IDs x {ROM_BITS} bit)")
    print("  Real captures use 27 IDs, so most of that table is empty. An")
    print("  ID-to-slot index ROM plus a 32-entry table would cut it to about")
    print("  2 BRAM18, at the cost of one more pipeline cycle.")
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
