"""Parse a yosys statistics log into a resource table.

Run synthesis against a real architecture, not the generic target. Generic
`synth` has no block-RAM primitive, so it maps the 2048-entry per-ID arrays to
flip-flops: 458 128 cells and 221 623 registers, which says nothing about the
design and everything about the target. `synth_xilinx -family xc7` infers the
memories properly.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys

LUT_RE = re.compile(r"LUT[1-6]$")


def parse(path: str) -> dict:
    txt = open(path).read()
    # yosys prints statistics more than once; keep only the last block, and
    # stop at the hierarchy summary so modules are not counted twice
    marks = [m.start() for m in re.finditer(r"Printing statistics", txt)]
    blk = txt[marks[-1]:]
    end = blk.find("=== design hierarchy ===")
    if end != -1:
        blk = blk[:end]

    mods, cur = {}, None
    for line in blk.splitlines():
        m = re.match(r"=== (\S+) ===", line.strip())
        if m:
            cur = m.group(1)
            mods.setdefault(cur, collections.Counter())
            continue
        if cur is None:
            continue
        m = re.match(r"\s{5}(\S+)\s+(\d+)\s*$", line)
        if m:
            mods[cur][m.group(1)] += int(m.group(2))
    return mods


def summarise(cells: collections.Counter) -> dict:
    return {
        "luts": sum(v for k, v in cells.items() if LUT_RE.match(k)),
        "ffs": sum(v for k, v in cells.items() if k.startswith("FD")),
        "ramb36": cells.get("RAMB36E1", 0),
        "ramb18": cells.get("RAMB18E1", 0),
        "dsp": cells.get("DSP48E1", 0),
        "carry4": cells.get("CARRY4", 0),
        "distram": cells.get("RAM256X1S", 0),
        "muxf": cells.get("MUXF7", 0) + cells.get("MUXF8", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    mods = parse(args.log)
    leaf = {k: v for k, v in mods.items()
            if v and not k.startswith("can_ids_top")}
    if not leaf:
        sys.exit("no leaf module statistics found in the log")

    rows = []
    total = collections.Counter()
    for name, cells in leaf.items():
        short = "can_ids_features" if "can_ids_features" in name else name
        rows.append((short, summarise(cells)))
        total.update(cells)
    rows.append(("TOTAL", summarise(total)))

    hdr = (f"{'module':<18}{'LUT':>7}{'FF':>7}{'RAMB36':>8}{'RAMB18':>8}"
           f"{'DSP48':>7}{'CARRY4':>8}{'MUXF':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in rows:
        print(f"{name:<18}{s['luts']:>7}{s['ffs']:>7}{s['ramb36']:>8}"
              f"{s['ramb18']:>8}{s['dsp']:>7}{s['carry4']:>8}{s['muxf']:>6}")

    t = summarise(total)
    bram18 = 2 * t["ramb36"] + t["ramb18"]
    print()
    print(f"BRAM18 equivalents  {bram18}  ({bram18 * 18} kbit of capacity)")
    print(f"distributed RAM     {t['distram']} x RAM256X1S")
    forest = next((s for n, s in rows if n == "rf_forest"), None)
    if forest and t["luts"]:
        print(f"forest share of LUTs  {forest['luts']} of {t['luts']} "
              f"({100.0 * forest['luts'] / t['luts']:.1f} %)")


if __name__ == "__main__":
    main()
