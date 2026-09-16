"""Emit the per-ID baseline ROM contents that can_ids_features.v loads.

Four $readmemh files, 2048 entries each, one per 11-bit CAN ID:
  mean_interval.mem  20-bit median inter-arrival time, microseconds
  recip.mem          16-bit reciprocal, so the RTL divides with a multiply
  mean_hamming.mem   7-bit median Hamming distance between consecutive payloads
  id_known.mem       1 if the ID appeared in clean training traffic
  const_mask.mem     64-bit mask of payload bits that never move in clean traffic
  const_val.mem      64-bit value those invariant bits hold

An ID that never appeared clean gets all zeros, which is what the Python
feature extractor assumes too, so model and RTL agree on unknown IDs.
"""

from __future__ import annotations

import argparse
import json
import os

RECIP_SHIFT = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/baseline.json")
    ap.add_argument("--rtl-dir", default="rtl")
    args = ap.parse_args()

    with open(args.baseline) as fh:
        bl = json.load(fh)
    mean_interval = {int(k): int(v) for k, v in bl["mean_interval_us"].items()}
    mean_hamming = {int(k): int(v) for k, v in bl["mean_hamming"].items()}
    const_mask = {int(k): int(v, 16) for k, v in bl.get("const_mask", {}).items()}
    const_val = {int(k): int(v, 16) for k, v in bl.get("const_val", {}).items()}

    n = 2048
    rows = []
    for cid in range(n):
        mi = mean_interval.get(cid, 0)
        rc = int(round((1 << RECIP_SHIFT) * 64.0 / mi)) if mi > 0 else 0
        rc = min(rc, 0xFFFF)
        rows.append((mi & 0xFFFFF, rc, mean_hamming.get(cid, 0) & 0x7F,
                     1 if cid in mean_interval else 0,
                     const_mask.get(cid, 0), const_val.get(cid, 0)))

    os.makedirs(args.rtl_dir, exist_ok=True)
    specs = [
        ("mean_interval.mem", 0, 5),
        ("recip.mem", 1, 4),
        ("mean_hamming.mem", 2, 2),
        ("id_known.mem", 3, 1),
        ("const_mask.mem", 4, 16),
        ("const_val.mem", 5, 16),
    ]
    for fname, col, digits in specs:
        path = os.path.join(args.rtl_dir, fname)
        with open(path, "w") as fh:
            for r in rows:
                fh.write(f"{r[col]:0{digits}x}\n")
        print(f"wrote {path}  ({n} entries)")

    known = sum(r[3] for r in rows)
    print(f"known IDs: {known}")
    over = [cid for cid, mi in mean_interval.items()
            if int(round((1 << RECIP_SHIFT) * 64.0 / mi)) > 0xFFFF]
    if over:
        print(f"WARNING: reciprocal saturated for IDs {[hex(c) for c in over]} "
              f"(interval below {(1 << RECIP_SHIFT) * 64 // 0xFFFF} us)")


if __name__ == "__main__":
    main()
