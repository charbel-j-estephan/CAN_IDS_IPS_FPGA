"""Generate a stand-in trace in exact HCRL DoS_dataset.csv format.

This exists only because the real HCRL file cannot be downloaded from this
sandbox. It is built to match the documented properties of DoS_dataset.csv so
the pipeline below is exercised on realistic data:

  * Hyundai YF Sonata style ID set, 27 periodic IDs at 10/20/50/100/200/1000 ms
  * payloads with rolling counters, random-walk sensor bytes and a checksum byte,
    which is what makes per-ID Hamming distance small and stable
  * DoS floods of CAN ID 0x0000 with an all-zero payload every 0.3 ms
  * flood bursts of 3 to 5 seconds, about 16 percent of all frames injected
  * a 500 kbit/s arbitration model, so a flood of the highest priority ID 0x000
    delays the legitimate frames around it. Those delayed frames stay labelled
    normal, exactly as in the real capture. Without this the problem would be
    far easier than reality.

Swap in the real DoS_dataset.csv and nothing downstream changes.
"""

from __future__ import annotations

import argparse
import numpy as np

T0 = 1478198376.389427          # same epoch the real capture starts at
FRAME_TIME = 0.000262           # 131 bits at 500 kbit/s, worst case stuffing
DOS_PERIOD = 0.0003             # documented 0.3 ms injection period
DOS_ID = 0x000

# id: period in seconds
NORMAL_IDS = {
    0x0316: 0.010, 0x018F: 0.010, 0x0260: 0.010, 0x02A0: 0.010,
    0x0329: 0.010, 0x0545: 0.010, 0x02C0: 0.010, 0x0153: 0.010,
    0x01F1: 0.010, 0x0140: 0.010, 0x04B1: 0.010, 0x0130: 0.010,
    0x0220: 0.020, 0x02B0: 0.020, 0x0350: 0.020, 0x043F: 0.020,
    0x0440: 0.020, 0x04F0: 0.020, 0x0131: 0.020, 0x0251: 0.020,
    0x00A0: 0.050, 0x00A1: 0.050, 0x0510: 0.050, 0x0517: 0.050,
    0x05F0: 0.100, 0x0690: 0.200, 0x057F: 1.000,
}

DLC = {i: 8 for i in NORMAL_IDS}
DLC[0x0130] = 7
DLC[0x0517] = 3
DLC[0x0690] = 4
DLC[0x057F] = 2


# Per-ID payload behaviour class. Real CAN traffic is a mix, and the mix is what
# makes the problem non-trivial: frames that repeat their payload byte for byte
# give a Hamming distance of zero, which is exactly what an all-zero DoS payload
# also gives. An IDS cannot separate the two on Hamming distance alone.
COUNTER, STATIC, SENSOR = 0, 1, 2
BEHAVIOUR = {
    cid: [COUNTER, STATIC, SENSOR][i % 3]
    for i, cid in enumerate(sorted(NORMAL_IDS))
}


def _payloads(can_id: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Per-ID payload stream, returned as (n, 8) uint8."""
    p = np.zeros((n, 8), dtype=np.uint8)
    kind = BEHAVIOUR[can_id]

    p[:, 4] = np.uint8(can_id & 0xFF)
    p[:, 5] = np.uint8((can_id >> 4) & 0xFF)

    if kind == COUNTER:
        # Rolling alive-counter, changes every single frame.
        p[:, 0] = (np.arange(n) % 16).astype(np.uint8) | (can_id & 0xF0)
        step = rng.integers(-6, 7, size=n)
        walk = np.clip(2048 + np.cumsum(step), 0, 65535).astype(np.uint16)
        p[:, 1] = (walk >> 8).astype(np.uint8)
        p[:, 2] = (walk & 0xFF).astype(np.uint8)
        p[:, 6] = ((np.arange(n) // 64) % 256).astype(np.uint8)

    elif kind == STATIC:
        # Discrete status frame. Holds one value for long stretches, so most
        # consecutive pairs are byte-for-byte identical.
        events = rng.random(n) < 0.004
        state = np.cumsum(events)
        p[:, 0] = (state % 256).astype(np.uint8)
        p[:, 1] = ((state * 37) % 256).astype(np.uint8)
        p[:, 3] = ((state * 11) % 64).astype(np.uint8) << 2

    else:  # SENSOR
        # Quantised analogue reading: it moves, but only every few frames, so
        # roughly two thirds of consecutive pairs repeat exactly.
        step = rng.integers(-3, 4, size=n)
        walk = np.clip(512 + np.cumsum(step), 0, 4095).astype(np.uint16)
        coarse = (walk >> 4).astype(np.uint16)
        p[:, 0] = (coarse >> 4).astype(np.uint8)
        p[:, 1] = ((coarse & 0x0F) << 4).astype(np.uint8)
        p[:, 6] = ((np.arange(n) // 256) % 256).astype(np.uint8)

    # XOR checksum over the payload, so it tracks whatever moved. For a STATIC
    # frame that held its value the checksum holds too and the repeat is exact.
    p[:, 7] = np.bitwise_xor.reduce(p[:, :7], axis=1)
    return p


DIAG_ID = 0x7E8          # OBD-II response ID, sends ISO-TP multiframe bursts
DIAG_BURST = 8           # consecutive frames per transfer
DIAG_PERIOD = 2.0        # a transfer every ~2 s


def _diag_stream(duration_s: float, rng: np.random.Generator):
    """ISO-TP style diagnostic traffic: bursts of consecutive same-ID frames.

    This is the reason a burst counter alone cannot flag a flood. Real buses
    carry back-to-back frames of one ID during every diagnostic transfer, and
    those frames are normal.
    """
    n_transfers = int(duration_s / DIAG_PERIOD)
    starts = np.arange(n_transfers) * DIAG_PERIOD + rng.uniform(
        0, DIAG_PERIOD * 0.5, size=n_transfers
    )
    t = (starts[:, None] + np.arange(DIAG_BURST)[None, :] * FRAME_TIME).ravel()
    n = len(t)
    p = np.zeros((n, 8), dtype=np.uint8)
    p[:, 0] = (0x20 | (np.arange(n) % DIAG_BURST)).astype(np.uint8)
    p[:, 1:] = rng.integers(0, 256, size=(n, 7), dtype=np.uint8)
    return t, p


def build(duration_s: float, attack_fraction: float, seed: int,
          attack_id: int = DOS_ID, stealth: bool = False):
    rng = np.random.default_rng(seed)

    # --- legitimate periodic traffic -------------------------------------
    times, ids, dlcs, data, labels = [], [], [], [], []
    for can_id, period in NORMAL_IDS.items():
        n = int(duration_s / period)
        base = np.arange(n) * period
        jitter = rng.normal(0.0, period * 0.015, size=n)
        t = np.clip(base + jitter, 0.0, None)
        t.sort()
        times.append(t)
        ids.append(np.full(n, can_id, dtype=np.int64))
        dlcs.append(np.full(n, DLC[can_id], dtype=np.int64))
        data.append(_payloads(can_id, n, rng))
        labels.append(np.zeros(n, dtype=np.int8))

    d_t, d_p = _diag_stream(duration_s, rng)
    times.append(d_t)
    ids.append(np.full(len(d_t), DIAG_ID, dtype=np.int64))
    dlcs.append(np.full(len(d_t), 8, dtype=np.int64))
    data.append(d_p)
    labels.append(np.zeros(len(d_t), dtype=np.int8))

    # --- DoS flood bursts -------------------------------------------------
    n_legit = sum(len(t) for t in times)
    attack_budget = int(n_legit * attack_fraction / (1.0 - attack_fraction))
    burst_seconds = rng.uniform(3.0, 5.0, size=400)
    per_burst = (burst_seconds / DOS_PERIOD).astype(int)
    keep = np.cumsum(per_burst) <= attack_budget
    burst_seconds = burst_seconds[keep]
    per_burst = per_burst[keep]

    # spread bursts over the trace, never in the first 60 s so the model has
    # clean traffic to learn the per-ID baselines from
    starts = np.sort(rng.uniform(60.0, duration_s - 10.0, size=len(per_burst)))
    a_times = []
    for start, count in zip(starts, per_burst):
        a_times.append(start + np.arange(count) * DOS_PERIOD)
    a_times = np.concatenate(a_times) if a_times else np.zeros(0)

    times.append(a_times)
    ids.append(np.full(len(a_times), attack_id, dtype=np.int64))
    dlcs.append(np.full(len(a_times), 8, dtype=np.int64))
    if stealth and attack_id in NORMAL_IDS:
        # Worst case attacker: replay a payload that is structurally valid for
        # the ID being flooded. Hamming distance, payload population count and
        # the learned payload invariant all go blind, so only timing features
        # remain. This is the case that decides whether the detector is real.
        victim = _payloads(attack_id, max(len(a_times), 1), rng)
        data.append(victim[: len(a_times)])
    else:
        data.append(np.zeros((len(a_times), 8), dtype=np.uint8))
    labels.append(np.ones(len(a_times), dtype=np.int8))

    t = np.concatenate(times)
    can_id = np.concatenate(ids)
    dlc = np.concatenate(dlcs)
    payload = np.concatenate(data)
    label = np.concatenate(labels)

    # --- arbitration: ID 0x000 always wins, everything else queues --------
    # sort by desired time, attack frames a hair earlier on ties
    # the flooded ID wins arbitration against anything of lower priority
    desired = t - np.where(label == 1, 1e-9, 0.0)
    order = np.argsort(desired, kind="stable")
    t, can_id, dlc, payload, label = (
        t[order], can_id[order], dlc[order], payload[order], label[order]
    )
    d = t.copy()
    idx = np.arange(len(d), dtype=np.float64) * FRAME_TIME
    actual = np.maximum.accumulate(d - idx) + idx
    t = actual

    return T0 + t, can_id, dlc, payload, label


def write_csv(path: str, t, can_id, dlc, payload, label) -> None:
    hexb = np.array([f"{v:02x}" for v in range(256)])
    cols = [np.char.mod("%.6f", t), np.char.mod("%04x", can_id),
            np.char.mod("%d", dlc)]
    byte_str = hexb[payload]
    flag = np.where(label == 1, "T", "R")

    with open(path, "w", newline="") as fh:
        n = len(t)
        chunk = 200_000
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            lines = []
            for i in range(lo, hi):
                k = dlc[i]
                lines.append(
                    cols[0][i] + "," + cols[1][i] + "," + cols[2][i] + ","
                    + ",".join(byte_str[i, :k]) + "," + flag[i]
                )
            fh.write("\n".join(lines))
            fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=1540.0)
    ap.add_argument("--attack-fraction", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--stealth", action="store_true",
        help="inject structurally valid payloads for the flooded ID instead of "
             "all zeros, which blinds every payload based feature",
    )
    ap.add_argument(
        "--attack-id", default="0x000",
        help="CAN ID the flood uses. 0x000 matches the real DoS capture. "
             "Point it at a legitimate ID (e.g. 0x316) for the harder case "
             "where the flooded ID has a learned baseline.",
    )
    args = ap.parse_args()

    t, can_id, dlc, payload, label = build(
        args.duration, args.attack_fraction, args.seed,
        attack_id=int(args.attack_id, 16), stealth=args.stealth,
    )
    write_csv(args.out, t, can_id, dlc, payload, label)
    n = len(t)
    print(f"frames          {n}")
    print(f"normal          {int((label == 0).sum())}")
    print(f"injected        {int((label == 1).sum())} "
          f"({100.0 * label.mean():.2f} %)")
    print(f"duration        {t[-1] - t[0]:.1f} s")
    print(f"mean bus rate   {n / (t[-1] - t[0]):.0f} frames/s")
    print(f"unique ids      {len(np.unique(can_id))}")


if __name__ == "__main__":
    main()
