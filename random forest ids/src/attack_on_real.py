"""Inject a flood into HCRL's real attack-free capture.

The real DoS capture only contains the crude attack: a flood of CAN ID 0x000,
which has no baseline entry and is therefore separable by construction. It
cannot tell us what happens against an attacker who floods a legitimate ID.

HCRL's fuzzy, gear and RPM captures do exactly that, but the point of this
script is that the question can be answered with the files already here. It
takes `normal_run_data.txt`, which is genuine attack-free traffic from a real
vehicle, and injects a flood on top of it. So the background traffic, its
jitter, its payload behaviour and its ID mix are all real; only the attack is
synthetic, and the attack is the part whose parameters we actually know
(0.3 ms injection period, 3 to 5 second bursts, about 16 % of frames).

Three attack modes, in increasing difficulty:

  zero-id    flood ID 0x000 with an all-zero payload. Reproduces the real DoS
             capture's attack on top of this different background.
  valid-id   flood a legitimate high-rate ID with an all-zero payload. Removes
             the whitelist shortcut but leaves the payload degenerate.
  stealth    flood a legitimate ID by replaying that ID's own real payloads,
             sampled from the clean capture. Blinds every payload feature, so
             only timing and rate are left.
  lowrate    stealth, but paced at a multiple of the victim's own period
             instead of every 0.3 ms. --rate 2 doubles the ID's frame rate,
             which is enough to spoof a gauge while leaving the inter-arrival
             ratio well inside what normal jitter can look like. This is the
             honest worst case: a fast flood is loud, and an attacker who only
             needs their frames acted on has no reason to be loud.
  mixed      every burst picks its own rate from MIXED_RATES. The attacker's
             rate is a free parameter, so a model trained at one rate learns a
             threshold that only covers that rate: one trained on 33x floods
             misses a 2x flood entirely. Training across the range is what fixes
             that, and this is the mode to train on.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from can_data import load_hcrl_normal_txt                  # noqa: E402

FRAME_TIME = 0.000262      # 131 bits at 500 kbit/s
DOS_PERIOD = 0.0003        # documented injection period
ZERO_ID = 0x000
# Rates for `mixed`, as multiples of the victim ID's own frame rate. The top of
# the range is roughly what the documented 0.3 ms DoS achieves against a 10 ms
# ID; the bottom is a flood quiet enough to pass for jitter.
MIXED_RATES = (2.0, 4.0, 8.0, 16.0, 33.0)


def build(clean, mode: str, attack_id: int, attack_fraction: float, seed: int,
          rate: float = 1.0):
    rng = np.random.default_rng(seed)

    t = clean["timestamp"].to_numpy(np.float64)
    t = t - t[0]
    can_id = clean["can_id"].to_numpy(np.int64)
    dlc = clean["dlc"].to_numpy(np.int64)
    payload = clean["payload"].to_numpy(np.uint64)
    duration = float(t[-1])

    if mode == "zero-id":
        inj_id = ZERO_ID
    else:
        inj_id = attack_id

    # lowrate paces injections off the victim's own period rather than the
    # documented 0.3 ms, so the flood adds `rate` extra frames per real one
    period = DOS_PERIOD
    victim_period = None
    if mode in ("lowrate", "mixed"):
        vt = t[can_id == inj_id]
        if len(vt) < 10:
            raise SystemExit(f"ID {inj_id:#05x} is not in the capture")
        victim_period = float(np.median(np.diff(vt)))
        period = victim_period / max(rate, 1e-6)
        if mode == "lowrate":
            print(f"victim period {victim_period * 1000:.2f} ms, "
                  f"injecting every {period * 1000:.2f} ms "
                  f"({rate:g}x its rate)")
        else:
            print(f"victim period {victim_period * 1000:.2f} ms, bursts at "
                  + ", ".join(f"{r:g}x" for r in MIXED_RATES))

    # how many frames to inject for the requested share
    n_clean = len(t)
    budget = int(n_clean * attack_fraction / (1.0 - attack_fraction))

    burst_seconds = rng.uniform(3.0, 5.0, size=4000)
    per_burst = (burst_seconds / period).astype(np.int64)
    keep = np.cumsum(per_burst) <= budget
    per_burst = per_burst[keep]
    burst_seconds = burst_seconds[keep]
    if len(per_burst) == 0:
        raise SystemExit("capture too short for any attack burst")

    # never in the first 60 s, so the baseline has clean traffic to learn from
    starts = np.sort(rng.uniform(60.0, max(61.0, duration - 10.0),
                                 size=len(per_burst)))
    if mode == "mixed":
        burst_rates = rng.choice(MIXED_RATES, size=len(per_burst))
        burst_periods = victim_period / burst_rates
        # a slow burst spends its frame budget over more time, so cap each
        # burst's length in frames to keep the bursts comparable in duration
        per_burst = np.minimum(
            per_burst, (burst_seconds / burst_periods).astype(np.int64)
        )
        a_times = np.concatenate(
            [st + np.arange(c) * pp
             for st, c, pp in zip(starts, per_burst, burst_periods) if c > 0]
        )
    else:
        a_times = np.concatenate(
            [s + np.arange(c) * period for s, c in zip(starts, per_burst)]
        )
    n_a = len(a_times)

    if mode in ("stealth", "lowrate", "mixed"):
        # replay the victim ID's own real payloads, so the injected frames are
        # structurally indistinguishable from that ID's genuine traffic
        victim = payload[can_id == inj_id]
        if len(victim) == 0:
            raise SystemExit(f"ID {inj_id:#05x} does not appear in the capture")
        a_payload = victim[rng.integers(0, len(victim), size=n_a)]
        a_dlc = np.full(n_a, int(np.median(dlc[can_id == inj_id])), dtype=np.int64)
    else:
        a_payload = np.zeros(n_a, dtype=np.uint64)
        a_dlc = np.full(n_a, 8, dtype=np.int64)

    all_t = np.concatenate([t, a_times])
    all_id = np.concatenate([can_id, np.full(n_a, inj_id, dtype=np.int64)])
    all_dlc = np.concatenate([dlc, a_dlc])
    all_pl = np.concatenate([payload, a_payload])
    all_lb = np.concatenate([np.zeros(len(t), np.int8), np.ones(n_a, np.int8)])

    # arbitration: the flooded ID wins ties, then the bus serialises everything
    desired = all_t - np.where(all_lb == 1, 1e-9, 0.0)
    order = np.argsort(desired, kind="stable")
    all_t, all_id, all_dlc, all_pl, all_lb = (
        all_t[order], all_id[order], all_dlc[order], all_pl[order], all_lb[order]
    )
    idx = np.arange(len(all_t), dtype=np.float64) * FRAME_TIME
    all_t = np.maximum.accumulate(all_t - idx) + idx

    return clean["timestamp"].iloc[0] + all_t, all_id, all_dlc, all_pl, all_lb


def write_csv(path, t, can_id, dlc, payload, label) -> None:
    hexb = np.array([f"{v:02x}" for v in range(256)])
    pbytes = np.zeros((len(t), 8), dtype=np.uint8)
    for i in range(8):
        pbytes[:, i] = ((payload >> np.uint64(8 * (7 - i))) & np.uint64(0xFF))
    byte_str = hexb[pbytes]
    ts = np.char.mod("%.6f", t)
    ids = np.char.mod("%04x", can_id)
    dl = np.char.mod("%d", dlc)
    flag = np.where(label == 1, "T", "R")
    with open(path, "w", newline="") as fh:
        for lo in range(0, len(t), 200_000):
            hi = min(lo + 200_000, len(t))
            fh.write("\n".join(
                ts[i] + "," + ids[i] + "," + dl[i] + ","
                + ",".join(byte_str[i, : dlc[i]]) + "," + flag[i]
                for i in range(lo, hi)
            ))
            fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="normal_run_data.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["zero-id", "valid-id", "stealth", "lowrate",
                             "mixed"])
    ap.add_argument("--rate", type=float, default=2.0,
                    help="for lowrate: extra frames injected per real frame "
                         "of the victim ID")
    ap.add_argument("--attack-id", default="",
                    help="hex ID to flood; defaults to the highest-rate ID in "
                         "the clean capture")
    ap.add_argument("--attack-fraction", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    clean = load_hcrl_normal_txt(args.clean)
    print(f"clean capture: {len(clean)} frames, "
          f"{len(clean.can_id.unique())} IDs, "
          f"{clean.timestamp.iloc[-1] - clean.timestamp.iloc[0]:.1f} s")

    if args.attack_id:
        attack_id = int(args.attack_id, 16)
    else:
        counts = clean.can_id.value_counts()
        attack_id = int(counts.index[0])
    # zero-id ignores attack_id and floods 0x000, so report what is really used
    injected_id = ZERO_ID if args.mode == "zero-id" else attack_id
    print(f"mode {args.mode}, flooding ID {injected_id:#05x}")

    t, can_id, dlc, payload, label = build(
        clean, args.mode, attack_id, args.attack_fraction, args.seed,
        rate=args.rate,
    )
    write_csv(args.out, t, can_id, dlc, payload, label)
    print(f"frames    {len(t)}")
    print(f"injected  {int(label.sum())} ({100 * label.mean():.2f} %)")
    print(f"duration  {t[-1] - t[0]:.1f} s")
    print(f"rate      {len(t) / (t[-1] - t[0]):.0f} frames/s")
    print(f"wrote     {args.out}")


if __name__ == "__main__":
    main()
