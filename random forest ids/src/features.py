"""Hardware-faithful feature extraction for a streaming CAN IDS.

Every feature here is computable in a single streaming pass, one frame at a
time, from a small amount of state kept per CAN ID:

    per-ID RAM (2048 entries)      per-ID ROM (trained offline)
      last_timestamp_us  (20b)       mean_interval_us   (20b)
      last_payload       (64b)       recip_interval     (16b)
      valid              (1b)        mean_hamming       (7b)
                                     id_known           (1b)

and three global registers: last bus timestamp, previous CAN ID, burst counter.

All features are integers, so the decision-tree thresholds are integers too and
quantising the trained model for hardware costs exactly nothing.

FEATURES
  0 dt_id        us since the previous frame of this ID, saturating
  1 dt_id_dev    |dt_id - mean_interval| for this ID, saturating
  2 dt_ratio_q6  dt_id / mean_interval in 1/64 steps (64 == exactly on period).
                 One 16x16 multiply by a stored reciprocal, no divider. This is
                 the scale-invariant version of feature 1, so one threshold
                 works for a 10 ms ID and a 1000 ms ID alike.
  3 hd           Hamming distance to the previous payload of this ID
  4 hd_dev       |hd - mean_hamming| for this ID
  5 dt_bus       us since the previous frame on the bus, any ID
  6 burst        run length of consecutive frames sharing this ID, capped
  7 id_known     1 if this ID appeared in clean training traffic
  8 can_id       the raw 11-bit identifier
  9 dlc          data length code
 10 pl_violation number of payload bits that break this ID's learned invariant.
                 For every ID the training pass records which payload bits never
                 change in clean traffic, and what value they hold. A frame that
                 flips one of those bits is structurally impossible for that ID.
                 Costs one 64-bit XOR, one AND and a popcount in hardware.
 11 pl_popcount  population count of the raw payload. Degenerate payloads (all
                 zeros, as the real DoS capture injects) sit at one extreme.
 12 id_rate      leaky bucket per ID, in the same 1/64 units as dt_ratio_q6.
                 Each frame of an ID adds 64 and drains dt_ratio_q6, so an ID
                 running at its nominal period sits still at 64 while one being
                 flooded climbs fast. Costs one subtract, one add and 12 bits of
                 state per ID, and reuses the reciprocal multiply already in the
                 datapath.
 13 bus_rate     the same bucket for the bus as a whole, against the nominal
                 frame interval learned from clean traffic. One register.
The two rate features exist because during a sustained flood of one ID, every
frame carrying that ID arrives at the flood period, injected or not. dt_id
therefore separates nothing for the victim ID. A bucket integrates over many
frames, so it still reports that the ID's stream is running far over rate even
when no single inter-arrival time says so.

The last two exist because timing and Hamming distance alone cannot separate a
flood that reuses a legitimate ID. Once an attacker floods your ID, your own
frames' inter-arrival times are corrupted too, so the injected frames and the
victim's real frames look alike on every timing feature. Payload content is the
only thing left that still differs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAP_DT = (1 << 20) - 1      # 1.048575 s, covers the 1000 ms IDs
CAP_BUS = (1 << 16) - 1     # 65.535 ms
CAP_RATIO = (1 << 12) - 1   # 64x the nominal period
CAP_BURST = 15
RECIP_SHIFT = 16
MIN_INVARIANT_SAMPLES = 500
CAP_RATE = (1 << 12) - 1
# Exponentially weighted rate estimate. Each frame adds RATE_INC and leaks a
# fraction of the bucket's own contents scaled by elapsed time, which gives the
# fixed point b = 4096 / dt_ratio_q6. An ID at its nominal period therefore
# settles at 64 and one being flooded 64x over rate saturates. A leak that did
# not depend on the bucket contents would have no restoring force at all and
# would random-walk to saturation on perfectly normal traffic.
RATE_SHIFT = 8
RATE_INC = 16
RATE_INIT = 64

FEATURE_NAMES = [
    "dt_id", "dt_id_dev", "dt_ratio_q6", "hd", "hd_dev",
    "dt_bus", "burst", "id_known", "can_id", "dlc",
    "pl_violation", "pl_popcount", "id_rate", "bus_rate",
]

# Feature subsets worth comparing. "timing" carries no identity information at
# all, so it is the one that still works when an attacker floods a valid ID.
FEATURE_SETS = {
    "full": FEATURE_NAMES,
    # no identity information at all, so it still works when a flood reuses a
    # legitimate ID. This is the set to deploy.
    "timing": ["dt_id", "dt_id_dev", "dt_ratio_q6", "hd", "hd_dev",
               "dt_bus", "burst", "dlc", "pl_violation", "pl_popcount",
               "id_rate", "bus_rate"],
    # timing and Hamming only, kept to show what the payload features buy
    "timing_only": ["dt_id", "dt_id_dev", "dt_ratio_q6", "hd", "hd_dev",
                    "dt_bus", "burst", "dlc", "id_rate", "bus_rate"],
    # what the model could do before the rate buckets were added, kept so the
    # gain from them is measurable rather than asserted
    "no_rate": ["dt_id", "dt_id_dev", "dt_ratio_q6", "hd", "hd_dev",
                "dt_bus", "burst", "dlc", "pl_violation", "pl_popcount"],
    # timing and rate only, no payload at all. For datasets that give signal
    # values rather than raw payload bytes, where anything payload-derived
    # would describe a reconstruction instead of the real frame.
    "no_payload": ["dt_id", "dt_id_dev", "dt_ratio_q6", "dt_bus", "burst",
                   "dlc", "id_rate", "bus_rate"],
    # Only features that are relative to what the ID itself normally does, so
    # a threshold learned on a 10 ms ID means the same thing on a 100 ms one.
    #
    # This exists because the opposite failed, measurably. A forest trained on
    # two 10 ms victim IDs put absolute thresholds in three of its five trees
    # -- one rooted on dt_id <= 5458 us, two on burst -- and those three then
    # voted "normal" on a 100 ms ID flooded at 8x its rate. The two trees using
    # dt_ratio_q6 and id_rate voted attack on 99 % and 96 % of those same
    # frames, so the right answer was there and the majority buried it. Taking
    # the absolute features away is the same move as taking can_id away: it
    # removes a shortcut that happens to work on the training victims.
    "scale_free": ["dt_ratio_q6", "id_rate", "bus_rate"],
    # scale-free and strictly per-ID. bus_rate is dropped because it is a
    # bus-wide signal being used to make a per-ID decision, which is a
    # category error and shows up as one: a model rooting on bus_rate <= 63,
    # where clean traffic sits at 60 and a finished flood leaves it at 64 to
    # 66 for 1.4 to 4.6 seconds, alarmed on eight innocent IDs at once two
    # and a half seconds after a DoS ended. The per-ID features carry the
    # signal on their own; bus_rate only adds cross-ID contamination.
    "per_id": ["dt_ratio_q6", "id_rate"],
    # scale-free plus the bus-level timing, which is an absolute microsecond
    # count but describes the bus rather than any one ID's period
    "scale_free_bus": ["dt_ratio_q6", "id_rate", "bus_rate", "dt_bus", "dlc"],
    # the two features the reference paper uses
    "paper": ["dt_id_dev", "hd_dev"],
}

_POPCNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Bit count of a uint64 array, via a byte lookup table."""
    b = np.ascontiguousarray(x.astype(np.uint64)).view(np.uint8)
    return _POPCNT8[b].reshape(-1, 8).sum(axis=1).astype(np.int64)


class Baseline:
    """Per-ID constants learned offline from clean frames, i.e. the ROM."""

    def __init__(self, mean_interval: dict, mean_hamming: dict,
                 const_mask: dict | None = None,
                 const_val: dict | None = None,
                 bus_interval_us: int = 0):
        self.mean_interval = mean_interval
        self.mean_hamming = mean_hamming
        # payload bits that never move in clean traffic, and their value
        self.const_mask = const_mask or {}
        self.const_val = const_val or {}
        # nominal gap between frames on the bus, for the global rate bucket
        self.bus_interval_us = bus_interval_us

    @property
    def known_ids(self) -> set:
        return set(self.mean_interval)

    def as_tables(self):
        """Dense per-ID lookup tables, indexed by the 11-bit CAN ID."""
        n = 2048
        mean_int = np.zeros(n, dtype=np.int64)
        recip = np.zeros(n, dtype=np.int64)
        mean_hd = np.zeros(n, dtype=np.int64)
        known = np.zeros(n, dtype=np.int64)
        cmask = np.zeros(n, dtype=np.uint64)
        cval = np.zeros(n, dtype=np.uint64)
        for cid, mi in self.mean_interval.items():
            if cid >= n:
                continue
            mean_int[cid] = mi
            recip[cid] = int(round((1 << RECIP_SHIFT) * 64.0 / mi)) if mi > 0 else 0
            mean_hd[cid] = self.mean_hamming.get(cid, 0)
            known[cid] = 1
            cmask[cid] = np.uint64(self.const_mask.get(cid, 0))
            cval[cid] = np.uint64(self.const_val.get(cid, 0))
        return mean_int, recip, mean_hd, known, cmask, cval


def _raw_state(df: pd.DataFrame):
    """Stateful quantities that do not depend on the learned baseline."""
    ts_us = np.rint((df["timestamp"].to_numpy() - df["timestamp"].iloc[0]) * 1e6)
    ts_us = ts_us.astype(np.int64)
    can_id = df["can_id"].to_numpy().astype(np.int64)
    payload = df["payload"].to_numpy().astype(np.uint64)

    s_ts = pd.Series(ts_us)
    prev_ts = s_ts.groupby(can_id).shift()
    first_seen = prev_ts.isna().to_numpy()
    dt_id = np.where(first_seen, CAP_DT, ts_us - prev_ts.fillna(0).to_numpy())
    dt_id = np.clip(dt_id, 0, CAP_DT).astype(np.int64)

    prev_pl = pd.Series(payload).groupby(can_id).shift(fill_value=0)
    hd = popcount64(payload ^ prev_pl.to_numpy().astype(np.uint64))

    dt_bus = np.empty(len(df), dtype=np.int64)
    dt_bus[0] = CAP_BUS
    dt_bus[1:] = np.clip(np.diff(ts_us), 0, CAP_BUS)

    new_run = np.ones(len(df), dtype=bool)
    new_run[1:] = can_id[1:] != can_id[:-1]
    run_id = np.cumsum(new_run)
    burst = pd.Series(np.zeros(len(df), dtype=np.int64)).groupby(run_id).cumcount()
    burst = np.minimum(burst.to_numpy(), CAP_BURST)

    return ts_us, can_id, dt_id, hd, dt_bus, burst


def fit_baseline(df: pd.DataFrame) -> Baseline:
    """Learn per-ID interval and Hamming means from clean frames only.

    Only rows labelled normal are used, and only from whatever truncation is
    passed in, so nothing from the test truncation can leak in.
    """
    clean = df[df["label"] == 0]
    _, can_id, dt_id, hd, dt_bus_clean, _ = _raw_state(clean)
    payload_clean = clean["payload"].to_numpy().astype(np.uint64)

    mean_interval, mean_hamming = {}, {}
    const_mask, const_val = {}, {}
    order = np.argsort(can_id, kind="stable")
    cid_sorted = can_id[order]
    bounds = np.searchsorted(cid_sorted, np.unique(cid_sorted), side="left")
    bounds = np.append(bounds, len(cid_sorted))
    uniq = np.unique(cid_sorted)

    dt_sorted = dt_id[order]
    hd_sorted = hd[order]
    pl_sorted = payload_clean[order]
    for k, cid in enumerate(uniq):
        lo, hi = bounds[k], bounds[k + 1]
        d = dt_sorted[lo:hi]
        d = d[d < CAP_DT]                     # drop the first-seen sentinel
        if len(d) >= 8:
            # median, not mean: robust to the queueing delays a flood causes
            mean_interval[int(cid)] = max(1, int(round(float(np.median(d)))))
            mean_hamming[int(cid)] = int(round(float(np.median(hd_sorted[lo:hi]))))

            pl = pl_sorted[lo:hi]
            ones = np.bitwise_and.reduce(pl)      # bits that are always 1
            union = np.bitwise_or.reduce(pl)      # bits that are ever 1
            zeros = ~union                        # bits that are always 0
            # A bit only counts as invariant with enough clean evidence behind
            # it. Below this many samples the "never changed" conclusion is an
            # artefact of a short training window and produces false positives.
            if len(pl) >= MIN_INVARIANT_SAMPLES:
                const_mask[int(cid)] = int(ones | zeros)
                const_val[int(cid)] = int(ones)
    valid_bus = dt_bus_clean[dt_bus_clean < CAP_BUS]
    bus_interval = max(1, int(round(float(np.median(valid_bus))))) if len(
        valid_bus) else 1
    return Baseline(mean_interval, mean_hamming, const_mask, const_val,
                    bus_interval)


def _rate_buckets(can_id, dt_ratio, bus_ratio):
    """Leaky-bucket rate estimators, per ID and for the bus as a whole.

    Per frame:   leak = min(b, (b * ratio) >> RATE_SHIFT)
                 b    = min(CAP_RATE, b - leak + RATE_INC)

    A genuine recurrence with clamping, so it cannot be vectorised. It is one
    small multiply, one subtract and one add per frame.
    """
    n = len(can_id)
    id_rate = np.empty(n, dtype=np.int32)
    bus_rate = np.empty(n, dtype=np.int32)
    buckets = {}
    bus = RATE_INIT
    cid_l = can_id.tolist()
    dtr_l = dt_ratio.tolist()
    busr_l = bus_ratio.tolist()
    for i in range(n):
        cid = cid_l[i]
        b = buckets.get(cid, RATE_INIT)
        leak = (b * dtr_l[i]) >> RATE_SHIFT
        if leak > b:
            leak = b
        b = b - leak + RATE_INC
        if b > CAP_RATE:
            b = CAP_RATE
        buckets[cid] = b
        id_rate[i] = b

        leak = (bus * busr_l[i]) >> RATE_SHIFT
        if leak > bus:
            leak = bus
        bus = bus - leak + RATE_INC
        if bus > CAP_RATE:
            bus = CAP_RATE
        bus_rate[i] = bus
    return id_rate, bus_rate


def extract(df: pd.DataFrame, baseline: Baseline) -> pd.DataFrame:
    """Build the full integer feature matrix for a trace."""
    ts_us, can_id, dt_id, hd, dt_bus, burst = _raw_state(df)
    mean_int_t, recip_t, mean_hd_t, known_t, cmask_t, cval_t = \
        baseline.as_tables()

    idx = np.clip(can_id, 0, 2047)
    mean_int = mean_int_t[idx]
    recip = recip_t[idx]
    mean_hd = mean_hd_t[idx]
    known = known_t[idx]

    dt_id_dev = np.clip(np.abs(dt_id - mean_int), 0, CAP_DT)
    dt_ratio = np.clip((dt_id * recip) >> RECIP_SHIFT, 0, CAP_RATIO)
    hd_dev = np.abs(hd - mean_hd)

    payload = df["payload"].to_numpy().astype(np.uint64)
    cmask = cmask_t[idx]
    cval = cval_t[idx]
    pl_violation = popcount64((payload ^ cval) & cmask)
    pl_popcount = popcount64(payload)

    bus_recip = (int(round((1 << RECIP_SHIFT) * 64.0 / baseline.bus_interval_us))
                 if baseline.bus_interval_us > 0 else 0)
    bus_ratio = np.clip((dt_bus * bus_recip) >> RECIP_SHIFT, 0, CAP_RATIO)
    id_rate, bus_rate = _rate_buckets(can_id, dt_ratio, bus_ratio)

    out = pd.DataFrame(
        {
            "dt_id": dt_id.astype(np.int32),
            "dt_id_dev": dt_id_dev.astype(np.int32),
            "dt_ratio_q6": dt_ratio.astype(np.int32),
            "hd": hd.astype(np.int16),
            "hd_dev": hd_dev.astype(np.int16),
            "dt_bus": dt_bus.astype(np.int32),
            "burst": burst.astype(np.int8),
            "id_known": known.astype(np.int8),
            "can_id": can_id.astype(np.int16),
            "dlc": df["dlc"].to_numpy().astype(np.int8),
            "pl_violation": pl_violation.astype(np.int16),
            "pl_popcount": pl_popcount.astype(np.int16),
            "id_rate": id_rate.astype(np.int32),
            "bus_rate": bus_rate.astype(np.int32),
        }
    )
    return out
