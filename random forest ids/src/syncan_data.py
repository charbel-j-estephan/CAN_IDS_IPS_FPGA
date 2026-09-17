"""Loader for the SynCAN benchmark (ETAS GmbH / Robert Bosch GmbH).

SynCAN is an independent CAN IDS benchmark, and `test_flooding` is a DoS on a
legitimate ID: "the attacker sends messages of a particular existing ID with
high frequency to the CAN bus". That is exactly the case the HCRL DoS capture
cannot exercise, so it is worth testing against.

    Hanselmann, Strauss, Dormann, Ulmer, "CANet: An Unsupervised Intrusion
    Detection System for High Dimensional CAN Bus Data", IEEE Access 8, 2020.
    https://github.com/etas/SynCAN

Licence note: SynCAN is free for academic and non-commercial research and
requires citation, but explicitly forbids redistributing the data or modified
versions. Trained models and metrics are fine. Keep the CSVs out of the repo.

Two differences from the HCRL format that matter:

**Signals, not payload bytes.** Columns are
`Label, Time, ID, Signal1_of_ID .. Signal4_of_ID`, with 1 to 4 float signals
per ID rather than 8 raw bytes. A payload is reconstructed by quantising each
signal to 16 bits and packing them big-endian, which is what a real CAN frame
encodes anyway, but it is a reconstruction and not the bytes a transceiver saw.
Prefer the `no_payload` feature set on this data; the payload-derived features
are computable but describe the reconstruction.

**Window labels, not injected-frame labels.** "During each attacked time
interval, the label of all IDs is set to one." So a frame labelled 1 is not
necessarily injected, it is merely inside an attacked window. HCRL labels only
the injected frames. Per-frame accuracy against SynCAN labels therefore
measures the labelling convention as much as the detector; use
`src/eval_windows.py` for a comparison that means something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID",
               "Signal4_of_ID"]


def load_syncan_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a SynCAN CSV into the same schema the HCRL loaders produce."""
    raw = pd.read_csv(path, nrows=nrows)

    label = raw["Label"].astype(np.int8).to_numpy()
    # Time is in milliseconds; the rest of the pipeline works in seconds
    timestamp = raw["Time"].astype(np.float64).to_numpy() / 1000.0

    # ids are the strings id1 .. id10
    can_id = (raw["ID"].astype(str).str.extract(r"(\d+)")[0]
              .astype(np.int64).to_numpy())

    sig = np.zeros((len(raw), 4), dtype=np.float64)
    present = np.zeros((len(raw), 4), dtype=bool)
    for i, c in enumerate(SIGNAL_COLS):
        if c in raw.columns:
            v = pd.to_numeric(raw[c], errors="coerce")
            present[:, i] = v.notna().to_numpy()
            sig[:, i] = v.fillna(0.0).to_numpy()

    n_sig = present.sum(axis=1)
    dlc = np.clip(2 * n_sig, 0, 8).astype(np.int64)

    # quantise each signal to 16 bits and pack big-endian into the payload
    q = np.clip(np.rint(sig * 65535.0), 0, 65535).astype(np.uint64)
    q = q * present.astype(np.uint64)
    payload = np.zeros(len(raw), dtype=np.uint64)
    for i in range(4):
        payload |= q[:, i] << np.uint64(16 * (3 - i))

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "can_id": can_id,
            "dlc": dlc,
            "payload": payload,
            "label": label,
        }
    )
    # the file is ordered by time already, but do not rely on it
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)
