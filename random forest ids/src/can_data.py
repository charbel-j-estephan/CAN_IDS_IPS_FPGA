"""Loader for the HCRL Car-Hacking CSV format (DoS_dataset.csv and friends).

Real file layout, no header row, variable column count:

    Timestamp,CAN_ID,DLC,DATA[0],...,DATA[DLC-1],Flag

    1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R
    1478198376.389636,0000,8,00,00,00,00,00,00,00,00,T

Flag R means a normal frame, T means an injected (attack) frame. When DLC is
below 8 the row is shorter, so the flag does not sit in a fixed column. This
loader handles that, and returns fixed-width integer arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_COLS = 12  # timestamp + id + dlc + 8 data bytes + flag


def _hex_to_int(series: pd.Series) -> np.ndarray:
    """Vectorised hex string to int, by mapping only the unique values."""
    codes, uniques = pd.factorize(series, use_na_sentinel=False)
    table = np.zeros(len(uniques), dtype=np.int64)
    for i, u in enumerate(uniques):
        # Short-DLC rows put the R/T flag where a data byte would be, so
        # anything that is not a hex literal is simply read as zero padding.
        if isinstance(u, str) and u.strip() != "":
            try:
                table[i] = int(u, 16)
            except ValueError:
                table[i] = 0
    out = np.where(codes < 0, 0, table[np.clip(codes, 0, None)])
    return out.astype(np.int64)


def load_hcrl_csv(path: str, nrows: int | None = None, skiprows: int = 0) -> pd.DataFrame:
    """Read an HCRL car-hacking CSV into a tidy frame.

    Returns columns: timestamp (float s), can_id (int), dlc (int),
    payload (uint64, MSB-first, zero padded), label (0 normal / 1 attack).
    """
    raw = pd.read_csv(
        path,
        header=None,
        names=list(range(MAX_COLS)),
        dtype=str,
        skiprows=skiprows,
        nrows=nrows,
        engine="c",
        on_bad_lines="skip",
    )

    timestamp = raw[0].astype(np.float64).to_numpy()
    can_id = _hex_to_int(raw[1])
    dlc = raw[2].astype(np.int64).to_numpy()
    dlc = np.clip(dlc, 0, 8)

    # Data bytes live in columns 3 .. 3+dlc-1, the flag in column 3+dlc.
    byte_cols = np.zeros((len(raw), 8), dtype=np.uint64)
    for col in range(3, 11):
        vals = _hex_to_int(raw[col]).astype(np.uint64)
        byte_cols[:, col - 3] = vals

    payload = np.zeros(len(raw), dtype=np.uint64)
    for i in range(8):
        keep = (i < dlc).astype(np.uint64)
        payload |= (byte_cols[:, i] * keep) << np.uint64(8 * (7 - i))

    flag_col = 3 + dlc
    flags = raw.to_numpy()
    rows = np.arange(len(raw))
    flag = flags[rows, np.clip(flag_col, 0, MAX_COLS - 1)]
    flag = pd.Series(flag).astype(str).str.strip().str.upper()
    label = (flag == "T").to_numpy().astype(np.int8)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "can_id": can_id,
            "dlc": dlc,
            "payload": payload,
            "label": label,
        }
    )


def truncate(df: pd.DataFrame, start_frac: float, end_frac: float) -> pd.DataFrame:
    """Take a contiguous truncation of the trace by fraction of frames.

    Contiguous matters: the features are sequential (inter-arrival time, Hamming
    distance against the previous frame of the same ID), so a random row split
    would leak the neighbours of every test frame into training.
    """
    n = len(df)
    lo = int(round(start_frac * n))
    hi = int(round(end_frac * n))
    return df.iloc[lo:hi].reset_index(drop=True)


def load_hcrl_normal_txt(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read HCRL's attack-free capture, `normal_run_data.txt`.

    Different layout from the attack CSVs, whitespace separated:

        Timestamp: 1479121434.850202    ID: 0350    000    DLC: 8    05 28 ... a2

    Every frame is normal, so label is 0 throughout. This capture is from a
    separate drive, so it is the honest source for the per-ID baseline: a
    deployed IDS learns its ROM contents from clean traffic recorded earlier,
    not from the normal frames of a capture that is already under attack.
    """
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=list(range(15)),
        dtype=str,
        nrows=nrows,
        engine="python",
        on_bad_lines="skip",
    )

    timestamp = raw[1].astype(np.float64).to_numpy()
    can_id = _hex_to_int(raw[3])
    dlc = np.clip(raw[6].astype(np.int64).to_numpy(), 0, 8)

    payload = np.zeros(len(raw), dtype=np.uint64)
    for i in range(8):
        vals = _hex_to_int(raw[7 + i]).astype(np.uint64)
        keep = (i < dlc).astype(np.uint64)
        payload |= (vals * keep) << np.uint64(8 * (7 - i))

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "can_id": can_id,
            "dlc": dlc,
            "payload": payload,
            "label": np.zeros(len(raw), dtype=np.int8),
        }
    )
