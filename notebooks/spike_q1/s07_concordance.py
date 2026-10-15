"""Step 4: concordance between label (a) EFFIS starts and label (b) VIIRS cluster starts.

    uv run python notebooks/spike_q1/s07_concordance.py

Strict: same cell, same day. Tolerant: an (a) start counts as seen by (b) if a (b)
start exists within +/-1 day in the same cell or one of its 8 neighbours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if n and pe < 1 else float("nan")


def main() -> None:
    df = pd.read_parquet(common.INTERIM / "daily.parquet", columns=["cell", "date", "year", "y_a", "y_b"])
    rows = []
    # (b) positive set expanded to the 3x3 neighbourhood and +/-1 day
    b_pos = df[df["y_b"] > 0][["cell", "date"]]
    exp = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for dd in (-1, 0, 1):
                exp.append(pd.DataFrame({
                    "cell": b_pos["cell"].to_numpy() + dy * 10_000 + dx,
                    "date": b_pos["date"] + pd.Timedelta(days=dd),
                }))
    seen = pd.concat(exp).drop_duplicates()
    seen["tol"] = True
    df = df.merge(seen, on=["cell", "date"], how="left")
    df["tol"] = df["tol"].fillna(False)
    for yr, g in [("all", df), *df.groupby("year")]:
        a, b = g["y_a"].to_numpy() > 0, g["y_b"].to_numpy() > 0
        both = int((a & b).sum())
        rows.append({
            "year": yr, "a_positives": int(a.sum()), "b_positives": int(b.sum()), "both_same_day": both,
            "jaccard": both / max(int((a | b).sum()), 1), "kappa": kappa(a, b),
            "a_seen_by_b_same_day": both / max(int(a.sum()), 1),
            "a_seen_by_b_tolerant": float((a & g["tol"].to_numpy()).sum() / max(int(a.sum()), 1)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(common.OUT / "concordance.csv", index=False)
    print(out.round(4).to_string(index=False))
    offsets = day_offsets(df, b_pos)
    offsets.to_csv(common.OUT / "concordance_offsets.csv", index=False)
    print("\nday offset of the nearest VIIRS start (3x3 cells, +/-5 d) relative to the EFFIS firedate:")
    print(offsets.to_string(index=False))


def day_offsets(df: pd.DataFrame, b_pos: pd.DataFrame, max_days: int = 5) -> pd.DataFrame:
    """For each EFFIS start, the signed day offset of the nearest VIIRS cluster
    start in the 3x3 neighbourhood within +/- max_days. Negative = VIIRS saw it
    first. A mass at -1 or -2 means EFFIS `firedate` lags ignition, which
    misaligns label (a) with day-J features for every model alike."""
    a_pos = df[df["y_a"] > 0][["cell", "date"]].reset_index(drop=True)
    a_pos["key"] = np.arange(len(a_pos))
    exp = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            exp.append(pd.DataFrame({"key": a_pos["key"], "cell": a_pos["cell"] + dy * 10_000 + dx,
                                     "a_date": a_pos["date"]}))
    nb = pd.concat(exp, ignore_index=True)
    m = nb.merge(b_pos.rename(columns={"date": "b_date"}), on="cell", how="inner")
    m["offset"] = (m["b_date"] - m["a_date"]).dt.days
    m = m[m["offset"].abs() <= max_days]
    best = m.loc[m.groupby("key")["offset"].apply(lambda s: s.abs().idxmin())]
    counts = best["offset"].value_counts().sort_index()
    res = counts.rename("n").reset_index().rename(columns={"index": "offset_days", "offset": "offset_days"})
    res["share_of_a"] = res["n"] / max(len(a_pos), 1)
    unmatched = len(a_pos) - int(counts.sum())
    res = pd.concat([res, pd.DataFrame({"offset_days": ["none"], "n": [unmatched],
                                        "share_of_a": [unmatched / max(len(a_pos), 1)]})], ignore_index=True)
    return res


if __name__ == "__main__":
    main()
