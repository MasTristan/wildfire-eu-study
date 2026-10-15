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


if __name__ == "__main__":
    main()
