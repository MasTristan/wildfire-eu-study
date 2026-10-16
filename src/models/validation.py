"""Validation protocol shared by the models: splits that respect time and
space, the metrics the pre-registered criteria use, reliability tables and
block bootstrap intervals.

Rules the splits enforce:
- temporal: a held-out year is never used for any choice (features that
  need history are computed from earlier years only, upstream);
- spatial: cells are grouped in blocks of `block_deg` degrees, so a block
  held out for validation shares no neighbour with the training blocks and
  spatial autocorrelation cannot inflate the score.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


def leave_one_year_out(years: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """(train_idx, test_idx, held_year) for each distinct year, in order."""
    y = np.asarray(years)
    for held in np.unique(y):
        test = np.flatnonzero(y == held)
        train = np.flatnonzero(y != held)
        yield train, test, int(held)


def spatial_blocks(lon: np.ndarray, lat: np.ndarray, block_deg: float = 2.0) -> np.ndarray:
    """Integer block id of each point on a block_deg x block_deg lattice."""
    bx = np.floor((np.asarray(lon, dtype="float64") + 180.0) / block_deg).astype("int64")
    by = np.floor((np.asarray(lat, dtype="float64") + 90.0) / block_deg).astype("int64")
    return by * 1000 + bx


def block_folds(blocks: np.ndarray, k: int = 5, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Grouped k-fold over spatial blocks: every row of a block lands in one fold."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    fold_of_block = dict(zip(uniq, rng.permutation(len(uniq)) % k, strict=True))
    fold = np.array([fold_of_block[b] for b in blocks])
    for f in range(k):
        yield np.flatnonzero(fold != f), np.flatnonzero(fold == f), f


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype="float64"), 1e-9, 1 - 1e-9)
    return {
        "n": float(len(y)),
        "positives": float(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)) if 0 < y.sum() < len(y) else float("nan"),
    }


def brier_skill(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray) -> float:
    """1 - Brier(model) / Brier(reference): positive means better than the reference."""
    b, b0 = brier_score_loss(y, p), brier_score_loss(y, p_ref)
    return float(1.0 - b / b0) if b0 > 0 else float("nan")


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed rate per decile of predicted probability (bins merged when predictions tie)."""
    q = pd.qcut(pd.Series(p), bins, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True)


def block_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray,
                          reps: int = 200, seed: int = 42) -> pd.DataFrame:
    """Percentile intervals of (Brier relative improvement of b over a, PR-AUC
    gain of b over a) under resampling of whole groups (weeks, blocks)."""
    rng = np.random.default_rng(seed)
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    d_brier, d_ap = [], []
    for _ in range(reps):
        pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
        idx = np.concatenate([idx_by_group[i] for i in pick])
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ba, bb = np.mean((p_a[idx] - yb) ** 2), np.mean((p_b[idx] - yb) ** 2)
        d_brier.append((ba - bb) / ba)
        d_ap.append(average_precision_score(yb, p_b[idx]) - average_precision_score(yb, p_a[idx]))
    rows = []
    for name, arr in (("brier_rel_improvement", d_brier), ("pr_auc_gain", d_ap)):
        lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
        rows.append({"quantity": name, "p2.5": lo, "median": med, "p97.5": hi, "reps": len(arr)})
    return pd.DataFrame(rows)
