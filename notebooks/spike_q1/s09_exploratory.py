"""Exploratory analyses E5 to E8 (EXPLORATORY.md, addendum of 2026-11-09), label (a).

    uv run python notebooks/spike_q1/s09_exploratory.py

Same pipeline as s06 and s08: hyper-parameters from out/criterion_a.json, one
negative sample shared by all models and drawn as s06 draws it (seed 42),
training 2019-2021, testing 2022 and 2023. Writes out/exploratory_recalibration.csv,
out/exploratory_label30ha.csv, out/exploratory_no_winter.csv and
out/exploratory_spatial_blocks_clim.csv. None of them changes the verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import s05_build_dataset as s05  # noqa: E402
import s06_train_evaluate as s06  # noqa: E402
import s08_exploratory as s08  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from src import config, grid  # noqa: E402
from src.models import validation  # noqa: E402

Y = s08.Y
TAG = s08.TAG
MIN_AREA_HA = 30.0
MODELS3 = ("lgbm_fwi", "lgbm_fwi_space", "lgbm_full")


def feature_sets() -> tuple[dict[str, list[str]], dict[str, dict]]:
    crit = json.loads((common.OUT / "criterion_a.json").read_text())["tuned"]
    feats = {"lgbm_fwi": list(s06.FWI_FEATS), "lgbm_fwi_space": s06.space_features(TAG),
             "lgbm_full": s06.full_features(TAG)}
    tuned = {"lgbm_fwi": crit["lgbm_fwi"], "lgbm_fwi_space": crit["lgbm_fwi_space"], "lgbm_full": crit["lgbm_full"]}
    return feats, tuned


def refit(df: pd.DataFrame, feats: dict, tuned: dict, models=MODELS3) -> dict:
    """Refit on the training years with one shared negative sample, as s06 does."""
    train = df[df["year"].isin(common.TRAIN_YEARS)]
    rng = np.random.default_rng(s06.SEED)
    sub, w = s06.sample_train(train, Y, rng)
    return {m: s08.fit_on(sub, w, feats[m], tuned[m]) for m in models}


def criterion(m3: dict, m4: dict) -> dict:
    rel = (m3["brier"] - m4["brier"]) / m3["brier"]
    gain = m4["pr_auc"] - m3["pr_auc"]
    return {"brier_rel_improvement": rel, "pr_auc_gain": gain, "passes": bool(rel >= 0.10 and gain >= 0.05)}


def score_models(df: pd.DataFrame, boosters: dict, feats: dict, tag: str, extra: dict | None = None) -> list[dict]:
    rows = []
    for yr in common.TEST_YEARS:
        te = df[df["year"] == yr]
        yt = te[Y].to_numpy()
        sc = {m: s08.score(yt, b.predict(te[feats[m]])) for m, b in boosters.items()}
        c = criterion(sc["lgbm_fwi"], sc["lgbm_full"])
        for m, s in sc.items():
            rows.append({"analysis": tag, "year": yr, "model": m, **s, **(extra or {}),
                         **(c if m == "lgbm_full" else {})})
    return rows


# ------------------------------------------------------------------------- E5
def intercept_shift(p: np.ndarray, y: np.ndarray) -> float:
    """Logit shift b such that mean(sigmoid(logit(p) + b)) matches the observed rate."""
    z = logit(np.clip(p, 1e-9, 1 - 1e-9))
    return float(brentq(lambda b: expit(z + b).sum() - y.sum(), -10.0, 10.0))


def e5(df: pd.DataFrame, feats: dict, tuned: dict) -> pd.DataFrame:
    boosters = refit(df, feats, tuned, ("lgbm_fwi", "lgbm_full"))
    val = df[df["year"] == 2021]
    rows = []
    sc: dict[tuple[int, str, str], dict] = {}
    for m, b in boosters.items():
        shift = intercept_shift(b.predict(val[feats[m]]), val[Y].to_numpy())
        for yr in common.TEST_YEARS:
            te = df[df["year"] == yr]
            yt = te[Y].to_numpy()
            p0 = b.predict(te[feats[m]])
            p1 = expit(logit(np.clip(p0, 1e-9, 1 - 1e-9)) + shift)
            for stage, p in (("raw", p0), ("recalibrated", p1)):
                sc[(yr, m, stage)] = {"brier": float(brier_score_loss(yt, p)), "obs_over_pred": float(yt.sum() / p.sum())}
            rows.append({"year": yr, "model": m, "shift_logit": shift, "base_rate_2021": float(val[Y].mean()),
                         "base_rate_test": float(yt.mean()),
                         "obs_over_pred_raw": sc[(yr, m, "raw")]["obs_over_pred"],
                         "obs_over_pred_recal": sc[(yr, m, "recalibrated")]["obs_over_pred"],
                         "brier_raw": sc[(yr, m, "raw")]["brier"], "brier_recal": sc[(yr, m, "recalibrated")]["brier"]})
    out = pd.DataFrame(rows)
    for yr in common.TEST_YEARS:
        for stage, col in (("raw", "brier_raw"), ("recal", "brier_recal")):
            b3 = out[(out.year == yr) & (out.model == "lgbm_fwi")][col].iloc[0]
            b4 = out[(out.year == yr) & (out.model == "lgbm_full")][col].iloc[0]
            out.loc[out.year == yr, f"brier_rel_gain_{stage}"] = (b3 - b4) / b3
    out["bar_met_recal"] = out["brier_rel_gain_recal"] >= 0.10
    return out


# ------------------------------------------------------------------------- E6
def label_floor(cells: pd.DataFrame, min_area: float) -> pd.DataFrame:
    """labels_effis of s05 with a size floor on the perimeter, same cell mapping."""
    fe = pd.read_parquet(config.DATA_PROCESSED / "fact_fireevent.parquet")
    fe = fe[(fe["source_key"] == "EFFIS_RDA") & fe["centroid_lat"].notna() & fe["firedate_key"].notna()]
    fe = fe[(fe["firedate_key"] >= s05.LAG_START) & (fe["firedate_key"] <= s05.DATES[-1]) & (fe["area_ha"] >= min_area)]
    fe = fe.assign(cell=common.cell_id(common.snap(fe["centroid_lon"]), common.snap(fe["centroid_lat"])),
                   date=fe["firedate_key"].dt.normalize())
    fe = fe[fe["cell"].isin(cells["cell"])]
    return fe.groupby(["cell", "date"]).size().rename("n_start").reset_index()


def e6(df: pd.DataFrame, feats: dict, tuned: dict) -> pd.DataFrame:
    cells = pd.read_parquet(common.INTERIM / "cells.parquet")
    lab = label_floor(cells, MIN_AREA_HA)
    lag = s05.lag_counts(lab, cells, s05.LAG_START, "a")
    d = df.drop(columns=[Y, "laga_7", "laga_30", "clim"]).merge(lag, on=["cell", "date"], how="left")
    d = d.merge(lab.rename(columns={"n_start": "n_e6"}), on=["cell", "date"], how="left")
    d[Y] = (d["n_e6"].fillna(0) > 0).astype("int8")
    d = s06.add_climatology(d.drop(columns=["n_e6"]), Y, common.TRAIN_YEARS)
    rows = score_models(d, refit(d, feats, tuned), feats, "label_30ha", {"min_area_ha": MIN_AREA_HA})
    out = pd.DataFrame(rows)
    yr_pos = d.groupby("year")[Y].agg(["sum", "mean"])
    out["positives_train"] = int(d[d["year"].isin(common.TRAIN_YEARS)][Y].sum())
    out["base_rate_train"] = float(d[d["year"].isin(common.TRAIN_YEARS)][Y].mean())
    print("label 30 ha positives and base rate per year:\n", yr_pos.round(5).to_string())
    return out


# ------------------------------------------------------------------------- E7
def e7(df: pd.DataFrame, feats: dict, tuned: dict) -> pd.DataFrame:
    winter = df["date"].dt.month <= 3
    kept = df[~winter]
    removed_train = float(df[winter & df["year"].isin(common.TRAIN_YEARS)][Y].sum() / df[df["year"].isin(common.TRAIN_YEARS)][Y].sum())
    rows = score_models(kept, refit(kept, feats, tuned), feats, "no_jan_mar", {"share_train_positives_removed": removed_train})
    out = pd.DataFrame(rows)
    for yr in common.TEST_YEARS:
        te = df[df["year"] == yr]
        out.loc[out.year == yr, "share_test_positives_removed"] = float(te[winter.loc[te.index]][Y].sum() / te[Y].sum())
    return out


# ------------------------------------------------------------------------- E8
def e8(df: pd.DataFrame, feats: dict, tuned: dict) -> pd.DataFrame:
    lon, lat = grid.cell_centre(df["cell"].to_numpy())
    blocks = validation.spatial_blocks(lon, lat, 2.0)
    blk = pd.Series(blocks, index=df.index)
    uniq = np.unique(blocks)
    train_years = df["year"].isin(common.TRAIN_YEARS)
    pooled: dict[tuple[int, str], list] = {}
    for _tr, te_idx, fold in validation.block_folds(uniq, k=5, seed=s06.SEED):
        in_held = blk.isin(set(uniq[te_idx]))
        d = df.copy()
        regional = df[~in_held & df["year"].isin(common.TEST_YEARS)].groupby("doy")["clim"].mean()
        d.loc[in_held, "clim"] = d.loc[in_held, "doy"].map(regional).to_numpy(dtype="float32")
        tr = d[train_years & ~in_held]
        rng = np.random.default_rng(s06.SEED + fold)
        for m in MODELS3:
            b = s08.fit(tr, feats[m], tuned[m], rng)
            for yr in common.TEST_YEARS:
                te = d[(d["year"] == yr) & in_held]
                if not te.empty:
                    pooled.setdefault((yr, m), []).append((te[Y].to_numpy(), b.predict(te[feats[m]])))
        print(f"fold {fold + 1}/5 done", flush=True)
    e1 = pd.read_csv(common.OUT / "exploratory_spatial_blocks.csv").set_index(["year", "model"])
    rows = []
    for (yr, m), parts in sorted(pooled.items()):
        s = s08.score(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
        u = float(e1.loc[(yr, m), "pr_auc_unblocked"])
        drop = (u - s["pr_auc"]) / u
        rows.append({"year": yr, "model": m, **s, "pr_auc_unblocked": u, "pr_auc_e1": float(e1.loc[(yr, m), "pr_auc"]),
                     "relative_drop": drop, "transfers": bool(drop <= 0.30)})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(common.INTERIM / "daily.parquet")
    s06.WEATHER[:] = [c for c in s06.WEATHER_ALL if c in df.columns]
    df = s06.add_climatology(df, Y, common.TRAIN_YEARS)
    feats, tuned = feature_sets()
    results = {
        "recalibration": e5(df, feats, tuned),
        "label30ha": e6(df, feats, tuned),
        "no_winter": e7(df, feats, tuned),
        "spatial_blocks_clim": e8(df, feats, tuned),
    }
    for name, tab in results.items():
        tab.to_csv(common.OUT / f"exploratory_{name}.csv", index=False)
        print(f"\n== {name} ==")
        print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    with common.timed("exploratory_e5_e8"):
        main()
