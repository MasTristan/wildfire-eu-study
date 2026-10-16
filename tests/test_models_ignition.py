"""Q1 production protocol on a synthetic feature store: the climatology
leak rule, sampling weights, and a full run producing every table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import grid
from src.models import ignition as ig


def _store(seed: int = 0, n_cells: int = 30, years=(2019, 2020, 2021, 2022)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lon = -4.0 + 0.25 * (np.arange(n_cells) % 6)
    lat = 40.0 + 0.25 * (np.arange(n_cells) // 6)
    cells = grid.cell_id(lon, lat)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    n = len(cells) * len(dates)
    df = pd.DataFrame({"cell": np.tile(cells, len(dates)), "date": np.repeat(dates, len(cells))})
    df["year"] = df["date"].dt.year.astype("int16")
    df["doy"] = df["date"].dt.dayofyear.astype("int16")
    df["country_key"] = np.where(df["cell"] % 2 == 0, "ES", "PT")
    season = np.exp(-((df["doy"] - 200) / 40.0) ** 2)
    fwi = np.clip(rng.gamma(2.0, 8.0, n) * (0.4 + season), 0, 120)
    df["fwi"] = fwi
    for c in ["ffmc", "dmc", "dc", "isi", "bui", "dsr"]:
        df[c] = fwi * rng.uniform(0.5, 1.5, n)
    for c in ig.WEATHER_FEATS:
        df[c] = rng.normal(size=n)
    for c in ig.STATIC_FEATS:
        df[c] = df["cell"].map(pd.Series(rng.uniform(size=n_cells), index=cells))
    cell_fx = df["cell"].map(pd.Series(rng.normal(size=n_cells), index=cells))
    p = 1 / (1 + np.exp(-(-6.0 + 0.06 * fwi + 0.8 * cell_fx + 0.4 * df["spei3"])))
    df["y_a"] = (rng.uniform(size=n) < p).astype("int8")
    df["n_a"] = df["y_a"].astype("int16")
    for w in (7, 30):
        df[f"laga_{w}"] = df.groupby("cell")["y_a"].transform(lambda s, w=w: s.rolling(w, min_periods=1).sum().shift(1)).fillna(0)
    return df


def test_climatology_leaves_own_year_out_and_never_uses_scored_years():
    df = _store(n_cells=6, years=(2019, 2020, 2021))
    c = df["cell"].iloc[0]
    # plant a burst of ignitions in 2021 on day 100 for one cell
    mask = (df["cell"] == c) & (df["year"] == 2021) & (df["doy"] == 100)
    df.loc[mask, "y_a"] = 1
    with_2021 = ig.add_climatology(df, "y_a", [2019, 2020, 2021])
    without = ig.add_climatology(df, "y_a", [2019, 2020])
    row_2019 = (with_2021["cell"] == c) & (with_2021["year"] == 2019) & (with_2021["doy"] == 100)
    row_2021 = (with_2021["cell"] == c) & (with_2021["year"] == 2021) & (with_2021["doy"] == 100)
    # the 2021 burst shows in a 2019 row only when 2021 is an allowed year
    assert with_2021.loc[row_2019, "clim"].item() > without.loc[row_2019, "clim"].item()
    # a 2021 row never sees its own year's burst
    assert with_2021.loc[row_2021, "clim"].item() == without.loc[row_2021, "clim"].item()


def test_sampling_keeps_every_positive_with_calibrating_weights():
    df = _store(n_cells=6, years=(2019, 2020))
    sub, w = ig.sample_negatives(df, "y_a", 20, np.random.default_rng(0))
    assert sub["y_a"].sum() == df["y_a"].sum()
    assert set(w[sub["y_a"].to_numpy() > 0]) == {1.0} and set(w[sub["y_a"].to_numpy() == 0]) == {20.0}


def test_full_protocol_produces_every_table(tmp_path):
    df = _store()
    p = ig.Protocol(grid={"num_leaves": [7], "min_data_in_leaf": [20]}, boot_reps=20, max_rounds=60, early_stopping=10, num_threads=2, block_folds=3, block_deg=0.5)
    res = ig.run_protocol(df, [2021, 2022], p, out_dir=tmp_path)
    assert set(res["metrics"]["model"]) == set(ig.MODELS) and set(res["metrics"]["held_year"]) == {2021, 2022}
    assert len(res["metrics"]) == 10 and res["metrics"]["pr_auc"].notna().all()
    assert set(res["criterion"]) == {"2021", "2022", "all_pass", "thresholds"}
    assert {"brier_rel_improvement", "pr_auc_gain", "pr_auc_rel_gain", "passes"} <= set(res["criterion"]["2021"])
    assert (res["shap"]["scope"] == "all").sum() == len(ig.full_features(df, "a"))
    assert len(res["spatial_blocks"]) == 2 and res["spatial_blocks"]["n"].sum() > 0
    assert len(res["predictions"]) == (df["year"].isin([2021, 2022])).sum()
    assert (tmp_path / "criterion_a.json").exists() and (tmp_path / "predictions_a.parquet").exists()
    cells = df[["cell", "country_key"]].drop_duplicates()
    summary = ig.monthly_country_summary(res["predictions"], cells)
    assert {"country_key", "month_key", "ignitions", "p_full_mean"} <= set(summary.columns)
