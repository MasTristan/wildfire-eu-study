"""Step 2: build the cell x day table (features and both label variants).

Reads data/raw/spike_q1 (downloads of s01 to s04), data/raw/edo (SPEI-3, v1) and
data/processed/fact_fireevent.parquet (EFFIS perimeters, v1). Writes
data/interim/spike_q1/{cells,daily}.parquet. Everything is known at day J except
the labels. Run after the downloads finish:

    uv run python notebooks/spike_q1/s05_build_dataset.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

from src import config  # noqa: E402
from src.ingest import geo_ref  # noqa: E402

DATES = pd.date_range("2019-01-01", "2023-12-31", freq="D")
LAG_START = pd.Timestamp("2016-01-01")  # EFFIS history used for the lag counts


# --------------------------------------------------------------------- readers
def open_any(path: Path) -> xr.Dataset:
    """CDS answers are plain NetCDF or a zip of NetCDF files (named .zip either way)."""
    if zipfile.is_zipfile(path):
        out = path.parent / (path.stem + "_unzipped")
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        parts = [xr.open_dataset(p) for p in sorted(out.glob("*.nc"))]
        ds = xr.merge(parts, compat="override")
    else:
        ds = xr.open_dataset(path)
    drop = [v for v in ("number", "expver", "surface", "band") if v in ds.variables]
    ds = ds.drop_vars(drop, errors="ignore")
    for old in ("valid_time", "time"):
        if old in ds.dims and old != "time":
            ds = ds.rename({old: "time"})
    return ds


def _frame(ds: xr.Dataset, variables: list[str]) -> pd.DataFrame:
    df = ds[variables].to_dataframe().reset_index()
    df["lon"] = common.snap(df["longitude"])
    df["lat"] = common.snap(df["latitude"])
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.drop(columns=["longitude", "latitude", "time"])


# ----------------------------------------------------------------------- stages
def build_fwi() -> pd.DataFrame:
    parts = []
    for name in common.AREAS:
        for year in common.YEARS:
            ds = open_any(common.RAW / f"fwi_{name}_{year}.zip")
            df = _frame(ds, list(common.FWI_RENAME)).rename(columns=common.FWI_RENAME)
            parts.append(df.dropna(subset=["fwi"]))
    fwi = pd.concat(parts, ignore_index=True)
    return fwi.drop_duplicates(["lon", "lat", "date"])


def build_cells(fwi: pd.DataFrame) -> pd.DataFrame:
    """Mesh: cells with a defined FWI whose centre is inside ES, PT or GR."""
    uniq = fwi[["lon", "lat"]].drop_duplicates()
    mask = geo_ref.grid_mask(np.sort(uniq["lon"].unique()), np.sort(uniq["lat"].unique()))
    mask = mask[mask["country_key"].isin(common.COUNTRIES)]
    cells = uniq.merge(mask.assign(lon=common.snap(mask["lon"]), lat=common.snap(mask["lat"])), on=["lon", "lat"])
    cells["cell"] = common.cell_id(cells["lon"], cells["lat"])
    return cells.drop_duplicates("cell").reset_index(drop=True)


def build_era5() -> pd.DataFrame:
    parts = []
    for name in common.AREAS:
        for year in common.YEARS:
            for month in range(1, 13):
                mean = open_any(common.RAW / f"era5_mean_{name}_{year}_{month:02d}.nc")
                tot = open_any(common.RAW / f"era5_sum_{name}_{year}_{month:02d}.nc")
                ds = xr.merge([mean, tot], compat="override")
                df = _frame(ds, ["t2m", "d2m", "u10", "v10", "tp"])
                parts.append(df)
    e = pd.concat(parts, ignore_index=True).drop_duplicates(["lon", "lat", "date"])
    t = e["t2m"] - 273.15
    td = e["d2m"] - 273.15

    def es(x):
        return 6.1094 * np.exp(17.625 * x / (x + 243.04))

    e["t2m_c"] = t
    e["d2m_c"] = td
    e["rh"] = (100 * es(td) / es(t)).clip(0, 100)
    e["wind"] = np.hypot(e["u10"], e["v10"])
    e["tp_mm"] = e["tp"] * 1000.0
    return e[["lon", "lat", "date", "t2m_c", "d2m_c", "rh", "wind", "u10", "v10", "tp_mm"]]


def build_static() -> pd.DataFrame:
    parts = []
    for name in common.AREAS:
        ds = open_any(common.RAW / f"static_{name}.nc")
        df = _frame(ds.isel(time=0, drop=True) if "time" in ds.dims else ds, ["cvh", "cvl", "tvh", "tvl", "lsm", "z"])
        parts.append(df.drop(columns=["date"], errors="ignore"))
    s = pd.concat(parts, ignore_index=True).drop_duplicates(["lon", "lat"])
    s["elev_m"] = s["z"] / 9.80665
    return s.drop(columns=["z"])


def build_spei(cells: pd.DataFrame) -> pd.DataFrame:
    """SPEI-3 of the last COMPLETED dekad before day J (a dekad's value uses data to its end)."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform(
        cells["lon"].to_numpy(), cells["lat"].to_numpy()
    )
    pieces = []
    for path in sorted((config.DATA_RAW / "edo").glob("sem03_m_edo_*.nc")):
        ds = xr.open_dataset(path)
        da = ds["sem03"].isel(band=0, drop=True)
        da = da.rename({"lon": "x", "lat": "y"})
        pts = da.sel(
            x=xr.DataArray(x, dims="cell"), y=xr.DataArray(y, dims="cell"), method="nearest"
        ).load()
        vals = pts.to_numpy().astype("float32")
        vals[~np.isfinite(vals)] = np.nan
        start = pd.to_datetime(pts["time"].to_numpy())
        pieces.append((start, vals))
    start = np.concatenate([p[0] for p in pieces])
    vals = np.concatenate([p[1] for p in pieces], axis=0)
    order = np.argsort(start)
    start, vals = start[order], vals[order]
    # a dekad starting the 1st or 11th ends 9 days later, the 21st one ends at month end
    end = np.where(
        pd.DatetimeIndex(start).day < 21,
        start + np.timedelta64(9, "D"),
        (pd.DatetimeIndex(start) + pd.offsets.MonthEnd(0)).to_numpy(),
    )
    idx = np.searchsorted(end, DATES.to_numpy(), side="left") - 1  # last end < day
    valid = idx >= 0
    out = np.full((len(DATES), len(cells)), np.nan, dtype="float32")
    out[valid] = vals[idx[valid]]
    return pd.DataFrame(
        {"cell": np.tile(cells["cell"].to_numpy(), len(DATES)), "date": np.repeat(DATES, len(cells)),
         "spei3": out.ravel()}
    )


def labels_effis(cells: pd.DataFrame) -> pd.DataFrame:
    fe = pd.read_parquet(config.DATA_PROCESSED / "fact_fireevent.parquet")
    fe = fe[(fe["source_key"] == "EFFIS_RDA") & fe["centroid_lat"].notna() & fe["firedate_key"].notna()]
    fe = fe[(fe["firedate_key"] >= LAG_START) & (fe["firedate_key"] <= DATES[-1])]
    fe = fe.assign(
        cell=common.cell_id(common.snap(fe["centroid_lon"]), common.snap(fe["centroid_lat"])),
        date=fe["firedate_key"].dt.normalize(),
    )
    fe = fe[fe["cell"].isin(cells["cell"])]
    return fe.groupby(["cell", "date"]).size().rename("n_start").reset_index()


def labels_viirs(cells: pd.DataFrame) -> pd.DataFrame:
    """First day of a hotspot cluster: DBSCAN 2 km per day, then chain to the day before."""
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import BallTree

    files = sorted((common.RAW / "viirs").glob("*.csv"))
    raw = pd.concat([pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "confidence", "type"],
                                 dtype={"confidence": str}) for f in files], ignore_index=True)
    raw = raw[(raw["type"] == 0) & (raw["confidence"] != "l")].drop_duplicates()
    raw["date"] = pd.to_datetime(raw["acq_date"])
    eps = 2.0 / 6371.0088
    rows = []
    prev_tree = None
    prev_day = None
    for day, g in raw.groupby("date", sort=True):
        pts = np.radians(g[["latitude", "longitude"]].to_numpy())
        lab = DBSCAN(eps=eps, min_samples=1, metric="haversine", algorithm="ball_tree").fit_predict(pts)
        if prev_tree is not None and (day - prev_day).days == 1:
            near = prev_tree.query_radius(pts, r=eps, count_only=True) > 0
        else:
            near = np.zeros(len(pts), dtype=bool)
        d = pd.DataFrame({"lab": lab, "near": near, "lat": g["latitude"].to_numpy(), "lon": g["longitude"].to_numpy()})
        cl = d.groupby("lab").agg(near=("near", "any"), lat=("lat", "mean"), lon=("lon", "mean"))
        new = cl[~cl["near"]]
        rows.append(pd.DataFrame({"date": day, "lat": new["lat"], "lon": new["lon"]}))
        prev_tree, prev_day = BallTree(pts, metric="haversine"), day
    nv = pd.concat(rows, ignore_index=True)
    nv["cell"] = common.cell_id(common.snap(nv["lon"]), common.snap(nv["lat"]))
    nv = nv[nv["cell"].isin(cells["cell"])]
    return nv.groupby(["cell", "date"]).size().rename("n_start").reset_index()


def lag_counts(lab: pd.DataFrame, cells: pd.DataFrame, start: pd.Timestamp, tag: str) -> pd.DataFrame:
    """Starts in the 7 and 30 days BEFORE day J (J excluded), per cell."""
    days = pd.date_range(start, DATES[-1], freq="D")
    piv = (
        lab.pivot_table(index="date", columns="cell", values="n_start", aggfunc="sum")
        .reindex(index=days, columns=cells["cell"]).fillna(0.0)
    )
    out = {}
    for w in (7, 30):
        out[f"lag{tag}_{w}"] = piv.rolling(w, min_periods=1).sum().shift(1).loc[DATES]
    frames = [v.stack().rename(k) for k, v in out.items()]
    res = pd.concat(frames, axis=1).reset_index()
    res.columns = ["date", "cell", *out.keys()]
    return res


def main() -> None:
    common.ensure_dirs()
    with common.timed("build_fwi"):
        fwi = build_fwi()
        cells = build_cells(fwi)
        fwi["cell"] = common.cell_id(fwi["lon"], fwi["lat"])
        fwi = fwi[fwi["cell"].isin(cells["cell"])].drop(columns=["lon", "lat"])
    print(f"mesh: {len(cells)} cells ({cells['country_key'].value_counts().to_dict()}), fwi rows {len(fwi):,}")

    with common.timed("build_era5"):
        era = build_era5()
        era["cell"] = common.cell_id(era["lon"], era["lat"])
        era = era[era["cell"].isin(cells["cell"])].drop(columns=["lon", "lat"])
    static = build_static()
    static["cell"] = common.cell_id(static["lon"], static["lat"])
    static = static[static["cell"].isin(cells["cell"])].drop(columns=["lon", "lat"])
    with common.timed("build_spei"):
        spei = build_spei(cells)

    with common.timed("build_labels"):
        la = labels_effis(cells)
        lb = labels_viirs(cells)
        lag_a = lag_counts(la, cells, LAG_START, "a")
        lag_b = lag_counts(lb, cells, pd.Timestamp("2019-01-01"), "b")

    with common.timed("assemble"):
        grid = pd.DataFrame(
            {"cell": np.tile(cells["cell"].to_numpy(), len(DATES)), "date": np.repeat(DATES, len(cells))}
        )
        df = grid.merge(fwi, on=["cell", "date"], how="left").merge(era, on=["cell", "date"], how="left")
        df = df.merge(spei, on=["cell", "date"], how="left").merge(static, on="cell", how="left")
        df = df.merge(lag_a, on=["cell", "date"], how="left").merge(lag_b, on=["cell", "date"], how="left")
        df = df.merge(la.rename(columns={"n_start": "n_a"}), on=["cell", "date"], how="left")
        df = df.merge(lb.rename(columns={"n_start": "n_b"}), on=["cell", "date"], how="left")
        df[["n_a", "n_b"]] = df[["n_a", "n_b"]].fillna(0).astype("int16")
        df["y_a"] = (df["n_a"] > 0).astype("int8")
        df["y_b"] = (df["n_b"] > 0).astype("int8")
        df["doy"] = df["date"].dt.dayofyear.astype("int16")
        df["year"] = df["date"].dt.year.astype("int16")
        # rolling 7-day rain before J (J excluded), per cell
        df = df.sort_values(["cell", "date"]).reset_index(drop=True)
        df["tp_7d"] = (
            df.groupby("cell")["tp_mm"].transform(lambda s: s.rolling(7, min_periods=3).sum().shift(1))
        )
        # the mesh keeps only days where the FWI exists (sea/NaN days drop out)
        df = df[df["fwi"].notna()].reset_index(drop=True)
        f32 = df.select_dtypes("float64").columns
        df[f32] = df[f32].astype("float32")
    cells.to_parquet(common.INTERIM / "cells.parquet")
    df.to_parquet(common.INTERIM / "daily.parquet")
    print(f"daily table: {len(df):,} rows, a-positives {int(df['y_a'].sum()):,}, b-positives {int(df['y_b'].sum()):,}")


if __name__ == "__main__":
    main()
