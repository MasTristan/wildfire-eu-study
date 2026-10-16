"""The feature store at cell x day (gold/features): every column known at
day J, labels and lagged ignition counts built from the past only.

Generalises notebooks/spike_q1/s05_build_dataset.py to the whole grid and
to one year at a time, so a year is one Dagster partition and one Parquet.
Each builder takes explicit inputs and returns a frame keyed by (cell, date)
or (cell), so all of them are testable on synthetic inputs; `assemble` joins
them and applies the leakage rules documented on each function.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src import grid

FWI_VARS = {  # NetCDF short name -> column
    "fwinx": "fwi", "ffmcode": "ffmc", "dufmcode": "dmc", "drtcode": "dc",
    "infsinx": "isi", "fbupinx": "bui", "fdsrte": "dsr",
}
STATIC_VARS = {"cvh": "cvh", "cvl": "cvl", "tvh": "tvh", "tvl": "tvl", "lsm": "lsm", "z": "z"}


# ----------------------------------------------------------------------- NetCDF
def open_any(path: Path) -> xr.Dataset:
    """CDS answers are plain NetCDF or a zip of NetCDF files (named .zip either way)."""
    if zipfile.is_zipfile(path):
        out = path.parent / (path.stem + "_unzipped")
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        ds = xr.merge([xr.open_dataset(p) for p in sorted(out.glob("*.nc"))], compat="override")
    else:
        ds = xr.open_dataset(path)
    ds = ds.drop_vars([v for v in ("number", "expver", "surface", "band") if v in ds.variables], errors="ignore")
    for old in ("valid_time",):
        if old in ds.dims:
            ds = ds.rename({old: "time"})
    for old, new in (("lon", "longitude"), ("lat", "latitude")):
        if old in ds.dims:
            ds = ds.rename({old: new})
    return ds


def _frame(ds: xr.Dataset, variables: list[str], with_time: bool = True) -> pd.DataFrame:
    present = [v for v in variables if v in ds.data_vars]
    df = ds[present].to_dataframe().reset_index()
    out = pd.DataFrame({"cell": grid.cell_id(df["longitude"], df["latitude"])})
    if with_time:
        out["date"] = pd.to_datetime(df["time"]).dt.normalize().to_numpy()
    for v in present:
        out[v] = df[v].to_numpy()
    return out


# ---------------------------------------------------------------------- builders
def fwi_daily(nc_paths: list[Path], cells: pd.DataFrame) -> pd.DataFrame:
    """Daily FWI components per cell from the EWDS files; rows without FWI dropped."""
    parts = []
    for p in nc_paths:
        ds = open_any(p)
        df = _frame(ds, list(FWI_VARS)).rename(columns=FWI_VARS)
        parts.append(df[df["cell"].isin(cells["cell"])].dropna(subset=["fwi"]))
        ds.close()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["cell", "date"])


WEATHER_VARS = ["t2m_c", "t2m_max", "rh", "rh_min", "wind", "wind_mean", "tp_mm"]


def weather_daily(nc_paths: list[Path], cells: pd.DataFrame) -> pd.DataFrame:
    """Daily weather per cell from the monthly files of src/ingest/era5_daily.py."""
    parts = []
    for p in nc_paths:
        ds = open_any(p)
        df = _frame(ds, WEATHER_VARS)
        parts.append(df[df["cell"].isin(cells["cell"])])
        ds.close()
    return pd.concat(parts, ignore_index=True).drop_duplicates(["cell", "date"])


def static_fields(nc_paths: list[Path], cells: pd.DataFrame) -> pd.DataFrame:
    """Static ERA5 surface description per cell: vegetation cover and type, land fraction, elevation."""
    parts = []
    for p in nc_paths:
        ds = open_any(p)
        if "time" in ds.dims:
            ds = ds.isel(time=0, drop=True)
        df = _frame(ds, list(STATIC_VARS), with_time=False)
        parts.append(df[df["cell"].isin(cells["cell"])])
        ds.close()
    s = pd.concat(parts, ignore_index=True).drop_duplicates("cell")
    if "z" in s.columns:
        s["elev_m"] = s["z"] / 9.80665
        s = s.drop(columns=["z"])
    return s


def spei_daily(edo_paths: list[Path], cells: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """SPEI-3 known at day J: the last dekad whose END is strictly before J.

    EDO publishes one value per dekad (1st, 11th, 21st) computed with data to
    the dekad's end; using the dekad containing J would read the future.
    """
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform(
        cells["lon"].to_numpy(), cells["lat"].to_numpy()
    )
    starts, vals = [], []
    for path in sorted(edo_paths):
        ds = xr.open_dataset(path)
        da = ds["sem03"]
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)
        da = da.rename({"lon": "x", "lat": "y"}) if "lon" in da.dims else da
        pts = da.sel(x=xr.DataArray(x, dims="cell"), y=xr.DataArray(y, dims="cell"), method="nearest").load()
        v = pts.to_numpy().astype("float32")
        v[~np.isfinite(v)] = np.nan
        starts.append(pd.to_datetime(pts["time"].to_numpy()))
        vals.append(v)
        ds.close()
    start = pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in starts]))
    val = np.concatenate(vals, axis=0)
    order = np.argsort(start.to_numpy())
    start, val = start[order], val[order]
    end = np.where(start.day < 21, (start + pd.Timedelta(days=9)).to_numpy(), (start + pd.offsets.MonthEnd(0)).to_numpy())
    idx = np.searchsorted(end, dates.to_numpy(), side="left") - 1  # last end < day
    out = np.full((len(dates), len(cells)), np.nan, dtype="float32")
    ok = idx >= 0
    out[ok] = val[idx[ok]]
    return pd.DataFrame({
        "cell": np.tile(cells["cell"].to_numpy(), len(dates)),
        "date": np.repeat(dates.to_numpy(), len(cells)),
        "spei3": out.ravel(),
    })


def ignition_labels(events: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """(cell, date, n_start): ignitions per cell and day from a frame with
    firedate_key, centroid_lon, centroid_lat (fact_fireevent, EFFIS rows)."""
    ev = events.dropna(subset=["firedate_key", "centroid_lon", "centroid_lat"])
    lab = pd.DataFrame({
        "cell": grid.cell_id(ev["centroid_lon"], ev["centroid_lat"]),
        "date": pd.to_datetime(ev["firedate_key"]).dt.normalize().to_numpy(),
    })
    lab = lab[lab["cell"].isin(cells["cell"])]
    return lab.groupby(["cell", "date"]).size().rename("n_start").reset_index()


def lag_counts(labels: pd.DataFrame, cells: pd.DataFrame, dates: pd.DatetimeIndex,
               history_start: pd.Timestamp, windows: tuple[int, ...] = (7, 30), tag: str = "") -> pd.DataFrame:
    """Ignitions in the `w` days BEFORE day J, J excluded, per cell, for each window."""
    days = pd.date_range(min(history_start, dates[0]), dates[-1], freq="D")
    piv = (
        labels.pivot_table(index="date", columns="cell", values="n_start", aggfunc="sum")
        .reindex(index=days, columns=cells["cell"]).fillna(0.0)
    )
    cols = {}
    for w in windows:
        cols[f"lag{tag}_{w}"] = piv.rolling(w, min_periods=1).sum().shift(1).loc[dates]
    res = pd.concat([v.stack().rename(k) for k, v in cols.items()], axis=1).reset_index()
    res.columns = ["date", "cell", *cols.keys()]
    return res


def assemble(cells: pd.DataFrame, dates: pd.DatetimeIndex, fwi: pd.DataFrame, weather: pd.DataFrame | None,
             spei: pd.DataFrame | None, static: pd.DataFrame | None, labels: pd.DataFrame,
             lags: pd.DataFrame) -> pd.DataFrame:
    """Join everything on (cell, date); keep days with a defined FWI; float32."""
    df = pd.DataFrame({
        "cell": np.tile(cells["cell"].to_numpy(), len(dates)),
        "date": np.repeat(dates.to_numpy(), len(cells)),
    })
    df = df.merge(fwi, on=["cell", "date"], how="left")
    if weather is not None:
        df = df.merge(weather, on=["cell", "date"], how="left")
    if spei is not None:
        df = df.merge(spei, on=["cell", "date"], how="left")
    if static is not None:
        df = df.merge(static, on="cell", how="left")
    df = df.merge(lags, on=["cell", "date"], how="left")
    df = df.merge(labels.rename(columns={"n_start": "n_a"}), on=["cell", "date"], how="left")
    df["n_a"] = df["n_a"].fillna(0).astype("int16")
    df["y_a"] = (df["n_a"] > 0).astype("int8")
    df["doy"] = df["date"].dt.dayofyear.astype("int16")
    df["year"] = df["date"].dt.year.astype("int16")
    df = df.sort_values(["cell", "date"]).reset_index(drop=True)
    if "tp_mm" in df.columns:
        # rain over the 7 days before J, J excluded; needs the previous year's tail for January
        df["tp_7d"] = df.groupby("cell")["tp_mm"].transform(lambda s: s.rolling(7, min_periods=3).sum().shift(1))
    df = df[df["fwi"].notna()].reset_index(drop=True)
    f64 = df.select_dtypes("float64").columns
    df[f64] = df[f64].astype("float32")
    return df
