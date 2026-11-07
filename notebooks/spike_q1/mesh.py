"""The spike's mesh: 0.25 degree cells of the FWI lattice whose centre lies in
Spain, Portugal or Greece. Shared by the weather download (which needs the
cell centres) and the dataset build (which joins everything on the cell id),
so both use exactly the same cells.
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

from src.ingest import geo_ref  # noqa: E402


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


def frame(ds: xr.Dataset, variables: list[str]) -> pd.DataFrame:
    df = ds[variables].to_dataframe().reset_index()
    df["lon"] = common.snap(df["longitude"])
    df["lat"] = common.snap(df["latitude"])
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.drop(columns=["longitude", "latitude", "time"])


def build_fwi() -> pd.DataFrame:
    parts = []
    for name in common.AREAS:
        for year in common.YEARS:
            ds = open_any(common.RAW / f"fwi_{name}_{year}.zip")
            df = frame(ds, list(common.FWI_RENAME)).rename(columns=common.FWI_RENAME)
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


def load_cells() -> pd.DataFrame:
    """Cells from the cached parquet if the dataset build already wrote it, else from the FWI files."""
    cached = common.INTERIM / "cells.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    cells = build_cells(build_fwi())
    common.ensure_dirs()
    cells.to_parquet(cached)
    return cells
