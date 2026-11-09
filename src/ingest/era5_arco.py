"""Hourly ERA5 from the ARCO-ERA5 public Zarr store on Google Cloud, read
over anonymous HTTPS, cropped to an area and reduced to UTC daily values
with the same function as the CDS path (src/ingest/era5_daily.py).

Fallback for the Q1 spike when the CDS queue stalls (PROTOCOL.md amendment
3, ADR 0008). The store is chunked one hour x the whole globe, so a month of
five variables reads about 9.8 GB to keep a few hundred kilobytes: each
chunk is decoded, cropped and dropped, nothing raw is kept on disk.

Facts verified on the store on 2026-11-07, re-checked at run time by
`check_store`: time index = hours since 1900-01-01 00:00, latitude 90 to -90
by 0.25, longitude 0 to 359.75 by 0.25, float32 Blosc-compressed chunks
named `<time>.0.0`.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.ingest import era5_daily

BASE = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VARIABLES = {  # short name used by the CDS files -> ARCO array name
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tp": "total_precipitation",
}
EPOCH = datetime(1900, 1, 1)
LAT = np.linspace(90.0, -90.0, 721)
LON = np.arange(1440) * 0.25

Getter = Callable[..., Any]


def time_index(ts: pd.Timestamp | datetime) -> int:
    return int((pd.Timestamp(ts).to_pydatetime() - EPOCH).total_seconds() // 3600)


def crop_indices(area: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of a CDS-style area [N, W, S, E], longitudes in -180..180.

    Rows run north to south and columns west to east, the order a CDS request
    with the same area returns; an area crossing the Greenwich meridian takes
    the columns at the end of the 0..360 axis first.
    """
    n, w, s, e = (float(v) for v in area)
    rows = np.flatnonzero((LAT <= n + 1e-6) & (LAT >= s - 1e-6))
    lon180 = np.where(LON >= 180.0, LON - 360.0, LON)
    cols = np.flatnonzero((lon180 >= w - 1e-6) & (lon180 <= e + 1e-6))
    cols = cols[np.argsort(lon180[cols])]
    return rows, cols


def area_coords(area: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = crop_indices(area)
    lon180 = np.where(LON >= 180.0, LON - 360.0, LON)
    return LAT[rows], lon180[cols]


class Store:
    """Minimal anonymous reader of the ARCO store: metadata once, then chunks."""

    def __init__(self, get: Getter | None = None, base: str = BASE):
        import requests

        self.base = base
        if get is None:
            sess = requests.Session()
            sess.headers["User-Agent"] = "wildfire-eu-study (research)"
            get = sess.get
        self.get = get
        self._meta: dict | None = None

    def meta(self) -> dict:
        if self._meta is None:
            r = self._get(f"{self.base}/.zmetadata")
            self._meta = r.json()["metadata"]
        return self._meta

    def _get(self, url: str, attempts: int = 8):
        for i in range(attempts):
            r = self.get(url, timeout=120)
            if r.status_code == 200:
                return r
            if r.status_code in (404,):
                raise FileNotFoundError(url)
            time.sleep(min(60, 2 ** i))
        raise RuntimeError(f"ARCO: gave up on {url} (last HTTP {r.status_code})")

    def _decode(self, name: str, raw: bytes) -> np.ndarray:
        import numcodecs

        m = self.meta()[f"{name}/.zarray"]
        return np.frombuffer(numcodecs.get_codec(m["compressor"]).decode(raw), dtype=m["dtype"])

    def coord(self, name: str) -> np.ndarray:
        return self._decode(name, self._get(f"{self.base}/{name}/0").content)

    def field(self, var: str, t_index: int) -> np.ndarray:
        arr = self._decode(var, self._get(f"{self.base}/{var}/{t_index}.0.0").content)
        return arr.reshape(721, 1440)

    def check_store(self) -> dict:
        """Re-verify the layout facts this module relies on; raise on any drift."""
        m = self.meta()
        lat, lon = self.coord("latitude"), self.coord("longitude")
        t0 = self._decode("time", self._get(f"{self.base}/time/0").content)
        problems = []
        if not np.allclose(lat, LAT):
            problems.append("latitude axis changed")
        if not np.allclose(lon, LON):
            problems.append("longitude axis changed")
        if int(t0[0]) != 0 or int(t0[1] - t0[0]) != 1:
            problems.append("time axis no longer hourly from 1900-01-01")
        for v in VARIABLES.values():
            if m[f"{v}/.zarray"]["chunks"] != [1, 721, 1440]:
                problems.append(f"{v} chunking changed")
        if problems:
            raise RuntimeError("ARCO store layout drifted: " + "; ".join(problems))
        attrs = m.get(".zattrs", {})
        return {"valid_time_stop": attrs.get("valid_time_stop"), "last_updated": attrs.get("last_updated")}


def month_hourly(store: Store, year: int, month: int, areas: Mapping[str, Sequence[float]],
                 workers: int = 8) -> dict[str, xr.Dataset]:
    """Hourly fields of one month for each area, shaped like a CDS answer
    (dims time, latitude, longitude; variables t2m, d2m, u10, v10, tp)."""
    hours = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Timestamp(year=year, month=month, day=1).days_in_month * 24, freq="h")
    idx = {name: crop_indices(a) for name, a in areas.items()}
    out = {name: {v: np.empty((len(hours), len(r), len(c)), dtype="float32") for v in VARIABLES}
           for name, (r, c) in idx.items()}

    def job(args):
        k, short, var = args
        f = store.field(var, time_index(hours[k]))
        for name, (r, c) in idx.items():
            out[name][short][k] = f[np.ix_(r, c)]

    tasks = [(k, short, var) for k in range(len(hours)) for short, var in VARIABLES.items()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(job, tasks))
    result = {}
    for name, a in areas.items():
        lat, lon = area_coords(a)
        result[name] = xr.Dataset(
            {v: (("time", "latitude", "longitude"), out[name][v]) for v in VARIABLES},
            coords={"time": hours, "latitude": lat, "longitude": lon},
        )
    return result


def download_month(store: Store, year: int, month: int, areas: Mapping[str, Sequence[float]],
                   targets: Mapping[str, Path], workers: int = 8) -> dict[str, Path]:
    """Daily files for every area of the month whose target is missing; resumable."""
    todo = {n: a for n, a in areas.items() if not (targets[n].exists() and targets[n].stat().st_size > 0)}
    if not todo:
        return dict(targets)
    hourly = month_hourly(store, year, month, todo, workers)
    for name, ds in hourly.items():
        targets[name].parent.mkdir(parents=True, exist_ok=True)
        tmp = targets[name].with_suffix(".tmp.nc")
        era5_daily.hourly_to_daily(ds).to_netcdf(tmp)
        tmp.replace(targets[name])
    return dict(targets)


TOLERANCES = {"t2m_c": 0.01, "t2m_max": 0.01, "rh": 0.1, "rh_min": 0.1, "wind": 0.01, "wind_mean": 0.01, "tp_mm": 0.01}


def compare_daily(reference: Path, candidate: Path, tolerances: Mapping[str, float] = TOLERANCES) -> pd.DataFrame:
    """Max absolute difference per variable between two daily files on the same
    grid and days; `passes` is True when every variable is within tolerance."""
    a, b = era5_daily.open_hourly(reference), era5_daily.open_hourly(candidate)
    rows = []
    for v, tol in tolerances.items():
        if v not in a or v not in b:
            rows.append({"variable": v, "max_abs_diff": float("nan"), "tolerance": tol, "passes": False, "note": "missing"})
            continue
        x, y = a[v].load(), b[v].load()
        same_grid = (x.sizes == y.sizes and np.allclose(x["latitude"], y["latitude"]) and np.allclose(x["longitude"], y["longitude"]))
        if not same_grid:
            rows.append({"variable": v, "max_abs_diff": float("nan"), "tolerance": tol, "passes": False, "note": "grid or days differ"})
            continue
        d = float(np.nanmax(np.abs(x.values - y.values)))
        rows.append({"variable": v, "max_abs_diff": d, "tolerance": tol, "passes": d <= tol, "note": ""})
    a.close()
    b.close()
    return pd.DataFrame(rows)
