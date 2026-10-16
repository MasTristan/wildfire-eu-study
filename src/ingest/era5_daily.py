"""Daily ERA5 weather per 0.25 degree cell from the hourly CDS reanalysis
(`reanalysis-era5-single-levels`), one request per month and area, reduced
to UTC daily values locally (ADR 0007).

Why hourly and not the derived daily-statistics dataset: a monthly hourly
request is answered in minutes by MARS, while the daily statistics are
computed on demand in a queue that took hours per request. Why not
Open-Meteo at this scale: its free tier allows about a hundred cell-years
per day (ADR 0006, superseded for full-scale use).

Daily variables written per month (`era5_daily_<area>_<year>_<month>.nc`):
t2m_c, t2m_max, rh, rh_min, wind, wind_mean (all from the hourly fields),
tp_mm (sum of the hourly accumulations). Humidity by Magnus-Tetens, as in
src/ingest/era5.py. Precipitation stamps are accumulations over the hour
BEFORE the stamp, so UTC day D is the sum of stamps D 01:00 to D+1 00:00.
"""
from __future__ import annotations

import shutil
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from src import config

DATASET = "reanalysis-era5-single-levels"
VARIABLES = ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
             "10m_v_component_of_wind", "total_precipitation"]
EUROPE: list[float] = [72.0, -25.0, 34.0, 45.0]  # N, W, S, E
DAILY_VARS = ["t2m_c", "t2m_max", "rh", "rh_min", "wind", "wind_mean", "tp_mm"]


def _es(t_c):
    """Saturation vapour pressure (hPa), Magnus-Tetens."""
    return 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))


def open_hourly(path: Path) -> xr.Dataset:
    """A CDS answer: one NetCDF, or a zip of two (instant and accumulated variables)."""
    if zipfile.is_zipfile(path):
        out = path.with_suffix("")
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        ds = xr.merge([xr.open_dataset(f) for f in sorted(out.glob("*.nc"))], compat="override")
    else:
        ds = xr.open_dataset(path)
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.variables], errors="ignore")
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds


def hourly_to_daily(ds: xr.Dataset) -> xr.Dataset:
    t = ds["t2m"] - 273.15
    td = ds["d2m"] - 273.15
    rh = (100.0 * _es(td) / _es(t)).clip(0, 100)
    ws = xr.apply_ufunc(np.hypot, ds["u10"], ds["v10"])
    daily = xr.Dataset({
        "t2m_c": t.resample(time="1D").mean(),
        "t2m_max": t.resample(time="1D").max(),
        "rh": rh.resample(time="1D").mean(),
        "rh_min": rh.resample(time="1D").min(),
        "wind": ws.resample(time="1D").max(),
        "wind_mean": ws.resample(time="1D").mean(),
        "tp_mm": (ds["tp"].shift(time=-1).resample(time="1D").sum() * 1000.0),
    })
    for v in DAILY_VARS:
        daily[v] = daily[v].astype("float32")
    return daily


def month_request(year: int, month: int, area: Sequence[float]) -> dict:
    return {
        "product_type": ["reanalysis"], "variable": VARIABLES,
        "year": [str(year)], "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": list(area), "grid": "0.25/0.25", "data_format": "netcdf",
    }


def download_month(year: int, month: int, area: Sequence[float], target_daily: Path, client=None,
                   keep_hourly: bool = False, sleep=time.sleep) -> Path:
    """Fetch one month of hourly fields, reduce to daily, write `target_daily`; resumable."""
    if target_daily.exists() and target_daily.stat().st_size > 0:
        return target_daily
    target_daily.parent.mkdir(parents=True, exist_ok=True)
    hourly = target_daily.with_name(target_daily.stem.replace("era5_daily", "era5h") + ".nc")
    if not (hourly.exists() and hourly.stat().st_size > 0):
        client = client or config.cds_client("cds")
        for _attempt in range(60):
            try:
                client.retrieve(DATASET, month_request(year, month, area), str(hourly))
                break
            except Exception as exc:  # cdsapi raises bare exceptions on queue limits
                if any(k in str(exc) for k in ("limited", "rejected", "queued")):
                    sleep(120)
                    continue
                raise
    ds = open_hourly(hourly)
    hourly_to_daily(ds).to_netcdf(target_daily)
    ds.close()
    if not keep_hourly:
        hourly.unlink(missing_ok=True)
        shutil.rmtree(hourly.with_suffix(""), ignore_errors=True)
    return target_daily


def daily_path(raw_dir: Path, area_name: str, year: int, month: int) -> Path:
    return raw_dir / f"era5_daily_{area_name}_{year}_{month:02d}.nc"
