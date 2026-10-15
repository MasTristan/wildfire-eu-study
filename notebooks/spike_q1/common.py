"""Shared constants and helpers for the Q1 spike (ignition prediction, ES/PT/GR, 2019-2023).

Everything the spike downloads or derives lives under data/raw/spike_q1 and
data/interim/spike_q1 (both gitignored). Every step is resumable: an existing,
non-empty output file is skipped.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from src import config

YEARS = [2019, 2020, 2021, 2022, 2023]
TRAIN_YEARS = [2019, 2020, 2021]
TEST_YEARS = [2022, 2023]
COUNTRIES = ["ES", "PT", "GR"]

# CDS "area" order is [N, W, S, E]. Boxes are padded; cells are masked to the
# three countries afterwards with the shared geo_ref point-in-polygon helper.
AREAS = {
    "iberia": [44.0, -10.0, 35.5, 4.5],
    "greece": [42.0, 19.0, 34.5, 30.0],
}
GRID = 0.25

RAW = config.DATA_RAW / "spike_q1"
INTERIM = config.DATA_INTERIM / "spike_q1"
OUT = Path(__file__).resolve().parent / "out"  # small tables and figures, committed
TIMINGS = INTERIM / "timings.json"

# fire-weather components requested from cems-fire-historical-v1
FWI_VARS = {
    "fire_weather_index": "fwinx",
    "fine_fuel_moisture_code": "ffmcode",
    "duff_moisture_code": "dufmcode",
    "drought_code": "drtcode",
    "initial_fire_spread_index": "infsinx",
    "build_up_index": "fbupinx",
    "fire_daily_severity_rating": "fdsrte",
}
# NetCDF short name -> model column\nFWI_RENAME = dict(zip(FWI_VARS.values(), ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "dsr"], strict=True))\nFWI_COLS = list(FWI_RENAME.values())


def ensure_dirs() -> None:
    for d in (RAW, INTERIM, OUT):
        d.mkdir(parents=True, exist_ok=True)


def done(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def snap(values: np.ndarray, step: float = GRID) -> np.ndarray:
    """Round coordinates to the 0.25 deg lattice so grids from different sources join exactly."""
    return np.round(np.asarray(values, dtype="float64") / step) * step


def cell_id(lon, lat) -> np.ndarray:
    """Integer cell id on the 0.25 deg lattice (lon, lat are cell centres)."""
    ix = np.round((np.asarray(lon) + 180.0) / GRID).astype("int64")
    iy = np.round((np.asarray(lat) + 90.0) / GRID).astype("int64")
    return iy * 10_000 + ix


@contextmanager
def timed(label: str):
    """Record wall time of a stage in timings.json (they size phase 2)."""
    ensure_dirs()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        data = json.loads(TIMINGS.read_text()) if TIMINGS.exists() else {}
        data[label] = round(dt, 1)
        TIMINGS.write_text(json.dumps(data, indent=2))
        print(f"[time] {label}: {dt:.1f}s")
