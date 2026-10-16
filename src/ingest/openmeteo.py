"""Daily ERA5 weather per grid cell from the Open-Meteo historical archive
(ADR 0006): model `era5` pinned, 0.25 degree, daily aggregates computed
server side in UTC. No key; CC BY 4.0, non-commercial use within the
published rate limits.

Layout of the bronze output: one JSON per batch of cell centres under
`out_dir`, plus `variables.json` recording what the API actually served.
Resumable batch by batch; a quota stop raises SystemExit with the server's
message so the run can be relaunched later and resume.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
BATCH = 20
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "vapour_pressure_deficit_max",
    "et0_fao_evapotranspiration",
]
# API name -> feature-store column
RENAME = {
    "temperature_2m_mean": "t2m_c",
    "temperature_2m_max": "t2m_max",
    "relative_humidity_2m_mean": "rh",
    "relative_humidity_2m_min": "rh_min",
    "wind_speed_10m_max": "wind",
    "wind_speed_10m_mean": "wind_mean",
    "precipitation_sum": "tp_mm",
    "vapour_pressure_deficit_max": "vpd_max",
    "et0_fao_evapotranspiration": "et0",
}
BASE_PARAMS = {"models": "era5", "timezone": "UTC", "wind_speed_unit": "ms"}

Getter = Callable[..., Any]  # requests.Session.get or a test double


class QuotaExhausted(SystemExit):
    pass


def request_batch(get: Getter, lats: list[float], lons: list[float], variables: list[str],
                  start: str, end: str, sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    params = {
        **BASE_PARAMS,
        "start_date": start,
        "end_date": end,
        "latitude": ",".join(f"{v:.3f}" for v in lats),
        "longitude": ",".join(f"{v:.3f}" for v in lons),
        "daily": ",".join(variables),
    }
    for attempt in range(30):
        r = get(URL, params=params, timeout=180)
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else [data]
        body = r.text[:300]
        low = body.lower()
        if r.status_code == 429 or "limit" in low:
            if "minute" in low:
                sleep(70)
            elif "hour" in low:
                sleep(3700)
            else:
                raise QuotaExhausted(f"Open-Meteo daily quota reached, relaunch later (resumable). Server: {body}")
            continue
        if r.status_code == 400:
            raise ValueError(body)
        sleep(10 * (attempt + 1))
    raise RuntimeError("Open-Meteo: gave up after 30 attempts")


def probe(get: Getter, lat: float, lon: float, start: str, end: str,
          variables: list[str] | None = None) -> list[str]:
    """The subset of `variables` the API serves, dropping rejected names one by one."""
    remaining = list(variables or DAILY_VARS)
    while remaining:
        try:
            request_batch(get, [lat], [lon], remaining, start, end)
            return remaining
        except ValueError as exc:
            bad = [v for v in remaining if v in str(exc)]
            if not bad:
                raise
            remaining = [v for v in remaining if v not in bad]
    raise RuntimeError("Open-Meteo served none of the requested daily variables")


def download(cells: pd.DataFrame, out_dir: Path, start: str, end: str, get: Getter | None = None,
             batch: int = BATCH, pause: float = 0.3, sleep: Callable[[float], None] = time.sleep) -> int:
    """Fetch every batch of `cells` (columns cell, lat, lon) not yet on disk; returns batches written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if get is None:
        sess = requests.Session()
        sess.headers["User-Agent"] = "wildfire-eu-study (research, non-commercial)"
        get = sess.get
    cells = cells.sort_values("cell").reset_index(drop=True)
    served_path = out_dir / "variables.json"
    if served_path.exists():
        variables = json.loads(served_path.read_text())["daily"]
    else:
        variables = probe(get, float(cells.loc[0, "lat"]), float(cells.loc[0, "lon"]), start, end)
        served_path.write_text(json.dumps({"daily": variables, "params": BASE_PARAMS, "start": start, "end": end}, indent=2))
    n_batches = (len(cells) + batch - 1) // batch
    written = 0
    for b in range(n_batches):
        target = out_dir / f"batch_{b:05d}.json"
        if target.exists() and target.stat().st_size > 0:
            continue
        chunk = cells.iloc[b * batch : (b + 1) * batch]
        data = request_batch(get, chunk["lat"].tolist(), chunk["lon"].tolist(), variables, start, end, sleep)
        if len(data) != len(chunk):
            raise RuntimeError(f"batch {b}: {len(data)} answers for {len(chunk)} cells")
        for rec, cell in zip(data, chunk["cell"].tolist(), strict=True):
            rec["cell"] = int(cell)
        target.write_text(json.dumps(data))
        written += 1
        sleep(pause)
    return written


def load(out_dir: Path, cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily weather table: cell, date, one feature-store column per served variable."""
    frames = []
    for path in sorted(out_dir.glob("batch_*.json")):
        for rec in json.loads(path.read_text()):
            d = rec["daily"]
            df = pd.DataFrame({RENAME.get(k, k): v for k, v in d.items() if k != "time"})
            df.insert(0, "date", pd.to_datetime(d["time"]))
            df.insert(0, "cell", int(rec["cell"]))
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no Open-Meteo batches under {out_dir}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["cell", "date"])
    if cells is not None:
        out = out[out["cell"].isin(cells["cell"])]
    return out.reset_index(drop=True)
