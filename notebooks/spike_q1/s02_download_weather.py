"""Step 1b: daily ERA5 weather per mesh cell, from the Open-Meteo historical
archive (model `era5`, 0.25 degree, the same reanalysis the FWI is computed
from), aggregated to daily values server side in UTC.

Why not the CDS: the first version of this step asked the CDS
`derived-era5-single-levels-daily-statistics` dataset for one file per
(statistic, month, area), 240 requests computed on demand in a queue that
did not drain (2026-10-15). Open-Meteo serves the same ERA5 fields with no
queue and no key; its daily aggregates are documented, and the request is
per cell centre, so the value is the cell's own.

Licence: Open-Meteo data are CC BY 4.0 (attribution in docs/sources.md),
free for non-commercial use within published rate limits. This script is
resumable (one JSON per batch of cells), backs off on 429, and stops cleanly
on the daily quota so it can be relaunched the next day.

    uv run python notebooks/spike_q1/s02_download_weather.py --probe   # one cell, prints the variables served
    uv run python notebooks/spike_q1/s02_download_weather.py           # everything, resumable
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import mesh  # noqa: E402

URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DIR = common.RAW / "openmeteo"
BATCH = 20  # cells per request; the API accepts coordinate lists
START, END = f"{common.YEARS[0]}-01-01", f"{common.YEARS[-1]}-12-31"

# Daily aggregates requested, in order of importance. The probe drops any the
# API does not serve and the dataset build tolerates their absence; what was
# actually obtained is recorded in openmeteo/variables.json for VERDICT.md.
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
PARAMS = {"models": "era5", "timezone": "UTC", "wind_speed_unit": "ms", "start_date": START, "end_date": END}


def _request(sess: requests.Session, lats: list[float], lons: list[float], variables: list[str]) -> list[dict]:
    params = {
        **PARAMS,
        "latitude": ",".join(f"{v:.3f}" for v in lats),
        "longitude": ",".join(f"{v:.3f}" for v in lons),
        "daily": ",".join(variables),
    }
    for attempt in range(30):
        r = sess.get(URL, params=params, timeout=180)
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else [data]
        body = r.text[:300]
        if r.status_code == 429 or "limit" in body.lower():
            # minute, hour or day quota; the message says which
            wait = 70 if "minute" in body.lower() else 3700 if "hour" in body.lower() else None
            if wait is None:
                raise SystemExit(f"daily quota reached, relaunch tomorrow (resumable). Server said: {body}")
            print(f"  rate limited, sleeping {wait} s ({body!r})", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 400:
            raise ValueError(body)
        print(f"  HTTP {r.status_code}, retry {attempt + 1}: {body!r}", flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("Open-Meteo: gave up after 30 attempts")


def probe(sess: requests.Session, lat: float, lon: float) -> list[str]:
    """Return the subset of DAILY_VARS the API serves, dropping the rejected ones one by one."""
    variables = list(DAILY_VARS)
    while variables:
        try:
            _request(sess, [lat], [lon], variables)
            return variables
        except ValueError as exc:
            bad = [v for v in variables if v in str(exc)]
            if not bad:
                raise
            print(f"  not served, dropped: {bad}")
            variables = [v for v in variables if v not in bad]
    raise RuntimeError("Open-Meteo served none of the requested daily variables")


def main() -> None:
    common.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = mesh.load_cells().sort_values("cell").reset_index(drop=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "wildfire-eu-study spike Q1 (research, non-commercial)"

    served_path = OUT_DIR / "variables.json"
    if served_path.exists():
        variables = json.loads(served_path.read_text())["daily"]
    else:
        variables = probe(sess, float(cells.loc[0, "lat"]), float(cells.loc[0, "lon"]))
        served_path.write_text(json.dumps({"daily": variables, "params": PARAMS}, indent=2))
    print(f"daily variables served: {variables}")
    if "--probe" in sys.argv:
        sample = _request(sess, [float(cells.loc[0, "lat"])], [float(cells.loc[0, "lon"])], variables)[0]
        print(json.dumps({k: sample[k] for k in ("latitude", "longitude", "elevation", "daily_units")}, indent=2))
        print({k: v[:3] for k, v in sample["daily"].items()})
        return

    n_batches = (len(cells) + BATCH - 1) // BATCH
    with common.timed("download_weather"):
        for b in range(n_batches):
            target = OUT_DIR / f"batch_{b:04d}.json"
            if common.done(target):
                continue
            chunk = cells.iloc[b * BATCH : (b + 1) * BATCH]
            data = _request(sess, chunk["lat"].tolist(), chunk["lon"].tolist(), variables)
            if len(data) != len(chunk):
                raise RuntimeError(f"batch {b}: {len(data)} answers for {len(chunk)} cells")
            for rec, cell in zip(data, chunk["cell"].tolist(), strict=True):
                rec["cell"] = int(cell)
            target.write_text(json.dumps(data))
            print(f"  batch {b + 1}/{n_batches} ({len(chunk)} cells)", flush=True)
            time.sleep(0.3)
    print(f"done: {n_batches} batches in {OUT_DIR}")


def load_weather() -> pd.DataFrame:
    """Daily weather table from the downloaded batches: cell, date, one column per served variable."""
    frames = []
    for path in sorted(OUT_DIR.glob("batch_*.json")):
        for rec in json.loads(path.read_text()):
            d = rec["daily"]
            df = pd.DataFrame({k: v for k, v in d.items() if k != "time"})
            df.insert(0, "date", pd.to_datetime(d["time"]))
            df.insert(0, "cell", rec["cell"])
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no Open-Meteo batches under {OUT_DIR} (run s02_download_weather.py)")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    main()
