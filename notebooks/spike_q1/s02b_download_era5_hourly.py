"""Step 1b (alternative): hourly ERA5 reduced to daily locally, one request per month and area.

The `derived-era5-single-levels-daily-statistics` queue took 3 to 7 hours per monthly
request (timings in data/interim/spike_q1/timings) and a whole year of hourly data
exceeds the CDS cost limit. The hourly `reanalysis-era5-single-levels` dataset answers
a monthly request in minutes; the UTC daily mean / sum is computed here and written
under the same names as s02 (era5_mean_*, era5_sum_*), so s05 reads either source.

    uv run python notebooks/spike_q1/s02b_download_era5_hourly.py [year ...]
"""

from __future__ import annotations

import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import xarray as xr  # noqa: E402

from src import config  # noqa: E402

VARS = ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind",
        "10m_v_component_of_wind", "total_precipitation"]


def to_daily(src: Path, dst_mean: Path, dst_sum: Path) -> None:
    if zipfile.is_zipfile(src):  # instant and accumulated variables arrive as two files
        out = src.with_suffix('')
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
        ds = xr.merge([xr.open_dataset(f) for f in sorted(out.glob('*.nc'))], compat='override')
    else:
        ds = xr.open_dataset(src)
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.variables])
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    ds[["t2m", "d2m", "u10", "v10"]].resample(time="1D").mean().to_netcdf(dst_mean)
    # tp is accumulated over the hour PRECEDING its stamp: UTC day D = stamps D 01h..D+1 00h,
    # hence shift(-1). The last hour of the last day of the month is taken from the next
    # month's stamp 00h, which is not in this file (one hour of 744, negligible).
    ds["tp"].shift(time=-1).resample(time="1D").sum().to_dataset(name="tp").to_netcdf(dst_sum)


def main() -> None:
    common.ensure_dirs()
    client = config.cds_client("cds")
    years = [int(a) for a in sys.argv[1:]] or common.YEARS
    for year in years:
        for month in range(1, 13):
            for name, area in common.AREAS.items():
                stem = f"{name}_{year}_{month:02d}"
                mean, tot = common.RAW / f"era5_mean_{stem}.nc", common.RAW / f"era5_sum_{stem}.nc"
                if common.done(mean) and common.done(tot):
                    continue
                hourly = common.RAW / f"era5h_{stem}.nc"
                if not common.done(hourly):
                    print(f"request {hourly.name} ...", flush=True)
                    with common.timed(f"download_era5h_{stem}"):
                        for _ in range(60):
                            try:
                                client.retrieve(
                                    "reanalysis-era5-single-levels",
                                    {
                                        "product_type": ["reanalysis"], "variable": VARS,
                                        "year": [str(year)], "month": [f"{month:02d}"],
                                        "day": [f"{d:02d}" for d in range(1, 32)],
                                        "time": [f"{h:02d}:00" for h in range(24)],
                                        "area": area, "grid": "0.25/0.25", "data_format": "netcdf",
                                    },
                                    str(hourly),
                                )
                                break
                            except Exception as exc:
                                if any(k in str(exc) for k in ("limited", "rejected", "queued")):
                                    time.sleep(120)
                                    continue
                                raise
                to_daily(hourly, mean, tot)
                hourly.unlink()
                shutil.rmtree(hourly.with_suffix(''), ignore_errors=True)
                print(f"daily {stem} written", flush=True)


if __name__ == "__main__":
    main()
