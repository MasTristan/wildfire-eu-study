"""Step 1b: ERA5 daily t2m, d2m, u10, v10 (daily mean) and tp (daily sum), 0.25 deg.

Dataset choice: `derived-era5-single-levels-daily-statistics` (CDS) rather than raw
hourly `reanalysis-era5-single-levels`. It is the same ERA5 reanalysis, the daily
aggregation is done server side in UTC, and it avoids moving 24x the volume for
five variables. One request per (statistic, year, month, area), resumable (a full
year exceeds the CDS cost limit).

    uv run python notebooks/spike_q1/s02_download_era5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

from src import config  # noqa: E402

DATASET = "derived-era5-single-levels-daily-statistics"
GROUPS = {
    "mean": (
        "daily_mean",
        ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
    ),
    "sum": ("daily_sum", ["total_precipitation"]),
}


def main() -> None:
    common.ensure_dirs()
    client = config.cds_client("cds")
    for year in common.YEARS:
        for month in range(1, 13):
            for area_name, area in common.AREAS.items():
                for tag, (stat, variables) in GROUPS.items():
                    target = common.RAW / f"era5_{tag}_{area_name}_{year}_{month:02d}.nc"
                    if common.done(target):
                        continue
                    print(f"request {target.name} ...", flush=True)
                    with common.timed(f"download_era5_{tag}_{area_name}_{year}_{month:02d}"):
                        client.retrieve(
                            DATASET,
                            {
                                "product_type": "reanalysis",
                                "variable": variables,
                                "year": str(year),
                                "month": f"{month:02d}",
                                "day": [f"{d:02d}" for d in range(1, 32)],
                                "daily_statistic": stat,
                                "time_zone": "utc+00:00",
                                "frequency": "1_hourly",
                                "area": area,
                                "grid": "0.25/0.25",
                                "data_format": "netcdf",
                            },
                            str(target),
                        )


if __name__ == "__main__":
    main()
