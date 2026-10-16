"""Step 1b: hourly ERA5 reduced to UTC daily values locally, one CDS request per
month and area (amendment 2 of PROTOCOL.md).

History: the derived daily-statistics dataset queued for hours per request,
Open-Meteo's free tier stopped after five batches a day. The hourly
`reanalysis-era5-single-levels` dataset answers a monthly request in minutes.
The reduction is shared with the production ingest (src/ingest/era5_daily.py)
so the spike and the feature store compute exactly the same daily fields:
t2m_c, t2m_max, rh, rh_min, wind (daily max), wind_mean, tp_mm.

    uv run python notebooks/spike_q1/s02b_download_era5_hourly.py [year ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

from src import config  # noqa: E402
from src.ingest import era5_daily  # noqa: E402


def main() -> None:
    common.ensure_dirs()
    client = config.cds_client("cds")
    years = [int(a) for a in sys.argv[1:]] or common.YEARS
    for year in years:
        for month in range(1, 13):
            for name, area in common.AREAS.items():
                target = era5_daily.daily_path(common.RAW, name, year, month)
                if common.done(target):
                    continue
                with common.timed(f"download_era5h_{name}_{year}_{month:02d}"):
                    era5_daily.download_month(year, month, area, target, client=client)
                print(f"daily {target.name} written", flush=True)


if __name__ == "__main__":
    main()
