"""Step 1a: all seven FWI-system components, daily, native 0.25 deg, per year and area.

The v1 raw files only hold `fwinx` at 0.5 deg, so the spike re-downloads
cems-fire-historical-v1 with every component. One request per (year, area),
resumable.

    uv run python notebooks/spike_q1/s01_download_fwi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

from src import config  # noqa: E402


def main() -> None:
    common.ensure_dirs()
    client = config.cds_client("ewds")
    for year in common.YEARS:
        for name, area in common.AREAS.items():
            target = common.RAW / f"fwi_{name}_{year}.zip"
            if common.done(target):
                print(f"skip {target.name}")
                continue
            print(f"request {target.name} ...", flush=True)
            with common.timed(f"download_fwi_{name}_{year}"):
                client.retrieve(
                    config.CDS_FWI_DATASET,
                    {
                        "product_type": "reanalysis",
                        "variable": list(common.FWI_VARS),
                        "dataset_type": "consolidated_dataset",
                        "system_version": "4_1",
                        "year": [str(year)],
                        "month": [f"{m:02d}" for m in range(1, 13)],
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "area": area,
                        "grid": "0.25/0.25",
                        "data_format": "netcdf",
                    },
                    str(target),
                )


if __name__ == "__main__":
    main()
