"""Step 1c: static surface description for the mesh (vegetation cover and type, orography).

CORINE needs a Copernicus Land account, ESA CCI land cover needs a licence the
account owner must accept on the CDS website, and the brief's fallback (EFFIS
perimeter cover fractions) would use post-J fire information. The spike therefore
takes the static ERA5 surface fields from `reanalysis-era5-single-levels` (same
licence as the other ERA5 data): high and low vegetation cover and type, land-sea
mask, geopotential. Static, leakage-free, 0.25 deg. Documented as a CORINE
substitute in VERDICT.md.

    uv run python notebooks/spike_q1/s03_download_landcover.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

from src import config  # noqa: E402

DATASET = "reanalysis-era5-single-levels"


def main() -> None:
    common.ensure_dirs()
    client = config.cds_client("cds")
    for name, area in common.AREAS.items():
        target = common.RAW / f"static_{name}.nc"
        if common.done(target):
            print(f"skip {target.name}")
            continue
        print(f"request {target.name} ...", flush=True)
        with common.timed(f"download_landcover_{name}"):
            client.retrieve(
                DATASET,
                {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "high_vegetation_cover", "low_vegetation_cover", "type_of_high_vegetation",
                        "type_of_low_vegetation", "land_sea_mask", "geopotential",
                    ],
                    "year": ["2018"], "month": ["07"], "day": ["01"], "time": ["00:00"],
                    "area": area, "grid": "0.25/0.25", "data_format": "netcdf",
                },
                str(target),
            )


if __name__ == "__main__":
    main()
