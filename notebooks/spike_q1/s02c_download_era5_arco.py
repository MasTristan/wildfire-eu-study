"""Step 1b (fallback): daily ERA5 from the ARCO-ERA5 public store, for the
months the CDS has not delivered (PROTOCOL.md amendment 3).

Writes the same files as s02b (era5_daily_<area>_<year>_<month>.nc), so s05
reads either source; months already on disk are skipped.

    uv run python notebooks/spike_q1/s02c_download_era5_arco.py --check
        verify the store layout, print its time coverage
    uv run python notebooks/spike_q1/s02c_download_era5_arco.py --verify 2019 07 iberia
        rebuild one month already obtained from the CDS into a temporary file and
        compare: exit code 0 when every variable is within tolerance, 1 otherwise
    uv run python notebooks/spike_q1/s02c_download_era5_arco.py [--workers 8]
        fill every missing month, both areas at once
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

from src.ingest import era5_arco, era5_daily  # noqa: E402


def _targets(year: int, month: int) -> dict[str, Path]:
    return {name: era5_daily.daily_path(common.RAW, name, year, month) for name in common.AREAS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", nargs=3, metavar=("YEAR", "MONTH", "AREA"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    store = era5_arco.Store()
    info = store.check_store()
    print(f"ARCO store layout ok, final data to {info['valid_time_stop']}, updated {info['last_updated']}")
    if args.check:
        return 0
    if args.verify:
        year, month, area = int(args.verify[0]), int(args.verify[1]), args.verify[2]
        ref = era5_daily.daily_path(common.RAW, area, year, month)
        if not ref.exists():
            print(f"no CDS reference file {ref.name}: pick a month already downloaded by s02b")
            return 1
        tmp = common.INTERIM / f"arco_verify_{area}_{year}_{month:02d}.nc"
        tmp.unlink(missing_ok=True)
        with common.timed(f"arco_verify_{area}_{year}_{month:02d}"):
            era5_arco.download_month(store, year, month, {area: common.AREAS[area]}, {area: tmp}, args.workers)
        report = era5_arco.compare_daily(ref, tmp)
        print(report.to_string(index=False))
        ok = bool(report["passes"].all())
        print("VERIFY PASS" if ok else "VERIFY FAIL")
        return 0 if ok else 1
    missing = [(y, m) for y in common.YEARS for m in range(1, 13)
               if any(not common.done(p) for p in _targets(y, m).values())]
    print(f"{len(missing)} months missing in at least one area")
    for i, (y, m) in enumerate(missing, 1):
        with common.timed(f"arco_{y}_{m:02d}"):
            era5_arco.download_month(store, y, m, common.AREAS, _targets(y, m), args.workers)
        print(f"  {i}/{len(missing)} {y}-{m:02d} written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
