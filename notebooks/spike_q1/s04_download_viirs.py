"""Step 1d: VIIRS S-NPP hotspots 2019-2023 (science quality, SP) for the three countries.

Raw material for the label variant (b). FIRMS area API, max day_range = 5 for this
key (see CLAUDE.md pitfalls); availability is read from /api/data_availability.
Resumable per (area, 5-day window).

    uv run python notebooks/spike_q1/s04_download_viirs.py
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

from src import config  # noqa: E402
from src.ingest import firms  # noqa: E402

SOURCE = "VIIRS_SNPP_SP"


def _bbox(area: list[float]) -> str:
    n, w, s, e = area
    return f"{w},{s},{e},{n}"


def main() -> None:
    common.ensure_dirs()
    vdir = common.RAW / "viirs"
    vdir.mkdir(exist_ok=True)
    sp_min, sp_max = firms._data_availability(SOURCE)
    print(f"{SOURCE} available {sp_min}..{sp_max}")
    start, end = date(common.YEARS[0], 1, 1), date(common.YEARS[-1], 12, 31)
    assert sp_min <= start and end <= sp_max, "study window not covered by SP"
    sess = requests.Session()
    with common.timed("download_viirs"):
        for name, area in common.AREAS.items():
            cur = start
            while cur <= end:
                win_end = min(cur + timedelta(days=4), end)
                days = (win_end - cur).days + 1
                target = vdir / f"{name}_{cur.isoformat()}.csv"
                if not common.done(target):
                    # the API takes (start_date, day_range), not an end date
                    url = (
                        f"{config.FIRMS_AREA_CSV_URL}/{config.FIRMS_MAP_KEY}/{SOURCE}/"
                        f"{_bbox(area)}/{days}/{cur.isoformat()}"
                    )
                    for attempt in range(5):
                        r = sess.get(url, timeout=120)
                        if r.status_code == 200 and r.text.startswith("latitude"):
                            target.write_text(r.text)
                            break
                        print(f"  retry {name} {cur} ({r.status_code}: {r.text[:80]!r})")
                        time.sleep(10 * (attempt + 1))
                    else:
                        raise RuntimeError(f"FIRMS failed for {name} {cur}")
                cur = win_end + timedelta(days=1)
            print(f"{name} done", flush=True)


if __name__ == "__main__":
    main()
