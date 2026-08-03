"""EFFIS ingest: aggregate Rapid Damage Assessment burnt-area statistics.

Pulls the verified /rda-stats endpoint for a range of years and writes a tidy
CSV to data/processed. This is the proof-of-life pipeline for the project; it
uses only the standard library plus requests so it runs before the full env is
installed.

Run:
    python src/ingest/effis.py
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

# Allow running as a plain script (python src/ingest/effis.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

_TIMEOUT = 30
_FIELDS = ["year", "area", "sum_per_forest_pct", "est_sum_total_ha", "sum_rda_total_ha"]


def parse_rda_payload(year: int, payload: dict) -> list[dict]:
    """Flatten the /rda-stats JSON payload into one tidy row per macro-region."""
    rows = []
    for region, rec in payload.items():
        rows.append(
            {
                "year": year,
                "area": rec.get("area", region),
                "sum_per_forest_pct": rec.get("sumPerForest%"),
                "est_sum_total_ha": rec.get("estSumTotal"),
                "sum_rda_total_ha": rec.get("sumRDATotal"),
            }
        )
    return rows


def fetch_rda_stats(year: int) -> list[dict]:
    """Return one row per macro-region (EU, EU_non, ME_AF, total) for `year`."""
    url = f"{config.EFFIS_RDA_STATS_URL}?year={year}"
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_rda_payload(year, payload)


def build(years: range) -> Path:
    all_rows: list[dict] = []
    for y in years:
        try:
            all_rows.extend(fetch_rda_stats(y))
            print(f"  {y}: ok ({len(all_rows)} rows total)")
        except Exception as exc:  # noqa: BLE001
            print(f"  {y}: FAILED ({exc})", file=sys.stderr)

    out = config.DATA_PROCESSED / "effis_rda_stats.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return out


if __name__ == "__main__":
    years = range(config.BASE_YEAR_MIN, config.BASE_YEAR_MAX + 1)
    print(f"EFFIS rda-stats: {years.start}-{years.stop - 1}")
    path = build(years)
    print(f"Wrote {path}")
