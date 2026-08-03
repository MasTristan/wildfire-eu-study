"""FIRMS ingest: VIIRS active-fire hotspots, aggregated to country x month.

Scope decision (documented, not silently assumed): hotspots are ingested for
the CURRENT CALENDAR YEAR only (year-to-date), matching the "saison en cours
/ live" page this fact serves (docs/dashboard-spec.md P1). Full multi-year
hotspot history is explicitly phase 2 in CLAUDE.md's own roadmap and would
multiply the request count for a facet nobody views historically (hotspots
are never compared to burnt-area hectares anyway, see data model guardrail).
Sensor scope: VIIRS_SNPP only (both its NRT and SP/science-quality feeds,
picked per date from the real /api/data_availability boundary). NOAA-20/21
VIIRS are additional feeds not included here.

Country attribution uses real point-in-polygon against Natural Earth's public
domain 1:50m admin-0 country boundaries (not a bounding-box approximation),
joined on ISO_A2 to our country_key.

API constraints verified live 2026-07-30:
  - max day_range per request = 5 (the docs elsewhere say 10; this MAP_KEY's
    tier rejects >5 with "Invalid day range. Expects [1..5].")
  - VIIRS_SNPP_NRT covers a rolling recent window; VIIRS_SNPP_SP covers the
    rest of the history up to a few months ago. The exact boundary is read
    from /api/data_availability/csv at run time, not hardcoded.

Run:
    python src/ingest/firms.py                # year-to-date, VIIRS_SNPP
"""
from __future__ import annotations

import io
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.ingest import geo_ref  # noqa: E402

_TIMEOUT = 60
_MAX_DAY_RANGE = 5
# Europe + Mediterranean + Near East + North Africa: a superset bbox covering
# every country in country_reference.csv, verified against their Natural
# Earth extents.
_BBOX = (-25, 20, 45, 72)  # west, south, east, north
# Country polygons (download, caching and the ISO_A2 sentinel fix) live in
# src/ingest/geo_ref.py, shared with cds_fwi and gwis.


def _data_availability(source: str) -> tuple[date, date]:
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{config.FIRMS_MAP_KEY}/{source}"
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    row = next(csv_row for csv_row in resp.text.splitlines()[1:2])
    _, min_d, max_d = row.split(",")
    return date.fromisoformat(min_d), date.fromisoformat(max_d)


def _fetch_window(source: str, end_date: date, day_range: int) -> pd.DataFrame:
    w, s, e, n = _BBOX
    url = (
        f"{config.FIRMS_AREA_CSV_URL}/{config.FIRMS_MAP_KEY}/{source}/"
        f"{w},{s},{e},{n}/{day_range}/{end_date.isoformat()}"
    )
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    if resp.text.startswith("Invalid") or resp.text.startswith("Error"):
        raise RuntimeError(f"FIRMS API error for {source} {end_date}: {resp.text[:200]}")
    return pd.read_csv(io.StringIO(resp.text))


def fetch_year_to_date(year: int) -> pd.DataFrame:
    nrt_min, nrt_max = _data_availability("VIIRS_SNPP_NRT")
    sp_min, sp_max = _data_availability("VIIRS_SNPP_SP")
    print(f"  VIIRS_SNPP_NRT available {nrt_min}..{nrt_max}; SP available {sp_min}..{sp_max}")

    start = date(year, 1, 1)
    today = date.today()
    raw_dir = config.DATA_RAW / "firms"
    raw_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    cursor = start
    while cursor <= today:
        window_end = min(cursor + timedelta(days=_MAX_DAY_RANGE - 1), today)
        source = "VIIRS_SNPP_SP" if window_end <= sp_max else "VIIRS_SNPP_NRT"
        day_range = (window_end - cursor).days + 1
        try:
            df = _fetch_window(source, window_end, day_range)
        except RuntimeError as exc:
            print(f"    {cursor}..{window_end} ({source}): SKIPPED ({exc})")
            cursor = window_end + timedelta(days=1)
            continue
        df["source_key"] = "FIRMS_VIIRS"
        parts.append(df)
        raw_path = raw_dir / f"viirs_{cursor.isoformat()}_{window_end.isoformat()}.csv"
        raw_path.write_text(df.to_csv(index=False), encoding="utf-8")
        print(f"    {cursor}..{window_end} ({source}): {len(df)} detections")
        cursor = window_end + timedelta(days=1)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def assign_country(points: pd.DataFrame) -> pd.DataFrame:
    countries = geo_ref.load_country_polygons()
    geom = [Point(xy) for xy in zip(points["longitude"], points["latitude"])]
    gpts = gpd.GeoDataFrame(points, geometry=geom, crs="EPSG:4326")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        joined = gpd.sjoin(gpts, countries, how="left", predicate="within")
    return joined.drop(columns=["geometry", "index_right"], errors="ignore")


def build(year: int | None = None) -> Path:
    year = year or date.today().year
    print(f"FIRMS: fetching VIIRS_SNPP hotspots for {year} year-to-date...")
    points = fetch_year_to_date(year)
    if points.empty:
        print("  no detections returned")
        out_df = pd.DataFrame(columns=[
            "country_key", "month_key", "source_key", "hotspot_count", "mean_frp", "max_frp",
        ])
    else:
        print(f"  {len(points)} raw detections, assigning country via point-in-polygon...")
        points = assign_country(points)
        unmatched = points["country_key"].isna().sum()
        if unmatched:
            print(f"  NOTE: {unmatched}/{len(points)} detections fall outside our tracked "
                  f"countries (open sea, non-EFFIS territory) and are dropped")
        points = points.dropna(subset=["country_key"])
        points["month_key"] = pd.to_datetime(points["acq_date"]).values.astype("datetime64[M]")
        out_df = points.groupby(["country_key", "month_key", "source_key"], as_index=False).agg(
            hotspot_count=("latitude", "count"),
            mean_frp=("frp", "mean"),
            max_frp=("frp", "max"),
        )

    out = config.DATA_PROCESSED / "fact_hotspot_countrymonth.parquet"
    out_df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(out_df)} country-month rows)")
    return out


if __name__ == "__main__":
    build()
