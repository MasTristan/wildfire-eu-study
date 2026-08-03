"""Build fact_fireevent (grain: individual fire), the primary granular fact.

Unions two sources at their natural grain, per docs/data-model.md guardrail 2
(France/BDIFF stays separate in definition, never reconciled into EFFIS
totals as a single number):

- EFFIS perimeters (data/raw/effis_perimeters/*.geojson + the flat attribute
  CSV already extracted to data/interim): centroid computed from the polygon,
  dominant land-cover taken as the argmax of the CORINE-aligned percent-area
  columns.
- BDIFF France (data/interim/bdiff_incendies.csv): commune-level centroid
  (joined from dim_commune), cause key, no land-cover (BDIFF's own land-use
  split does not align with EFFIS/CORINE classes, so it is left null rather
  than force-mapped).

Run:
    python src/transform/fact_fireevent.py
"""
from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

# The largest raw pages (up to ~107 MB) exceed GDAL's default per-object size
# guard for the GeoJSON driver; must be set before any GDAL/pyogrio call.
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.transform.dimensions import REFERENCE_DIR, cause_key_from_label  # noqa: E402

_LANDCOVER_COLS = {
    "BROADLEA": "broadleaf",
    "CONIFER": "conifer",
    "MIXED": "mixed",
    "SCLEROPH": "sclerophyllous",
    "TRANSIT": "transitional",
    "OTHERNATLC": "other_natural",
    "AGRIAREAS": "agricultural",
    "ARTIFSURF": "artificial",
    "OTHERLC": "other",
}


def _load_effis_centroids() -> pd.DataFrame:
    pages = sorted((config.DATA_RAW / "effis_perimeters").glob("page_*.geojson"))
    if not pages:
        raise FileNotFoundError("no EFFIS perimeter pages: run src/ingest/effis_perimeters.py --full")
    parts = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
        for p in pages:
            gdf = gpd.read_file(p, columns=["id"])
            c = gdf.geometry.centroid
            parts.append(pd.DataFrame({
                "id": gdf["id"].astype(str),
                "centroid_lon": c.x.values,
                "centroid_lat": c.y.values,
            }))
            print(f"  {p.name}: {len(gdf)} geometries")
    # Same WFS pagination-overlap issue as the attribute CSV (non-unique
    # FIREDATE sort key repeats page-boundary features across pages).
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset="id")


def _dominant_landcover(row: pd.Series) -> str | None:
    vals = {key: row[col] for col, key in _LANDCOVER_COLS.items()}
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if not vals or max(vals.values()) <= 0:
        return None
    return max(vals, key=vals.get)


def build_effis_events() -> pd.DataFrame:
    print("EFFIS: loading attributes...")
    attrs = pd.read_csv(config.DATA_INTERIM / "effis_perimeters_attrs.csv", dtype={"id": str})
    # WFS pagination sorts by FIREDATE only, which is not unique: features
    # sharing the exact page-boundary timestamp can be emitted on both pages.
    # Confirmed exact full-row duplicates (not a same-id-different-content
    # conflict), safe to drop.
    before = len(attrs)
    attrs = attrs.drop_duplicates(subset="id")
    if len(attrs) < before:
        print(f"  NOTE: dropped {before - len(attrs)} duplicate rows from WFS pagination overlap")

    print("EFFIS: computing centroids from geometry...")
    centroids = _load_effis_centroids()
    df = attrs.merge(centroids, on="id", how="left")
    missing_centroid = df["centroid_lat"].isna().sum()
    if missing_centroid:
        print(f"  NOTE: {missing_centroid} rows without a matched geometry (dropped)")
        df = df.dropna(subset=["centroid_lat"])

    ref = pd.read_csv(REFERENCE_DIR / "country_reference.csv")
    code_map = dict(zip(ref["effis_raw_code"], ref["country_key"]))
    df["country_key"] = df["COUNTRY"].map(code_map)
    unmapped = df["country_key"].isna().sum()
    if unmapped:
        print(f"  NOTE: dropping {unmapped} rows with unrecognised/blank COUNTRY code")
        df = df.dropna(subset=["country_key"])

    df["landcover_dominant_key"] = df.apply(_dominant_landcover, axis=1)
    # FIREDATE mixes whole-second and fractional-second timestamps.
    df["firedate_key"] = pd.to_datetime(df["FIREDATE"], format="ISO8601").dt.normalize()

    out = pd.DataFrame({
        "fire_event_key": "EFFIS_" + df["id"],
        "source_key": "EFFIS_RDA",
        "country_key": df["country_key"],
        "commune_key": None,
        "firedate_key": df["firedate_key"],
        "cause_key": None,
        "landcover_dominant_key": df["landcover_dominant_key"],
        "area_ha": pd.to_numeric(df["AREA_HA"], errors="coerce"),
        "ba_class": df["CLASS"],
        "pct_natura2000": pd.to_numeric(df["PERCNA2K"], errors="coerce"),
        "centroid_lat": df["centroid_lat"],
        "centroid_lon": df["centroid_lon"],
    })
    print(f"  {len(out)} EFFIS fire events")
    return out


def build_bdiff_events() -> pd.DataFrame:
    print("BDIFF: loading fire records...")
    bdiff_csv = config.DATA_INTERIM / "bdiff_incendies.csv"
    with bdiff_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    df = pd.DataFrame(rows)

    communes = pd.read_parquet(config.DATA_PROCESSED / "dim_commune.parquet")
    commune_centroid = communes.set_index("commune_key")[["centroid_lat", "centroid_lon"]]

    df["commune_key"] = df["commune_insee"].where(df["commune_insee"].isin(communes["commune_key"]))
    unmatched = df["commune_key"].isna().sum()
    if unmatched:
        print(f"  NOTE: {unmatched}/{len(df)} records have a commune_insee not in dim_commune "
              f"(kept, centroid left null: pre-merger/overseas/renamed communes)")

    joined = df.join(commune_centroid, on="commune_key")
    firedate = pd.to_datetime(df["alert_datetime"], format="ISO8601", errors="coerce").dt.normalize()
    area_ha = pd.to_numeric(df["area_total_m2"], errors="coerce") / 10_000

    out = pd.DataFrame({
        "fire_event_key": "BDIFF_" + df["year"] + "_" + df["fire_number"],
        "source_key": "BDIFF",
        "country_key": "FR",
        "commune_key": df["commune_key"],
        "firedate_key": firedate,
        "cause_key": df["cause_raw"].map(cause_key_from_label),
        "landcover_dominant_key": None,
        "area_ha": area_ha,
        "ba_class": None,
        "pct_natura2000": None,
        "centroid_lat": joined["centroid_lat"],
        "centroid_lon": joined["centroid_lon"],
    })
    print(f"  {len(out)} BDIFF fire events")
    return out


def build() -> Path:
    effis = build_effis_events()
    bdiff = build_bdiff_events()
    fact = pd.concat([effis, bdiff], ignore_index=True)
    fact["firedate_key"] = pd.to_datetime(fact["firedate_key"]).dt.normalize()

    out = config.DATA_PROCESSED / "fact_fireevent.parquet"
    fact.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(fact)} rows: {len(effis)} EFFIS + {len(bdiff)} BDIFF)")
    return out


if __name__ == "__main__":
    build()
