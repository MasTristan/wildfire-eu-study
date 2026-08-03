"""Shared country geometry reference (Natural Earth admin-0, 1:50m).

Three modules attribute a point or a grid cell to a country (firms, cds_fwi,
gwis). They used to each read the cached GeoJSON on their own; they now go
through here, because the ISO code in that file needs a fix that must not be
applied in one module and forgotten in another.

The fix, measured 2026-07-31: Natural Earth sets `ISO_A2` to the sentinel
"-99" for France, Norway and Kosovo (a long-standing quirk of the admin-0
dataset, where sovereignty-disputed or overseas-territory-carrying entries are
left uncoded). Filtering on `ISO_A2` therefore silently dropped France and
Norway from every point-in-polygon join: France, which has its own deep-dive
page in the report, was absent from fact_fireweather_countrymonth and
fact_hotspot_countrymonth entirely. `ISO_A2_EH` carries the correct code in
those rows, so the effective code is ISO_A2_EH first, ISO_A2 as fallback.

Note this means the France geometry includes the overseas departments. Callers
that aggregate a national total over an area-restricted product must clip or
exclude accordingly (see src/ingest/gwis.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.transform.dimensions import REFERENCE_DIR  # noqa: E402

NE_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_50m_admin_0_countries.geojson"
)
_SENTINEL = "-99"


def cache_path() -> Path:
    return config.DATA_RAW / "reference" / "ne_countries.geojson"


def ensure_cached() -> Path:
    cache = cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print("  downloading Natural Earth admin-0 countries (1:50m)...")
        resp = requests.get(NE_COUNTRIES_URL, timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    return cache


def load_country_polygons(dissolve: bool = False) -> gpd.GeoDataFrame:
    """Tracked countries only, with a reliable `country_key` (ISO 3166-1 alpha-2).

    Set dissolve=True to get exactly one row per country (multipart geometry).
    """
    gdf = gpd.read_file(ensure_cached(), columns=["ISO_A2", "ISO_A2_EH"])
    code = gdf["ISO_A2_EH"].where(
        gdf["ISO_A2_EH"].notna() & (gdf["ISO_A2_EH"] != _SENTINEL), gdf["ISO_A2"]
    )
    gdf["country_key"] = code

    ref = pd.read_csv(REFERENCE_DIR / "country_reference.csv")
    tracked = set(ref["country_key"])
    gdf = gdf[gdf["country_key"].isin(tracked)][["country_key", "geometry"]]

    found = set(gdf["country_key"])
    missing = sorted(tracked - found)
    if missing:
        print(f"  NOTE: no Natural Earth polygon for {missing} "
              f"(they cannot be attributed by point-in-polygon)")

    if dissolve:
        gdf = gdf.dissolve(by="country_key").reset_index()
    return gdf.reset_index(drop=True)


def grid_mask(lons, lats, crs: str = "EPSG:4326") -> pd.DataFrame:
    """(lon, lat) -> country_key lookup for a regular grid, one row per cell
    centre that falls inside a tracked country.

    Shared by the FWI and ERA5 aggregations so both attribute the exact same
    cells to the same countries; a divergence there would silently bias one
    series against the other on the very chart that compares them.
    """
    import numpy as np

    countries = load_country_polygons(dissolve=True)
    if crs != "EPSG:4326":
        countries = countries.to_crs(crs)

    lon_grid, lat_grid = np.meshgrid(np.asarray(lons), np.asarray(lats))
    points = gpd.GeoDataFrame(
        {"lon": lon_grid.ravel(), "lat": lat_grid.ravel()},
        geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
        crs=crs,
    )
    joined = gpd.sjoin(points, countries, how="inner", predicate="within")
    return joined[["lon", "lat", "country_key"]].reset_index(drop=True)
