"""GWIS ingest: wildfire emissions by country and month (GFAS product).

Source: Global Wildfire Information System country-profile bulk download,
`emission_gfas_full_<start>_<end>.zip` on the EFFIS/GWIS public S3 bucket.
Open data, no account, no key. GWIS is JRC + Copernicus, same family as EFFIS.

Which of the three GWIS emission products, and why GFAS:
    GFAS (Copernicus CAMS Global Fire Assimilation System) converts satellite
    fire radiative power directly into emitted mass. GFED and FAO are the two
    alternatives offered; both are burnt-area-driven, so pairing them with our
    EFFIS burnt area would restate the same measurement twice under two names.
    GFAS is the only one of the three that is an independent observation of the
    fire, which is exactly what dim_source calls a control source.

Units: tonnes of the pollutant, per country and month. Verified against the
France 2022 season (4.35 Mt CO2 over ~72 kha burnt, i.e. ~60 t CO2/ha, inside
the 30-100 t/ha range published for Mediterranean forest fuel loads).

Expect GFAS totals to sit ABOVE any figure derived from EFFIS perimeters: GFAS
detects every vegetation fire the satellite sees, including agricultural
burning and fires below the EFFIS perimeter-mapping size threshold. The two are
not reconcilable and are never summed; see docs/sources.md.

Rows are national only (the `gid_1` region column is empty throughout this
product), so no subnational de-duplication is needed, but it is asserted rather
than assumed.

Domain restriction, and why it is not optional: GFAS ships pre-aggregated
NATIONAL totals, which cannot be clipped afterwards. For a country whose
territory mostly lies outside the study domain the national total is not a
European figure at all. Russia is the clear case: 935 Mt CO2 in 2021, almost
entirely Siberian, against ~1 000 ha of EFFIS-mapped Russian burnt area over
the whole series. Left in, it would outweigh every European country combined
on any emissions visual.

So each country's land area is intersected with the study bbox (the same
[72N, 25W, 20S, 45E] box the FWI and ERA5 grids use) from the cached Natural
Earth geometries, and only countries at or above `_MIN_DOMAIN_COVERAGE` are
kept. The excluded list is printed at every run, never hardcoded.

Threshold placement is not a judgement call here: measured 2026-07-31, the
coverage distribution over the 49 tracked countries is bimodal with a single
outlier. Russia 12%, then Georgia 81%, Norway 84% (Svalbard), France 86%
(overseas departments), and 45 countries above 97%. Any threshold between 0.2
and 0.8 selects exactly the same set, so 0.5 is used and the numbers are
printed so a future shift in the distribution is visible rather than silent.
France and Norway are kept deliberately: their out-of-domain territory is
small AND low fire activity relative to the metropolitan seasons that drive
their series (the French monthly profile peaks in July, with the metropolitan
summer, not in the Guiana dry season).

Run:
    python src/ingest/gwis.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.ingest import geo_ref  # noqa: E402
from src.transform.dimensions import REFERENCE_DIR  # noqa: E402

GWIS_BUCKET = "https://effis-gwis-cms.s3.eu-west-1.amazonaws.com/apps/country.profile"
GWIS_EMISSION_FILE = "emission_gfas_full_2003_2024.zip"
_TIMEOUT = 300
# The five pollutants kept out of the sixteen shipped: CO2 is the headline
# figure the data model asks for, the rest are the ones a fire-emissions
# discussion actually uses. Everything else stays in the raw zip.
_POLLUTANTS = {"CO2": "co2_t", "CH4": "ch4_t", "PM25": "pm25_t"}
# Same bbox as src/ingest/cds_fwi.py and src/ingest/era5.py: N, W, S, E.
_STUDY_AREA = [72, -25, 20, 45]
_MIN_DOMAIN_COVERAGE = 0.5


def countries_in_domain() -> tuple[set[str], dict[str, float]]:
    """ISO2 keys whose land area is at least _MIN_DOMAIN_COVERAGE inside the
    study bbox, plus the measured coverage of every tracked country.

    Areas are computed in EPSG:6933 (equal-area) so the ratio is a real area
    ratio and not a degree-square artefact that would flatter high latitudes.
    """
    gdf = geo_ref.load_country_polygons(dissolve=True).set_index("country_key")

    north, west, south, east = _STUDY_AREA
    clip = gpd.GeoSeries([box(west, south, east, north)], crs="EPSG:4326")
    equal_area = "EPSG:6933"
    full = gdf.geometry.to_crs(equal_area).area
    inside = gdf.geometry.intersection(clip.union_all()).to_crs(equal_area).area
    coverage = (inside / full).to_dict()
    keep = {iso for iso, frac in coverage.items() if frac >= _MIN_DOMAIN_COVERAGE}
    return keep, coverage


def download() -> Path:
    out_dir = config.DATA_RAW / "gwis"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / GWIS_EMISSION_FILE
    if target.exists() and target.stat().st_size > 0:
        print(f"  cached: {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    url = f"{GWIS_BUCKET}/{GWIS_EMISSION_FILE}"
    print(f"  downloading {url} ...")
    resp = requests.get(url, timeout=_TIMEOUT)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    print(f"  wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def normalise(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        df = pd.read_csv(io.BytesIO(zf.read(name)), sep=";")
    print(f"  {len(df)} raw rows, {df['year'].min()}-{df['year'].max()}")

    if df["gid_1"].notna().any():
        raise ValueError(
            "GWIS emissions file now carries subnational (gid_1) rows; summing "
            "it as-is would double count. Filter to gid_1.isna() first."
        )

    ref = pd.read_csv(REFERENCE_DIR / "country_reference.csv")
    iso3_to_key = dict(zip(ref["iso3"], ref["country_key"], strict=False))
    df = df[df["gid_0"].isin(iso3_to_key)].copy()
    df["country_key"] = df["gid_0"].map(iso3_to_key)

    missing = sorted(set(ref["iso3"]) - set(df["gid_0"]))
    if missing:
        print(f"  NOTE: no GWIS emissions rows for {missing} (left absent, never zero-filled)")

    keep, coverage = countries_in_domain()
    partial = sorted((iso, f) for iso, f in coverage.items() if f < 0.99)
    print("  domain coverage below 99%: "
          + ", ".join(f"{iso} {f:.0%}" for iso, f in sorted(partial, key=lambda t: t[1])))
    dropped = sorted(set(df["country_key"]) - keep)
    if dropped:
        detail = ", ".join(f"{iso} {coverage.get(iso, float('nan')):.0%}" for iso in dropped)
        print(f"  domain filter: dropping {len(dropped)} country/countries whose land area is "
              f"under {_MIN_DOMAIN_COVERAGE:.0%} inside the study bbox ({detail}). "
              f"Their national GFAS total is not a European figure.")
    df = df[df["country_key"].isin(keep)]

    df["month_key"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    out = df.rename(columns=_POLLUTANTS)[
        ["country_key", "month_key", *_POLLUTANTS.values()]
    ]
    return out.sort_values(["country_key", "month_key"]).reset_index(drop=True)


def build() -> Path:
    print("GWIS: wildfire emissions (GFAS), country x month")
    monthly = normalise(download())
    out = config.DATA_INTERIM / "gwis_gfas_emissions.csv"
    monthly.to_csv(out, index=False)
    span = f"{monthly['month_key'].min():%Y-%m} to {monthly['month_key'].max():%Y-%m}"
    print(f"Wrote {out} ({len(monthly)} country-month rows, {span}, "
          f"{monthly['country_key'].nunique()} countries)")
    return out


if __name__ == "__main__":
    build()
