"""EDO ingest: SPEI-3 drought index, aggregated to country x month.

Source: Copernicus European Drought Observatory, product `sem03` (3-month
Standardized Precipitation-Evapotranspiration Index, MARSMet input), published
as one netCDF per year on the open EDO data server. No account, no key.

Why SPEI-3 rather than SPEI-1, -6 or -12: the index has to be commensurate
with a fire season. SPEI-1 tracks a single month's weather, which the FWI
already measures better; SPEI-12 tracks multi-year hydrology, which a summer
does not respond to. SPEI-3 in July summarises the spring-to-midsummer water
balance, the accumulation that sets fuel dryness.

Reading the value: SPEI is a standardized anomaly, so it is unitless and
centred on zero. Negative is drier than the 1991-2020 reference distribution,
positive is wetter; EDO's own classification calls below -1 moderate drought,
below -1.5 severe, below -2 extreme.

Grid and aggregation: the product is a 10 km ETRS89-LAEA (EPSG:3035) raster on
dekads (the 1st, 11th and 21st of each month, 36 steps per year). Country
polygons are reprojected to EPSG:3035 once and grid-cell centres are matched
against them, the same point-in-polygon logic used for FWI and hotspots. The
monthly value is the mean over the country's cells and over that month's
dekads. Non-finite cells (the product uses +/-inf outside its mask, with no
_FillValue attribute) are dropped, never zero-filled.

A country with no cell inside the EDO mask simply gets no row: North African
and Middle Eastern countries fall largely outside the European domain, and
their SPEI stays null in the fact table rather than being invented.

Run:
    python src/ingest/edo_spei.py                # 2016-2026
    python src/ingest/edo_spei.py --years 2016 2017
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.ingest import geo_ref  # noqa: E402

EDO_BASE = (
    "https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/"
    "EDO_MARSMet_Standardized_Precipitation_Evapotranspiration_Index_SPEI3/ver1-1-0"
)
_VAR = "sem03"
_GRID_CRS = "EPSG:3035"
_YEAR_MIN = 2016  # aligned with fact_fireweather_countrymonth / EFFIS perimeters
_YEAR_MAX = 2026
_YEARS = [str(y) for y in range(_YEAR_MIN, _YEAR_MAX + 1)]
# SPEI is a standardized normal deviate; |value| > 5 is not a drought reading,
# it is the product's out-of-mask filler leaking through as a finite number.
_PLAUSIBLE_ABS_MAX = 5.0
_TIMEOUT = 600


def _raw_dir() -> Path:
    out = config.DATA_RAW / "edo"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _remote_names() -> dict[str, str]:
    """Map year -> remote filename, read from the directory index.

    The date span is baked into each filename (and the last year's file is
    truncated at the latest published dekad), so it is listed rather than
    guessed.
    """
    import re

    resp = requests.get(f"{EDO_BASE}/", timeout=120)
    resp.raise_for_status()
    names = re.findall(rf'href="({_VAR}_m_edo_(\d{{4}})\d{{4}}_\d{{8}}_t\.nc)"', resp.text)
    return {year: name for name, year in names}


def download_year(year: str, remote: dict[str, str]) -> Path | None:
    if year not in remote:
        print(f"  {year}: not published by EDO, skipped (no imputation)")
        return None
    target = _raw_dir() / remote[year]
    if target.exists() and target.stat().st_size > 0:
        print(f"  {year}: cached ({target.stat().st_size / 1e6:.1f} MB)")
        return target
    print(f"  {year}: downloading {remote[year]} ...")
    resp = requests.get(f"{EDO_BASE}/{remote[year]}", timeout=_TIMEOUT)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    print(f"    {target.stat().st_size / 1e6:.1f} MB")
    return target


def _country_grid_mask(xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
    """(x, y) -> country_key lookup for the EDO grid, computed once."""
    countries = geo_ref.load_country_polygons(dissolve=True).to_crs(_GRID_CRS)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = gpd.GeoDataFrame(
        {"lon": x_grid.ravel(), "lat": y_grid.ravel()},
        geometry=gpd.points_from_xy(x_grid.ravel(), y_grid.ravel()),
        crs=_GRID_CRS,
    )
    joined = gpd.sjoin(points, countries, how="inner", predicate="within")
    return joined[["lon", "lat", "country_key"]]


def aggregate_file(nc_path: Path, mask: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"  aggregating {nc_path.name} ...")
    ds = xr.open_dataset(nc_path)
    da = ds[_VAR]
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)

    if mask is None:
        mask = _country_grid_mask(ds["lon"].values, ds["lat"].values)

    df = da.to_dataframe(name="spei").reset_index()
    df = df.rename(columns={"lon": "lon", "lat": "lat", "time": "date"})
    df = df.merge(mask, on=["lon", "lat"], how="inner")
    df = df[np.isfinite(df["spei"]) & (df["spei"].abs() <= _PLAUSIBLE_ABS_MAX)]

    df["month_key"] = pd.to_datetime(df["date"]).values.astype("datetime64[M]")
    monthly = df.groupby(["country_key", "month_key"], as_index=False)["spei"].mean()
    ds.close()
    return monthly, mask


def build(years: list[str] | None = None) -> Path:
    years = years or _YEARS
    print(f"EDO: SPEI-3 drought index, {years[0]}-{years[-1]}")
    remote = _remote_names()

    frames: list[pd.DataFrame] = []
    mask: pd.DataFrame | None = None
    for year in years:
        path = download_year(year, remote)
        if path is None:
            continue
        monthly, mask = aggregate_file(path, mask)
        frames.append(monthly)

    if not frames:
        raise RuntimeError("EDO published no file for the requested years")

    out_df = pd.concat(frames, ignore_index=True)
    dupes = out_df.duplicated(subset=["country_key", "month_key"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicated country-month SPEI rows")
    out_df = out_df.sort_values(["country_key", "month_key"]).reset_index(drop=True)

    out = config.DATA_INTERIM / "edo_spei3_countrymonth.csv"
    out_df.to_csv(out, index=False)
    span = f"{out_df['month_key'].min():%Y-%m} to {out_df['month_key'].max():%Y-%m}"
    print(f"Wrote {out} ({len(out_df)} country-month rows, {span}, "
          f"{out_df['country_key'].nunique()} countries in the EDO mask)")
    return out


def _parse_years(argv: list[str]) -> list[str] | None:
    if "--years" not in argv:
        return None
    picked = [a for a in argv[argv.index("--years") + 1:] if not a.startswith("--")]
    return picked or None


if __name__ == "__main__":
    build(years=_parse_years(sys.argv))
