"""CDS/EWDS ingest: Fire Weather Index (cems-fire-historical-v1), aggregated
to country x month.

Scope decision: 2016-01 through the current month, daily, Europe+Med+MENA
bbox at 0.5deg. 2016 is the EFFIS perimeter start, so the fire-weather series
and the burnt-area series now share a common horizon and can legitimately be
put on the same chart (phase 3, lot 1; before that the floor was 2023 and the
two series were four times apart in length). `cems-fire-historical-v1` itself
goes back to 1940, the floor is ours.

Only `fire_weather_index` is pulled here, and the output is the interim table
data/interim/fwi_countrymonth.csv. Temperature, precipitation and humidity
come from src/ingest/era5.py, SPEI from src/ingest/edo_spei.py, each writing
its own interim table; src/transform/fireweather.py joins the three into
data/processed/fact_fireweather_countrymonth.parquet. No module writes columns
it did not measure.

Danger-class thresholds for days_fwi_high / days_fwi_very_high are EFFIS's
own published classes (Table 2, "Fire Weather Index (FWI) - Dataset
Description", ECMWF Copernicus Knowledge Base:
https://confluence.ecmwf.int/spaces/CKB/pages/283569774/):
    very low <5.2, low 5.2-11.2, moderate 11.2-21.3, high 21.3-38.0,
    very high 38.0-50.0, extreme >=50.0
days_fwi_high counts days where the country's spatial-mean FWI >= 21.3
(High class or above); days_fwi_very_high uses >= 38.0. mean_fwi/max_fwi are
computed over all grid-cell-days within the country for the month (max_fwi
is the single hottest grid cell/day observed, an extreme signal distinct
from the day-count thresholds which use the country-wide daily mean).

Country attribution: geo_ref.grid_mask(), the same point-in-polygon helper
src/ingest/era5.py uses on the same 0.5 deg grid, so the fire-weather and the
climate series attribute identical cells to identical countries. A divergence
there would bias one series against the other on the very chart that compares
them.

Run:
    python src/ingest/cds_fwi.py             # download (slow, EWDS queue) + aggregate
    python src/ingest/cds_fwi.py --agg-only  # skip download, aggregate existing files
    python src/ingest/cds_fwi.py --download-only
    python src/ingest/cds_fwi.py --years 2016 2017   # subset, for reruns

One request per year: the EWDS queue times out on an 11-year submission, and a
per-year file makes the download resumable (an already-present, non-empty year
file is skipped) instead of all-or-nothing over several hours.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.ingest import geo_ref  # noqa: E402

_AREA = [72, -25, 20, 45]  # N, W, S, E
_YEAR_MIN = 2016  # EFFIS perimeter floor: the two series must share a horizon
_YEAR_MAX = 2026
_YEARS = [str(y) for y in range(_YEAR_MIN, _YEAR_MAX + 1)]
_HIGH_THRESHOLD = 21.3
_VERY_HIGH_THRESHOLD = 38.0


def _raw_dir() -> Path:
    out_dir = config.DATA_RAW / "cds_fwi"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def year_path(year: str) -> Path:
    return _raw_dir() / f"fwi_{year}.nc"


def download_year(year: str) -> Path:
    target = year_path(year)
    if target.exists() and target.stat().st_size > 0:
        print(f"  {year}: already downloaded ({target.stat().st_size / 1e6:.1f} MB), skipping")
        return target

    client = config.cds_client("ewds")
    print(f"  {year}: submitting fire_weather_index request, area={_AREA}, grid=0.5/0.5 "
          f"(EWDS queue, can take a long while)...")
    client.retrieve(
        config.CDS_FWI_DATASET,
        {
            "product_type": "reanalysis",
            "variable": "fire_weather_index",
            "dataset_type": "consolidated_dataset",
            "system_version": "4_1",
            "year": [year],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "area": _AREA,
            "grid": "0.5/0.5",
            "data_format": "netcdf",
        },
        str(target),
    )
    print(f"  {year}: downloaded -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def download(years: list[str] | None = None) -> list[Path]:
    years = years or _YEARS
    print(f"CDS/EWDS: fire weather index, {years[0]}-{years[-1]}, one request per year.")
    paths = []
    for year in years:
        try:
            paths.append(download_year(year))
        except Exception as exc:  # noqa: BLE001 - one bad year must not sink the run
            # The last requested year is routinely unavailable: consolidation of
            # cems-fire-historical-v1 lags real time. Keep going and report.
            print(f"  {year}: FAILED ({type(exc).__name__}: {str(exc)[:200]})")
    return paths


def aggregate_file(nc_path: Path) -> pd.DataFrame:
    print(f"Opening {nc_path}...")
    ds = xr.open_dataset(nc_path)
    var = "fwinx" if "fwinx" in ds.data_vars else list(ds.data_vars)[0]
    print(f"  variable: {var}, dims: {ds[var].dims}, shape: {ds[var].shape}")

    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"

    # Same helper as src/ingest/era5.py: the two series are plotted against each
    # other, so they must attribute the same cells to the same countries.
    mask = geo_ref.grid_mask(ds[lon_name].values, ds[lat_name].values)
    print(f"  {len(mask)}/{len(ds[lon_name]) * len(ds[lat_name])} grid cells fall in a tracked country")

    df = ds[[var]].to_dataframe().reset_index()
    df = df.rename(columns={lon_name: "lon", lat_name: "lat", time_dim: "date", var: "fwi"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.merge(mask, on=["lon", "lat"], how="inner")
    df = df.dropna(subset=["fwi"])

    daily_country = df.groupby(["country_key", "date"], as_index=False)["fwi"].mean()
    daily_country["month_key"] = daily_country["date"].values.astype("datetime64[M]")

    monthly = daily_country.groupby(["country_key", "month_key"], as_index=False).agg(
        mean_fwi=("fwi", "mean"),
        days_fwi_high=("fwi", lambda s: int((s >= _HIGH_THRESHOLD).sum())),
        days_fwi_very_high=("fwi", lambda s: int((s >= _VERY_HIGH_THRESHOLD).sum())),
    )
    cell_max = df.groupby(["country_key", df["date"].values.astype("datetime64[M]")])["fwi"].max()
    cell_max.index.set_names(["country_key", "month_key"], inplace=True)
    monthly = monthly.merge(cell_max.rename("max_fwi").reset_index(), on=["country_key", "month_key"])
    ds.close()
    return monthly


def aggregate(nc_paths: list[Path]) -> pd.DataFrame:
    """Aggregate year by year and concatenate. Holding 11 years of daily
    grid-cell rows in one frame is ~60M rows; per-year keeps peak memory at
    one year's worth."""
    frames = [aggregate_file(p) for p in sorted(nc_paths)]
    if not frames:
        raise RuntimeError("no FWI netCDF file to aggregate")
    monthly = pd.concat(frames, ignore_index=True)

    # A country-month can appear twice only if two year files overlap; guard it
    # rather than trust the file naming.
    dupes = monthly.duplicated(subset=["country_key", "month_key"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicated country-month rows across year files")

    return monthly.sort_values(["country_key", "month_key"]).reset_index(drop=True)[[
        "country_key", "month_key", "mean_fwi", "max_fwi",
        "days_fwi_high", "days_fwi_very_high",
    ]]


def build(agg_only: bool = False, download_only: bool = False,
          years: list[str] | None = None) -> Path | None:
    years = years or _YEARS
    if not agg_only:
        download(years)
    if download_only:
        return None

    nc_paths = [p for p in (year_path(y) for y in years) if p.exists() and p.stat().st_size > 0]
    if not nc_paths:
        raise FileNotFoundError(f"no fwi_<year>.nc under {_raw_dir()}")

    monthly = aggregate(nc_paths)
    out = config.DATA_INTERIM / "fwi_countrymonth.csv"
    monthly.to_csv(out, index=False)
    span = f"{monthly['month_key'].min():%Y-%m} to {monthly['month_key'].max():%Y-%m}"
    print(f"Wrote {out} ({len(monthly)} country-month rows, {span})")
    print("  -> run src/transform/fireweather.py to assemble the fact table")
    return out


def _parse_years(argv: list[str]) -> list[str] | None:
    if "--years" not in argv:
        return None
    picked = [a for a in argv[argv.index("--years") + 1:] if not a.startswith("--")]
    return picked or None


if __name__ == "__main__":
    build(
        agg_only="--agg-only" in sys.argv,
        download_only="--download-only" in sys.argv,
        years=_parse_years(sys.argv),
    )
