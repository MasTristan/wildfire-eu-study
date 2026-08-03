"""CDS ingest: ERA5 temperature, precipitation and humidity, country x month.

Dataset: `reanalysis-era5-single-levels-monthly-means`, product type
`monthly_averaged_reanalysis`, on the main Climate Data Store. Note this is a
DIFFERENT store from the EWDS one src/ingest/cds_fwi.py uses; both are reached
with the same ECMWF personal access token through config.cds_client(), which
passes url and key explicitly rather than relying on ~/.cdsapirc (that file
holds a single url/key couple and cannot serve both stores).

Grid: 0.5 degrees, the same resolution and bbox as the FWI pull, and the
country attribution goes through the same geo_ref.grid_mask(). The two series
are put on the same charts, so they must be aggregated over the same cells.

ERA5-Land (0.1 deg) was the alternative and was rejected: over this bbox it is
roughly 25x the volume for a country-mean that the report reads at one decimal
place, and it would no longer match the FWI grid.

Variables, and what each becomes:

    2m_temperature (K)           -> mean_temp, degrees Celsius
    total_precipitation (m/day)  -> total_precip, mm per month
    2m_dewpoint_temperature (K)  -> mean_rh, percent

  total_precipitation in the monthly-means product is the mean DAILY
  accumulation over the month, in metres. Converting to mm per month is
  therefore value * 1000 * days_in_month, not value * 1000.

  mean_rh is derived, not measured: ERA5 publishes dewpoint, and relative
  humidity follows from the Magnus-Tetens saturation vapour pressure ratio
  e(Td)/e(T) with the standard coefficients 17.67 / 243.5 degC. Deriving it
  from monthly-mean temperature and dewpoint gives the humidity of the mean
  state rather than the mean of the daily humidities; the two differ by a few
  points in a month with a strong diurnal cycle. Documented approximation,
  adequate for a country-month descriptive column, not for a physical budget.

precip_anomaly is the deviation from the 1991-2020 monthly normal, in percent
of that normal. 1991-2020 is the current WMO climatological standard normal
period, and it is the same reference EDO uses for SPEI, so the two drought
signals on the Why page are anchored to the same baseline. The normal is
computed from this same ERA5 pull, per country and per calendar month, so it
carries the same grid and the same country masks as the value it corrects.
Where the normal is under _MIN_NORMAL_MM the anomaly is left null: a percent
deviation from an almost-dry normal is a division artefact, not a signal.

Run:
    python src/ingest/era5.py                  # download (chunked) + aggregate
    python src/ingest/era5.py --agg-only
    python src/ingest/era5.py --download-only
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.ingest import geo_ref  # noqa: E402

_AREA = [72, -25, 20, 45]  # N, W, S, E, identical to src/ingest/cds_fwi.py
_NORMAL_START, _NORMAL_END = 1991, 2020  # WMO standard normal period
_YEAR_MAX = 2026
_MIN_NORMAL_MM = 5.0
# One request per decade: resumable, and well inside the CDS per-request cap.
_CHUNKS = [(1991, 2000), (2001, 2010), (2011, 2020), (2021, _YEAR_MAX)]

_VARIABLES = ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"]
_SHORT_NAMES = {"t2m": "t2m", "d2m": "d2m", "tp": "tp"}


def _raw_dir() -> Path:
    out = config.DATA_RAW / "era5"
    out.mkdir(parents=True, exist_ok=True)
    return out


def chunk_path(start: int, end: int) -> Path:
    return _raw_dir() / f"era5_monthly_{start}_{end}.nc"


def download_chunk(start: int, end: int) -> Path:
    target = chunk_path(start, end)
    if target.exists() and target.stat().st_size > 0:
        print(f"  {start}-{end}: already downloaded ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    client = config.cds_client("cds")
    print(f"  {start}-{end}: submitting request (CDS queue)...")
    client.retrieve(
        config.CDS_ERA5_DATASET,
        {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": _VARIABLES,
            "year": [str(y) for y in range(start, end + 1)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"],
            "area": _AREA,
            "grid": ["0.5", "0.5"],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    print(f"  {start}-{end}: downloaded -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def download() -> list[Path]:
    print(f"CDS: ERA5 monthly means, {_CHUNKS[0][0]}-{_CHUNKS[-1][1]}, one request per decade.")
    paths = []
    for start, end in _CHUNKS:
        try:
            paths.append(download_chunk(start, end))
        except Exception as exc:  # noqa: BLE001 - a failed decade must not sink the rest
            print(f"  {start}-{end}: FAILED ({type(exc).__name__}: {str(exc)[:200]})")
    return paths


def _relative_humidity(t_k: pd.Series, td_k: pd.Series) -> pd.Series:
    """Magnus-Tetens: RH = 100 * e(Td) / e(T), coefficients 17.67 / 243.5 degC."""
    def sat(temp_c):
        return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

    rh = 100.0 * sat(td_k - 273.15) / sat(t_k - 273.15)
    return rh.clip(upper=100.0)


def _open_dataset(path: Path) -> xr.Dataset:
    """CDS ignores download_format=unarchived when a request mixes instantaneous
    and accumulated variables: it returns a zip holding one netCDF per step type
    (`avgua` for t2m/d2m, `avgad` for tp). Verified 2026-07-31. Merge them back
    into the single dataset the rest of this module expects.
    """
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path)

    extract_dir = path.with_suffix("")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(extract_dir)
        members = [extract_dir / n for n in zf.namelist() if n.endswith(".nc")]
    if not members:
        raise ValueError(f"{path.name} is a zip with no netCDF member")
    return xr.merge([xr.open_dataset(m) for m in members], compat="override")


def aggregate_file(nc_path: Path, mask: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"  aggregating {nc_path.name} ...")
    ds = _open_dataset(nc_path)
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"

    present = [v for v in _SHORT_NAMES if v in ds.data_vars]
    if len(present) < 3:
        raise ValueError(f"{nc_path.name}: expected t2m/d2m/tp, found {list(ds.data_vars)}")

    if mask is None:
        mask = geo_ref.grid_mask(ds[lon_name].values, ds[lat_name].values)

    df = ds[present].to_dataframe().reset_index()
    df = df.rename(columns={lon_name: "lon", lat_name: "lat", time_dim: "month_key"})
    df = df.merge(mask, on=["lon", "lat"], how="inner")
    df["month_key"] = pd.to_datetime(df["month_key"]).values.astype("datetime64[M]")

    df["mean_temp"] = df["t2m"] - 273.15
    df["mean_rh"] = _relative_humidity(df["t2m"], df["d2m"])
    days = pd.to_datetime(df["month_key"]).dt.days_in_month
    df["total_precip"] = df["tp"] * 1000.0 * days  # m/day -> mm/month

    monthly = df.groupby(["country_key", "month_key"], as_index=False).agg(
        mean_temp=("mean_temp", "mean"),
        total_precip=("total_precip", "mean"),
        mean_rh=("mean_rh", "mean"),
    )
    ds.close()
    return monthly, mask


def add_precip_anomaly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Percent deviation of total_precip from its 1991-2020 monthly normal."""
    df = monthly.copy()
    df["month"] = df["month_key"].dt.month
    ref = df[df["month_key"].dt.year.between(_NORMAL_START, _NORMAL_END)]
    normal = ref.groupby(["country_key", "month"], as_index=False)["total_precip"].mean()
    normal = normal.rename(columns={"total_precip": "precip_normal"})

    n_years = ref["month_key"].dt.year.nunique()
    if n_years < _NORMAL_END - _NORMAL_START + 1:
        print(f"  WARNING: normal computed on {n_years} years, not 30. "
              f"precip_anomaly is against an incomplete baseline.")

    df = df.merge(normal, on=["country_key", "month"], how="left")
    usable = df["precip_normal"] >= _MIN_NORMAL_MM
    df["precip_anomaly"] = np.where(
        usable, (df["total_precip"] - df["precip_normal"]) / df["precip_normal"] * 100.0, np.nan
    )
    dropped = int((~usable & df["precip_normal"].notna()).sum())
    if dropped:
        print(f"  {dropped} country-months left with a null precip_anomaly: their "
              f"1991-2020 normal is under {_MIN_NORMAL_MM} mm (arid climates, "
              f"percent deviation would be a division artefact)")
    return df.drop(columns=["month", "precip_normal"])


def build(agg_only: bool = False, download_only: bool = False) -> Path | None:
    if not agg_only:
        download()
    if download_only:
        return None

    paths = [p for p in (chunk_path(s, e) for s, e in _CHUNKS)
             if p.exists() and p.stat().st_size > 0]
    if not paths:
        raise FileNotFoundError(f"no era5_monthly_*.nc under {_raw_dir()}")

    frames: list[pd.DataFrame] = []
    mask: pd.DataFrame | None = None
    for path in sorted(paths):
        monthly, mask = aggregate_file(path, mask)
        frames.append(monthly)

    allm = pd.concat(frames, ignore_index=True)
    dupes = allm.duplicated(subset=["country_key", "month_key"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicated country-month rows across ERA5 chunks")

    allm = add_precip_anomaly(allm).sort_values(["country_key", "month_key"])
    out = config.DATA_INTERIM / "era5_countrymonth.csv"
    allm.to_csv(out, index=False)
    span = f"{allm['month_key'].min():%Y-%m} to {allm['month_key'].max():%Y-%m}"
    print(f"Wrote {out} ({len(allm)} country-month rows, {span}, "
          f"{allm['country_key'].nunique()} countries)")
    print("  -> run src/transform/fireweather.py to assemble the fact table")
    return out


if __name__ == "__main__":
    build(agg_only="--agg-only" in sys.argv, download_only="--download-only" in sys.argv)
