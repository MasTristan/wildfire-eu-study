"""Assemble fact_fireweather_countrymonth from the three climate ingests.

Grain: country x month, per docs/data-model.md. Columns:
    mean_fwi, max_fwi, days_fwi_high, days_fwi_very_high  <- src/ingest/cds_fwi.py
    mean_temp, total_precip, precip_anomaly, mean_rh       <- src/ingest/era5.py
    spei                                                    <- src/ingest/edo_spei.py

The three sources do not share a footprint and are not made to. FWI and ERA5
run on the same 0.5 deg bbox and cover the same countries; EDO's mask is
European and stops short of most of North Africa and the Middle East. So the
join is a LEFT join onto the FWI spine: a country-month with no SPEI keeps a
null SPEI, and the row still carries its fire weather. Nothing is imputed and
no row is dropped for being partially covered.

Coverage per column is printed at every run: an empty column is the exact
failure this repo has been bitten by before, and it must be visible from the
build log rather than discovered on a report page.

Run:
    python src/transform/fireweather.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

_COLUMNS = [
    "country_key", "month_key",
    "mean_fwi", "max_fwi", "days_fwi_high", "days_fwi_very_high",
    "mean_temp", "total_precip", "precip_anomaly", "mean_rh", "spei",
]


def _read_interim(name: str, columns: list[str]) -> pd.DataFrame:
    path = config.DATA_INTERIM / name
    if not path.exists():
        print(f"  NOTE: {path.name} missing, its columns stay null "
              f"(not imputed). Run the matching ingest module.")
        return pd.DataFrame(columns=["country_key", "month_key", *columns])
    df = pd.read_csv(path, parse_dates=["month_key"])
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns {missing}")
    return df[["country_key", "month_key", *columns]]


def build() -> Path:
    fwi = _read_interim(
        "fwi_countrymonth.csv",
        ["mean_fwi", "max_fwi", "days_fwi_high", "days_fwi_very_high"],
    )
    if fwi.empty:
        raise FileNotFoundError(
            "fwi_countrymonth.csv is the spine of this table and is missing: "
            "run src/ingest/cds_fwi.py first"
        )
    era5 = _read_interim(
        "era5_countrymonth.csv",
        ["mean_temp", "total_precip", "precip_anomaly", "mean_rh"],
    )
    spei = _read_interim("edo_spei3_countrymonth.csv", ["spei"])

    out = fwi.merge(era5, on=["country_key", "month_key"], how="left")
    out = out.merge(spei, on=["country_key", "month_key"], how="left")

    for col in _COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series([float("nan")] * len(out), dtype="float64")
    out = out[_COLUMNS].sort_values(["country_key", "month_key"]).reset_index(drop=True)

    _report_coverage(out)
    path = config.DATA_PROCESSED / "fact_fireweather_countrymonth.parquet"
    out.to_parquet(path, index=False)
    span = f"{out['month_key'].min():%Y-%m} to {out['month_key'].max():%Y-%m}"
    print(f"Wrote {path} ({len(out)} country-month rows, {span}, "
          f"{out['country_key'].nunique()} countries)")
    return path


def _report_coverage(df: pd.DataFrame) -> None:
    print("  column coverage (non-null share of rows):")
    for col in _COLUMNS[2:]:
        share = df[col].notna().mean()
        flag = "  <-- EMPTY" if share == 0 else ""
        print(f"    {col:<20} {share:6.1%}{flag}")


if __name__ == "__main__":
    build()
