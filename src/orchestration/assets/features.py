"""Feature store assets (gold/features): the common grid, daily weather from
Open-Meteo, static surface fields, and the cell x day table one year per
partition. Builders live in src/features/daily.py; this module only wires
inputs and outputs.
"""

from datetime import date
from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, MaterializeResult, asset

from src import config, grid, lake
from src.orchestration.assets import bronze, silver
from src.orchestration.partitions import yearly

CELLS_PATH = lake.SILVER / "grid_cells.parquet"
OPENMETEO_DIR = config.DATA_RAW / "openmeteo"
STATIC_PATH = config.DATA_RAW / "era5_static" / "static_europe.nc"
FEATURES_START = f"{config.BASE_YEAR_MIN + 10}-01-01"  # 2016: the EFFIS perimeter floor


def _fwi_year_path(year: str) -> Path:
    from src.ingest import cds_fwi

    return cds_fwi.year_path(year)


@asset(key_prefix="silver", group_name="silver", kinds={"python"}, deps=[bronze.cds_fwi_raw],
       description="The common 0.25 degree mesh: FWI lattice cells whose centre is in a tracked country (ADR 0003)")
def grid_cells() -> MaterializeResult:
    from src.features import daily

    first = next(p for p in (_fwi_year_path(y) for y in yearly.get_partition_keys()) if p.exists())
    ds = daily.open_any(first)
    cells = grid.mesh_from_lattice(ds["longitude"].values, ds["latitude"].values)
    ds.close()
    lake.ensure_dirs()
    cells.to_parquet(CELLS_PATH, index=False)
    return MaterializeResult(metadata={"cells": int(len(cells)), "countries": int(cells["country_key"].nunique())})


@asset(key_prefix="bronze", group_name="bronze", kinds={"python"}, deps=[grid_cells],
       description="Daily ERA5 weather per cell from the Open-Meteo archive, resumable by batch (ADR 0006)")
def openmeteo_daily() -> MaterializeResult:
    from src.ingest import openmeteo

    cells = pd.read_parquet(CELLS_PATH)
    end = (date.today() - pd.Timedelta(days=7)).isoformat()  # the archive lags a few days
    written = openmeteo.download(cells, OPENMETEO_DIR, FEATURES_START, end)
    return MaterializeResult(metadata={"batches_written": written, "cells": int(len(cells))})


@asset(key_prefix="bronze", group_name="bronze", kinds={"python"},
       description="Static ERA5 surface fields for Europe: vegetation cover and type, land-sea mask, geopotential")
def era5_static() -> MaterializeResult:
    STATIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not (STATIC_PATH.exists() and STATIC_PATH.stat().st_size > 0):
        client = config.cds_client("cds")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": ["high_vegetation_cover", "low_vegetation_cover", "type_of_high_vegetation",
                             "type_of_low_vegetation", "land_sea_mask", "geopotential"],
                "year": ["2018"], "month": ["07"], "day": ["01"], "time": ["00:00"],
                "area": [72, -25, 34, 45], "grid": "0.25/0.25", "data_format": "netcdf",
            },
            str(STATIC_PATH),
        )
    return MaterializeResult(metadata={"path": str(STATIC_PATH)})


@asset(key_prefix="gold", group_name="gold", kinds={"python"}, partitions_def=yearly,
       deps=[grid_cells, bronze.cds_fwi_raw, openmeteo_daily, era5_static, bronze.edo_spei3_countrymonth, silver.fact_fireevent],
       description="Feature store at cell x day for one year: FWI components, weather, SPEI, static fields, labels and lags")
def features_daily(context: AssetExecutionContext) -> MaterializeResult:
    from src.features import daily
    from src.ingest import openmeteo

    year = int(context.partition_key)
    cells = pd.read_parquet(CELLS_PATH)
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    fwi = daily.fwi_daily([_fwi_year_path(str(year))], cells)
    weather = openmeteo.load(OPENMETEO_DIR, cells)
    # the previous December is needed for the 7-day rain window of early January
    weather = weather[(weather["date"] >= dates[0] - pd.Timedelta(days=10)) & (weather["date"] <= dates[-1])]
    edo_paths = sorted((config.DATA_RAW / "edo").glob("sem03_m_edo_*.nc"))
    spei = daily.spei_daily(edo_paths, cells, dates) if edo_paths else None
    static = daily.static_fields([STATIC_PATH], cells) if STATIC_PATH.exists() else None
    events = pd.read_parquet(lake.SILVER / "fact_fireevent.parquet")
    events = events[events["source_key"] == "EFFIS_RDA"]
    labels = daily.ignition_labels(events, cells)
    lags = daily.lag_counts(labels, cells, dates, pd.Timestamp(FEATURES_START), tag="a")
    df = daily.assemble(cells, dates, fwi, weather, spei, static, labels, lags)
    out = lake.GOLD_FEATURES / f"daily_{year}.parquet"
    lake.ensure_dirs()
    df.to_parquet(out, index=False)
    return MaterializeResult(metadata={"rows": int(len(df)), "positives": int(df["y_a"].sum()), "path": str(out)})
