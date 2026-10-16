"""Silver: one clean, typed table per source, written to the silver folder
(data/processed during the migration, see src/lake.py).

Each asset calls the v1 transform function unchanged and declares the bronze
tables it reads, so the lineage graph is real even before the functions are
rewritten. Every silver table is validated against its pandera contract on
write: a table that violates its contract never reaches the lake.
"""
from __future__ import annotations

import pandas as pd
from dagster import AssetsDefinition, MaterializeResult, asset

from src import contracts, lake
from src.orchestration.assets import bronze

GROUP = "silver"
PREFIX = "silver"


def _write(name: str, df: pd.DataFrame) -> MaterializeResult:
    lake.ensure_dirs()
    df = contracts.validate(name, df)
    path = lake.SILVER / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return MaterializeResult(metadata={"rows": int(len(df)), "columns": list(df.columns), "path": str(path)})


def _dimension(name: str, builder_name: str, deps: list) -> AssetsDefinition:
    @asset(name=name, key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, deps=deps, description=f"{name}, conformed dimension")
    def _asset() -> MaterializeResult:
        from src.transform import dimensions

        builder = getattr(dimensions, builder_name)
        return _write(name, builder())

    return _asset


dim_date = _dimension("dim_date", "build_dim_date", [])
dim_country = _dimension("dim_country", "build_dim_country", [])
dim_commune = _dimension("dim_commune", "build_dim_commune", [bronze.bdiff_incendies])
dim_landcover = _dimension("dim_landcover", "build_dim_landcover", [])
dim_firecause = _dimension("dim_firecause", "build_dim_firecause", [bronze.bdiff_incendies])
dim_source = _dimension("dim_source", "build_dim_source", [])


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    deps=[bronze.effis_perimeters, bronze.bdiff_incendies, dim_commune],
    description="fact_fireevent, one row per fire: EFFIS perimeters and BDIFF declarations",
)
def fact_fireevent() -> MaterializeResult:
    from src.transform import fact_fireevent as mod

    effis = mod.build_effis_events()
    bdiff = mod.build_bdiff_events()
    fact = pd.concat([effis, bdiff], ignore_index=True)
    fact["firedate_key"] = pd.to_datetime(fact["firedate_key"]).dt.normalize()
    return _write("fact_fireevent", fact)


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    deps=[bronze.fwi_countrymonth, bronze.era5_countrymonth, bronze.edo_spei3_countrymonth],
    description="fact_fireweather_countrymonth: FWI, ERA5 and SPEI joined at country x month, never imputed",
)
def fact_fireweather_countrymonth() -> MaterializeResult:
    from src.transform import fireweather

    # The v1 function writes to data/processed itself (= silver during the migration).
    path = fireweather.build()
    df = pd.read_parquet(path)
    return _write("fact_fireweather_countrymonth", df)


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    deps=[bronze.firms_hotspots_raw, dim_country],
    description="fact_hotspot_countrymonth: VIIRS detections counted at country x month over every raw window on disk",
)
def fact_hotspot_countrymonth() -> MaterializeResult:
    from src.ingest import firms

    return _write("fact_hotspot_countrymonth", firms.aggregate_raw())
