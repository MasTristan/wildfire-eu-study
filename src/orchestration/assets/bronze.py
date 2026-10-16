"""Bronze: fetch each source and leave it as received (data/raw) plus the
one per-source interim table the v1 ingest module derives from it.

Every asset here is a thin wrapper around the v1 `src/ingest/<source>.build`
entry point, called unchanged. The v1 modules are resumable file by file, so
re-materialising an asset re-downloads nothing that is already on disk.

Partitioning is honest about what the v1 code can do today: FIRMS and the FWI
download accept a year, the others rebuild their whole span in one call
(cheap after the first run, because of the resumable downloads). Phase 1 of
PLAN.md rewrites them as partition-safe assets; the asset keys stay.
"""
from datetime import date

from dagster import AssetExecutionContext, MaterializeResult, StaticPartitionsDefinition, asset

from src import config
from src.orchestration.partitions import yearly

era5_chunks = StaticPartitionsDefinition(["1991-2000", "2001-2010", "2011-2020", "2021-2026"])

GROUP = "bronze"
PREFIX = "bronze"


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, description="EFFIS rda-stats, burnt area by macro-region (control figure)")
def effis_rda_stats() -> MaterializeResult:
    from src.ingest import effis

    years = range(config.BASE_YEAR_MIN, date.today().year + 1)
    out = effis.build(years)
    return MaterializeResult(metadata={"path": str(out), "years": f"{years.start}-{years.stop - 1}"})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, description="EFFIS burnt-area perimeters (WFS), attributes and GeoJSON pages")
def effis_perimeters() -> MaterializeResult:
    from src.ingest import effis_perimeters as mod

    n = mod.build(page_size=5000, max_pages=None, save_geometry=True)
    return MaterializeResult(metadata={"features": n})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    partitions_def=yearly,
    description="BDIFF export for one year, raw zip",
)
def bdiff_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import bdiff

    path = bdiff.fetch_year(int(context.partition_key))
    return MaterializeResult(metadata={"path": str(path), "bytes": path.stat().st_size})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, deps=[bdiff_raw],
       description="BDIFF, French fires at commune grain with causes, parsed from every yearly zip on disk")
def bdiff_incendies() -> MaterializeResult:
    from src.ingest import bdiff

    rows = bdiff.rows_from_raw()
    out = bdiff.write_interim(rows)
    return MaterializeResult(metadata={"path": str(out), "rows": len(rows)})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    partitions_def=yearly,
    description="NASA FIRMS VIIRS S-NPP hotspots, raw CSV windows for one year",
)
def firms_hotspots_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import firms

    year = int(context.partition_key)
    points = firms.fetch_year_to_date(year)
    return MaterializeResult(metadata={"year": year, "detections": int(len(points))})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, description="GWIS/GFAS wildfire emissions, country x month")
def gwis_gfas_emissions() -> MaterializeResult:
    from src.ingest import gwis

    out = gwis.build()
    return MaterializeResult(metadata={"path": str(out)})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    partitions_def=yearly,
    description="CEMS EWDS fire weather indices (ERA5-based), one NetCDF per year",
)
def cds_fwi_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import cds_fwi

    cds_fwi.build(download_only=True, years=[context.partition_key])
    return MaterializeResult(metadata={"year": context.partition_key})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    partitions_def=yearly,
    description="The seven FWI-system components at 0.25 degree for one year over Europe (EWDS), the feature store's spine",
)
def cds_fwi_full_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import cds_fwi

    path = cds_fwi.download_year_full(context.partition_key)
    return MaterializeResult(metadata={"path": str(path), "bytes": path.stat().st_size})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    deps=[cds_fwi_raw],
    description="FWI aggregated to country x month from every downloaded year",
)
def fwi_countrymonth() -> MaterializeResult:
    from src.ingest import cds_fwi

    out = cds_fwi.build(agg_only=True)
    return MaterializeResult(metadata={"path": str(out)})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    partitions_def=era5_chunks,
    description="ERA5 monthly means, one NetCDF per decade (CDS)",
)
def era5_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import era5

    start, end = (int(v) for v in context.partition_key.split("-"))
    path = era5.download_chunk(start, end)
    return MaterializeResult(metadata={"path": str(path)})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, deps=[era5_raw],
       description="ERA5 monthly means aggregated to country x month, with precipitation anomaly")
def era5_countrymonth() -> MaterializeResult:
    from src.ingest import era5

    out = era5.build(agg_only=True)
    return MaterializeResult(metadata={"path": str(out)})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    partitions_def=yearly,
    description="EDO SPEI-3 dekadal grid, one NetCDF per year",
)
def edo_spei_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import edo_spei

    path = edo_spei.download_year(context.partition_key, edo_spei._remote_names())
    return MaterializeResult(metadata={"path": str(path) if path else "not published by EDO"})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, deps=[edo_spei_raw],
       description="EDO SPEI-3 drought index aggregated to country x month, never imputed")
def edo_spei3_countrymonth() -> MaterializeResult:
    from src.ingest import edo_spei

    out = edo_spei.build()  # downloads are cached file by file, so this only aggregates
    return MaterializeResult(metadata={"path": str(out)})
