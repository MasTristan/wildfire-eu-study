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

from dagster import AssetExecutionContext, MaterializeResult, asset

from src import config
from src.orchestration.partitions import yearly

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


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, description="BDIFF, French fires at commune grain with causes")
def bdiff_incendies() -> MaterializeResult:
    from src.ingest import bdiff

    out = bdiff.build(config.BASE_YEAR_MIN, date.today().year)
    return MaterializeResult(metadata={"path": str(out)})


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
    deps=[cds_fwi_raw],
    description="FWI aggregated to country x month from every downloaded year",
)
def fwi_countrymonth() -> MaterializeResult:
    from src.ingest import cds_fwi

    out = cds_fwi.build(agg_only=True)
    return MaterializeResult(metadata={"path": str(out)})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, description="ERA5 monthly means aggregated to country x month, with precipitation anomaly")
def era5_countrymonth() -> MaterializeResult:
    from src.ingest import era5

    out = era5.build()
    return MaterializeResult(metadata={"path": str(out)})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, description="EDO SPEI-3 drought index aggregated to country x month")
def edo_spei3_countrymonth() -> MaterializeResult:
    from src.ingest import edo_spei

    out = edo_spei.build()
    return MaterializeResult(metadata={"path": str(out)})
