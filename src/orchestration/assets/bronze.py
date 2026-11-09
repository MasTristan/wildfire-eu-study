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

from dagster import (
    AssetExecutionContext,
    BackfillPolicy,
    MaterializeResult,
    StaticPartitionsDefinition,
    asset,
)

from src import config
from src.orchestration.partitions import yearly

era5_chunks = StaticPartitionsDefinition(["1991-2000", "2001-2010", "2011-2020", "2021-2026"])

# Every partitioned asset runs a partition range as ONE run (no daemon needed:
# `dagster asset materialize --partition-range` refuses multi-run backfills
# without one) and loops over context.partition_keys. The functions it calls
# skip files already on disk, so a range rerun downloads only what is missing.
SINGLE_RUN = BackfillPolicy.single_run()

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
    backfill_policy=SINGLE_RUN,
    description="BDIFF export, one raw zip per year",
)
def bdiff_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import bdiff

    paths = [bdiff.fetch_year(int(k)) for k in context.partition_keys]
    return MaterializeResult(metadata={"years": ",".join(context.partition_keys),
                                       "bytes": sum(p.stat().st_size for p in paths)})


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
    backfill_policy=SINGLE_RUN,
    description="NASA FIRMS VIIRS S-NPP hotspots, raw CSV windows per year (refetched on every run)",
)
def firms_hotspots_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import firms

    detections = sum(int(len(firms.fetch_year_to_date(int(k)))) for k in context.partition_keys)
    return MaterializeResult(metadata={"years": ",".join(context.partition_keys), "detections": detections})


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
    backfill_policy=SINGLE_RUN,
    description="CEMS EWDS fire weather indices (ERA5-based), one NetCDF per year",
)
def cds_fwi_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import cds_fwi

    # download() skips years already on disk and reports, without raising, a
    # year the EWDS has not consolidated yet
    cds_fwi.build(download_only=True, years=list(context.partition_keys))
    return MaterializeResult(metadata={"years": ",".join(context.partition_keys)})


@asset(
    key_prefix=PREFIX,
    group_name=GROUP,
    kinds={"python"},
    partitions_def=yearly,
    backfill_policy=SINGLE_RUN,
    description="The seven FWI-system components at 0.25 degree, one file per year over Europe (EWDS), the feature store's spine",
)
def cds_fwi_full_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import cds_fwi

    paths = [cds_fwi.download_year_full(k) for k in context.partition_keys]
    return MaterializeResult(metadata={"years": ",".join(context.partition_keys),
                                       "bytes": sum(p.stat().st_size for p in paths)})


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
    backfill_policy=SINGLE_RUN,
    description="ERA5 monthly means, one NetCDF per decade (CDS)",
)
def era5_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import era5

    paths = []
    for key in context.partition_keys:
        start, end = (int(v) for v in key.split("-"))
        paths.append(era5.download_chunk(start, end))
    return MaterializeResult(metadata={"decades": ",".join(context.partition_keys),
                                       "paths": ", ".join(p.name for p in paths)})


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
    backfill_policy=SINGLE_RUN,
    description="EDO SPEI-3 dekadal grid, one NetCDF per year",
)
def edo_spei_raw(context: AssetExecutionContext) -> MaterializeResult:
    from src.ingest import edo_spei

    remote = edo_spei._RemoteIndex()  # read only if a year is missing from disk
    found = [k for k in context.partition_keys if edo_spei.download_year(k, remote) is not None]
    missing = sorted(set(context.partition_keys) - set(found))
    return MaterializeResult(metadata={"years": ",".join(found), "not_published": ",".join(missing) or "none"})


@asset(key_prefix=PREFIX, group_name=GROUP, kinds={"python"}, deps=[edo_spei_raw],
       description="EDO SPEI-3 drought index aggregated to country x month, never imputed")
def edo_spei3_countrymonth() -> MaterializeResult:
    from src.ingest import edo_spei

    out = edo_spei.build()  # downloads are cached file by file, so this only aggregates
    return MaterializeResult(metadata={"path": str(out)})
