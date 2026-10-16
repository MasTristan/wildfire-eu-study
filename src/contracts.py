"""Gold-layer data contracts: one pandera schema per parquet that Power BI reads.

These schemas are the executable version of docs/data-model.md and of the
TMDL column definitions in powerbi/WildfireEU.SemanticModel. They encode
three things, in this order of importance:

1. the column set and the dtype each column is read with (what the M
   partitions expect, so a schema change here is a schema change for the
   report);
2. the grain, as a uniqueness constraint on the key columns (one row per
   natural grain, never two sources merged in one record);
3. the business invariants the data-layer tests already assert one by one
   (surfaces never negative, SPEI never imputed, GWIS rows never carrying a
   burnt area), so they fail at the point of writing rather than at test time.

`GOLD_TABLES` is also the reference used by the v1 -> v2 parity test: the
key columns listed here are the sort order the comparison relies on.

Nothing here touches the data: schemas describe, `validate_processed`
reports. The transform scripts opt in by calling `validate(name, df)` before
writing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column

from src import config

_SOURCE_KEYS = {
    "EFFIS_STATS", "EFFIS_RDA", "BDIFF", "FIRMS_VIIRS", "GWIS_GFAS", "CDS_FWI", "ERA5", "EDO_SPEI",
}
_LANDCOVER_KEYS = {
    "broadleaf", "conifer", "mixed", "sclerophyllous", "transitional",
    "other_natural", "agricultural", "artificial", "other",
}
_CAUSE_GROUPS = {"natural", "accidental", "malicious", "involuntary", "unknown"}
_MACRO_REGIONS = {"EU", "EU_non", "ME_AF"}

_DT = "datetime64[ns]"


def _first_of_month(s: pd.Series) -> pd.Series:
    return s.dt.day == 1


def _first_of_year(s: pd.Series) -> pd.Series:
    return (s.dt.day == 1) & (s.dt.month == 1)


def _string(nullable: bool = False, **kw) -> Column:
    # Parquet strings come back as object dtype under pandas 2.2 defaults.
    return Column(object, nullable=nullable, **kw)


@dataclass(frozen=True)
class GoldTable:
    name: str
    keys: tuple[str, ...]
    schema: pa.DataFrameSchema
    # Invariants spanning several rows or columns, checked by name below.
    notes: tuple[str, ...] = field(default_factory=tuple)


def _no_source_mixing(df: pd.DataFrame) -> bool:
    """GWIS rows carry emissions only; every other source carries no emissions."""
    gwis = df["source_key"] == "GWIS_GFAS"
    return bool(
        df.loc[gwis, "burnt_area_ha"].isna().all()
        and df.loc[gwis, "number_of_fires"].isna().all()
        and df.loc[~gwis, "co2_emissions_t"].isna().all()
    )


def _spei_not_imputed(s: pd.Series) -> bool:
    spei = s.dropna()
    return spei.empty or (spei == 0).mean() < 0.01


GOLD_TABLES: dict[str, GoldTable] = {
    t.name: t
    for t in (
        GoldTable(
            "dim_date",
            ("date_key",),
            pa.DataFrameSchema(
                {
                    "date_key": Column(_DT, unique=True, coerce=True),
                    "year": Column(int, Check.in_range(2000, 2100), coerce=True),
                    "quarter": Column(int, Check.in_range(1, 4), coerce=True),
                    "month": Column(int, Check.in_range(1, 12), coerce=True),
                    "month_name": _string(),
                    "week_iso": Column(int, Check.in_range(1, 53), coerce=True),
                    "day_of_year": Column(int, Check.in_range(1, 366), coerce=True),
                    "is_fire_season": Column(bool),
                    "season_label": _string(),
                },
                checks=Check(
                    lambda df: set(df.loc[df["is_fire_season"], "month"]) == {6, 7, 8, 9},
                    error="fire season is June to September",
                ),
                strict=False,
            ),
        ),
        GoldTable(
            "dim_country",
            ("country_key",),
            pa.DataFrameSchema(
                {
                    "country_key": _string(unique=True),
                    "effis_raw_code": _string(nullable=True),
                    "iso2": _string(nullable=True),
                    "iso3": _string(nullable=True),
                    "name_en": _string(),
                    "name_fr": _string(nullable=True),
                    "macro_region": _string(checks=Check.isin(_MACRO_REGIONS)),
                    "is_eu": Column(bool),
                    "is_mediterranean": Column(bool),
                    "land_area_km2": Column(float, Check.ge(0), nullable=True, coerce=True),
                    "forest_area_ha": Column(float, Check.ge(0), nullable=True, coerce=True),
                },
                strict=False,
            ),
        ),
        GoldTable(
            "dim_commune",
            ("commune_key",),
            pa.DataFrameSchema(
                {
                    "commune_key": _string(unique=True),
                    "name": _string(),
                    "department_code": _string(),
                    "department_name": _string(nullable=True),
                    "region_code": _string(nullable=True),
                    "region_name": _string(nullable=True),
                    "country_key": _string(checks=Check.equal_to("FR")),
                    "centroid_lon": Column(float, Check.in_range(-180, 180), nullable=True, coerce=True),
                    "centroid_lat": Column(float, Check.in_range(-90, 90), nullable=True, coerce=True),
                },
                strict=False,
            ),
        ),
        GoldTable(
            "dim_source",
            ("source_key",),
            pa.DataFrameSchema(
                {
                    "source_key": _string(unique=True, checks=Check.isin(_SOURCE_KEYS)),
                    "measure_type": _string(),
                    "spatial_resolution": _string(nullable=True),
                    "min_year": Column(int, Check.in_range(1940, 2100), coerce=True),
                    "method_note": _string(nullable=True),
                },
                strict=False,
            ),
        ),
        GoldTable(
            "dim_landcover",
            ("landcover_key",),
            pa.DataFrameSchema(
                {
                    "landcover_key": _string(unique=True, checks=Check.isin(_LANDCOVER_KEYS)),
                    "class_code": _string(nullable=True),
                    "class_name": _string(),
                    "corine_mapping": _string(nullable=True),
                },
                strict=False,
            ),
        ),
        GoldTable(
            "dim_firecause",
            ("cause_key",),
            pa.DataFrameSchema(
                {
                    "cause_key": _string(unique=True),
                    "cause_group": _string(checks=Check.isin(_CAUSE_GROUPS)),
                    "cause_detail": _string(nullable=True),
                },
                strict=False,
            ),
        ),
        GoldTable(
            "fact_fireevent",
            ("fire_event_key",),
            pa.DataFrameSchema(
                {
                    "fire_event_key": _string(unique=True),
                    "source_key": _string(checks=Check.isin({"EFFIS_RDA", "BDIFF"})),
                    "country_key": _string(),
                    "commune_key": _string(nullable=True),
                    "firedate_key": Column(_DT, coerce=True),
                    "cause_key": _string(nullable=True),
                    "landcover_dominant_key": _string(nullable=True, checks=Check.isin(_LANDCOVER_KEYS)),
                    "area_ha": Column(float, Check.ge(0), coerce=True),
                    "ba_class": _string(nullable=True),
                    "pct_natura2000": Column(float, Check.in_range(0, 100.5), nullable=True, coerce=True),  # EFFIS PERCNA2K overshoots 100 (float noise, one polygon at 100.2456)
                    "centroid_lat": Column(float, Check.in_range(-90, 90), nullable=True, coerce=True),
                    "centroid_lon": Column(float, Check.in_range(-180, 180), nullable=True, coerce=True),
                },
                strict=False,
            ),
            notes=("BDIFF rows are the only ones allowed a commune_key",),
        ),
        GoldTable(
            "fact_burntarea_countrymonth",
            ("country_key", "month_key", "source_key"),
            pa.DataFrameSchema(
                {
                    "country_key": _string(),
                    "month_key": Column(_DT, Check(_first_of_month, error="month grain"), coerce=True),
                    "source_key": _string(checks=Check.isin({"EFFIS_RDA", "BDIFF"})),
                    "burnt_area_ha": Column(float, Check.ge(0), coerce=True),
                    "number_of_fires": Column("Int64", Check.ge(0), coerce=True),
                },
                unique=["country_key", "month_key", "source_key"],
                strict=False,
            ),
        ),
        GoldTable(
            "fact_burntarea_countryyear",
            ("country_key", "year_key", "source_key"),
            pa.DataFrameSchema(
                {
                    "country_key": _string(),
                    "year_key": Column(_DT, Check(_first_of_year, error="year grain"), coerce=True),
                    "source_key": _string(checks=Check.isin({"EFFIS_RDA", "BDIFF", "GWIS_GFAS"})),
                    "burnt_area_ha": Column(float, Check.ge(0), nullable=True, coerce=True),
                    "number_of_fires": Column("Int64", Check.ge(0), nullable=True, coerce=True),
                    "co2_emissions_t": Column(float, Check.ge(0), nullable=True, coerce=True),
                },
                checks=Check(_no_source_mixing, error="GWIS rows carry emissions only, EFFIS/BDIFF rows never do"),
                unique=["country_key", "year_key", "source_key"],
                strict=False,
            ),
        ),
        GoldTable(
            "fact_burntarea_landcover",
            ("country_key", "year_key", "landcover_key", "source_key"),
            pa.DataFrameSchema(
                {
                    "country_key": _string(),
                    "year_key": Column(_DT, Check(_first_of_year, error="year grain"), coerce=True),
                    "landcover_key": _string(checks=Check.isin(_LANDCOVER_KEYS)),
                    "source_key": _string(checks=Check.equal_to("EFFIS_RDA")),
                    "burnt_area_ha": Column(float, Check.ge(0), coerce=True),
                },
                unique=["country_key", "year_key", "landcover_key", "source_key"],
                strict=False,
            ),
        ),
        GoldTable(
            "fact_hotspot_countrymonth",
            ("country_key", "month_key", "source_key"),
            pa.DataFrameSchema(
                {
                    "country_key": _string(),
                    "month_key": Column(_DT, Check(_first_of_month, error="month grain"), coerce=True),
                    "source_key": _string(checks=Check.isin({"FIRMS_VIIRS"})),
                    "hotspot_count": Column("Int64", Check.ge(0), coerce=True),
                    "mean_frp": Column(float, Check.ge(0), nullable=True, coerce=True),
                    "max_frp": Column(float, Check.ge(0), nullable=True, coerce=True),
                },
                unique=["country_key", "month_key", "source_key"],
                strict=False,
            ),
        ),
        GoldTable(
            "fact_fireweather_countrymonth",
            ("country_key", "month_key"),
            pa.DataFrameSchema(
                {
                    "country_key": _string(),
                    "month_key": Column(_DT, Check(_first_of_month, error="month grain"), coerce=True),
                    "mean_fwi": Column(float, Check.ge(0), nullable=True, coerce=True),
                    "max_fwi": Column(float, Check.ge(0), nullable=True, coerce=True),
                    "days_fwi_high": Column("Int64", Check.in_range(0, 31), nullable=True, coerce=True),
                    "days_fwi_very_high": Column("Int64", Check.in_range(0, 31), nullable=True, coerce=True),
                    "mean_temp": Column(float, Check.in_range(-40, 45), nullable=True, coerce=True),
                    "total_precip": Column(float, Check.ge(0), nullable=True, coerce=True),
                    "precip_anomaly": Column(float, nullable=True, coerce=True),
                    "mean_rh": Column(float, Check.in_range(0, 100), nullable=True, coerce=True),
                    "spei": Column(
                        float,
                        [Check.in_range(-5, 5), Check(_spei_not_imputed, error="a mass of exact zeros means imputation")],
                        nullable=True,
                        coerce=True,
                    ),
                },
                unique=["country_key", "month_key"],
                strict=False,
            ),
        ),
    )
}


# Gold tables that exist only in v2: contract-checked like the others, but
# outside the v1 parity set and not bound by the v1 TMDL.
V2_GOLD_TABLES: dict[str, GoldTable] = {
    t.name: t
    for t in (
        GoldTable(
            "fact_season_countryyear",
            ("country_key", "year_key", "source_key"),
            pa.DataFrameSchema(
                {
                    "country_key": _string(),
                    "year_key": Column(_DT, Check(_first_of_year, error="year grain"), coerce=True),
                    "source_key": _string(checks=Check.isin({"EFFIS_RDA", "BDIFF"})),
                    "burnt_area_ha": Column(float, Check.ge(0), coerce=True),
                    "number_of_fires": Column("Int64", Check.ge(1), coerce=True),
                    "onset_date": Column(_DT, coerce=True),
                    "end_date": Column(_DT, coerce=True),
                    "onset_doy": Column("Int64", Check.in_range(1, 366), coerce=True),
                    "end_doy": Column("Int64", Check.in_range(1, 366), coerce=True),
                    "season_length_days": Column("Int64", Check.in_range(1, 366), coerce=True),
                    "low_count": Column(bool, coerce=True),
                },
                checks=Check(lambda df: bool((df["end_date"] >= df["onset_date"]).all()), error="end before onset"),
                unique=["country_key", "year_key", "source_key"],
                strict=False,
            ),
        ),
    )
}

ALL_GOLD_TABLES: dict[str, GoldTable] = {**GOLD_TABLES, **V2_GOLD_TABLES}


def validate(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate `df` against the contract of gold table `name`, raising on failure.

    Returns the (possibly dtype-coerced) frame so callers can write exactly
    what was validated.
    """
    table = ALL_GOLD_TABLES[name]
    return table.schema.validate(df, lazy=True)


def validate_processed(folder: Path | None = None) -> dict[str, str]:
    """Validate every gold parquet present in `folder`; return {table: 'ok' | error}."""
    folder = folder or config.DATA_PROCESSED
    results: dict[str, str] = {}
    for name in ALL_GOLD_TABLES:
        path = folder / f"{name}.parquet"
        if not path.exists():
            results[name] = "absent"
            continue
        try:
            validate(name, pd.read_parquet(path))
            results[name] = "ok"
        except pa.errors.SchemaErrors as exc:
            results[name] = f"FAILED\n{exc.failure_cases.head(20).to_string()}"
    return results


if __name__ == "__main__":
    import sys

    failed = False
    for table_name, status in validate_processed().items():
        print(f"{table_name:34s} {status}")
        failed |= status.startswith("FAILED")
    sys.exit(1 if failed else 0)
