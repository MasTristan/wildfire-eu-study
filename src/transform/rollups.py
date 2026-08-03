"""Roll up fact_fireevent into the descriptive marts (country x month/year,
country x year x land cover), per docs/data-model.md.

fact_fireevent stays the primary granular source; these are pure aggregations
of it, one row per (country, period, source) so EFFIS and BDIFF never mix in
a single summed figure (dim_source stays a grain key, not just an attribute).

co2_emissions_t comes from GWIS/GFAS (src/ingest/gwis.py), which measures the
fire independently of EFFIS. It therefore gets its OWN rows in
fact_burntarea_countryyear, under source_key = "GWIS_GFAS", with
burnt_area_ha / number_of_fires left null. Writing GFAS tonnes onto an
EFFIS_RDA row would be exactly the source mixing the grain rule above exists to
prevent, and would let a reader attribute GWIS emissions to EFFIS.

Consequence for the report: any measure over co2_emissions_t must clear the
dim_source filter (see the `CO2 Emissions (t)` measure), otherwise a page
filtered to EFFIS_RDA shows a blank.

Run:
    python src/transform/rollups.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402


def _load_fact_fireevent() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_PROCESSED / "fact_fireevent.parquet")


def build_countrymonth(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()
    df["month_key"] = df["firedate_key"].values.astype("datetime64[M]")
    g = df.groupby(["country_key", "month_key", "source_key"], as_index=False).agg(
        burnt_area_ha=("area_ha", "sum"),
        number_of_fires=("fire_event_key", "count"),
    )
    return g


def build_countryyear(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact.copy()
    df["year_key"] = pd.to_datetime(df["firedate_key"].dt.year.astype(str) + "-01-01")
    g = df.groupby(["country_key", "year_key", "source_key"], as_index=False).agg(
        burnt_area_ha=("area_ha", "sum"),
        number_of_fires=("fire_event_key", "count"),
    )
    g["co2_emissions_t"] = pd.Series([float("nan")] * len(g), dtype="float64")
    return pd.concat([g, _gwis_emission_rows()], ignore_index=True)


def _gwis_emission_rows() -> pd.DataFrame:
    """GWIS/GFAS emissions as their own source rows at country x year."""
    src = config.DATA_INTERIM / "gwis_gfas_emissions.csv"
    if not src.exists():
        print("  NOTE: gwis_gfas_emissions.csv missing, co2_emissions_t stays empty "
              "(run src/ingest/gwis.py). Not imputed.")
        return pd.DataFrame(
            columns=["country_key", "year_key", "source_key", "burnt_area_ha",
                     "number_of_fires", "co2_emissions_t"]
        )

    em = pd.read_csv(src, parse_dates=["month_key"])
    em["year"] = em["month_key"].dt.year
    em = em[em["year"].between(config.BASE_YEAR_MIN, config.BASE_YEAR_MAX)]
    yearly = em.groupby(["country_key", "year"], as_index=False)["co2_t"].sum()
    yearly["year_key"] = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
    yearly["source_key"] = "GWIS_GFAS"
    yearly["burnt_area_ha"] = pd.Series([float("nan")] * len(yearly), dtype="float64")
    yearly["number_of_fires"] = pd.Series([float("nan")] * len(yearly), dtype="float64")
    print(f"  + {len(yearly)} GWIS_GFAS emission rows "
          f"({yearly['year'].min()}-{yearly['year'].max()})")
    return yearly.rename(columns={"co2_t": "co2_emissions_t"})[
        ["country_key", "year_key", "source_key", "burnt_area_ha",
         "number_of_fires", "co2_emissions_t"]
    ]


def build_landcover(fact: pd.DataFrame) -> pd.DataFrame:
    df = fact[fact["source_key"] == "EFFIS_RDA"].copy()  # only EFFIS carries land cover
    df = df.dropna(subset=["landcover_dominant_key"])
    df["year_key"] = pd.to_datetime(df["firedate_key"].dt.year.astype(str) + "-01-01")
    g = df.groupby(
        ["country_key", "year_key", "landcover_dominant_key", "source_key"], as_index=False
    ).agg(burnt_area_ha=("area_ha", "sum"))
    g = g.rename(columns={"landcover_dominant_key": "landcover_key"})
    return g


def _sanity_check_against_rda_stats(countryyear: pd.DataFrame) -> None:
    """Print-only cross-check vs the independently-ingested EFFIS rda-stats
    macro-region totals (control figure, never a competing source per the
    data model's Dim_Source guardrail)."""
    rda = pd.read_csv(config.DATA_PROCESSED / "effis_rda_stats.csv")
    rda_eu = rda[rda["area"] == "EU"].set_index("year")["sum_rda_total_ha"]

    rollup_eu = countryyear[countryyear["source_key"] == "EFFIS_RDA"].copy()
    rollup_eu["year"] = rollup_eu["year_key"].dt.year
    eu_countries = pd.read_csv(
        Path(__file__).resolve().parent / "reference" / "country_reference.csv"
    ).query("macro_region == 'EU'")["country_key"]
    rollup_eu = rollup_eu[rollup_eu["country_key"].isin(eu_countries)]
    rollup_by_year = rollup_eu.groupby("year")["burnt_area_ha"].sum()

    print("  Sanity check vs EFFIS rda-stats (EU macro-region, control figure only):")
    for year in sorted(set(rollup_by_year.index) & set(rda_eu.index)):
        our = rollup_by_year.get(year, 0)
        theirs = rda_eu.get(year, float("nan"))
        print(f"    {year}: rollup={our:,.0f} ha  vs  rda-stats={theirs:,.0f} ha")


def _document_gwis_vs_effis(countryyear: pd.DataFrame) -> None:
    """GWIS emissions and EFFIS burnt area measure different populations of
    fire. Print the implied intensity so the divergence is a documented,
    inspected figure rather than an unexamined one."""
    df = countryyear.copy()
    df["year"] = df["year_key"].dt.year
    effis = (df[df["source_key"] == "EFFIS_RDA"]
             .groupby("year")["burnt_area_ha"].sum())
    gwis = (df[df["source_key"] == "GWIS_GFAS"]
            .groupby("year")["co2_emissions_t"].sum())
    common = sorted(set(effis.index) & set(gwis.index))
    if not common:
        return
    print("  GWIS/GFAS emissions vs EFFIS burnt area (different populations, "
          "never summed; GFAS also sees agricultural and sub-threshold fires):")
    for year in common[-5:]:
        ha, t = effis[year], gwis[year]
        ratio = t / ha if ha else float("nan")
        print(f"    {year}: EFFIS {ha:,.0f} ha, GFAS {t / 1e6:,.2f} Mt CO2 "
              f"-> {ratio:,.0f} t CO2 per EFFIS-mapped ha")


def build() -> None:
    fact = _load_fact_fireevent()

    print("Building fact_burntarea_countrymonth...")
    cm = build_countrymonth(fact)
    cm.to_parquet(config.DATA_PROCESSED / "fact_burntarea_countrymonth.parquet", index=False)
    print(f"  {len(cm)} rows")

    print("Building fact_burntarea_countryyear...")
    cy = build_countryyear(fact)
    cy.to_parquet(config.DATA_PROCESSED / "fact_burntarea_countryyear.parquet", index=False)
    print(f"  {len(cy)} rows")

    print("Building fact_burntarea_landcover...")
    lc = build_landcover(fact)
    lc.to_parquet(config.DATA_PROCESSED / "fact_burntarea_landcover.parquet", index=False)
    print(f"  {len(lc)} rows")

    _sanity_check_against_rda_stats(cy)
    _document_gwis_vs_effis(cy)


if __name__ == "__main__":
    build()
