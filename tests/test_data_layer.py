"""Data-layer contract tests: schema, referential integrity, known invariants.

These need the actual parquet outputs (data/processed/), which are gitignored
and not produced in CI (no network there, see README.md / ci.yml). Every test
here skips cleanly when its input is absent, so CI stays green; run locally
after the ingest/transform pipeline to get real coverage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import config

PROCESSED = config.DATA_PROCESSED


def _load_or_skip(name: str) -> pd.DataFrame:
    path = PROCESSED / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built (run the ingest/transform pipeline locally)")
    return pd.read_parquet(path)


# --- Dimensions -----------------------------------------------------------------

def test_dim_date_covers_expected_range():
    df = _load_or_skip("dim_date")
    assert df["date_key"].min() <= pd.Timestamp("2000-01-01")
    assert df["date_key"].max() >= pd.Timestamp("2026-12-31")
    assert df["is_fire_season"].dtype == bool
    assert set(df.loc[df["is_fire_season"], "month"]) == {6, 7, 8, 9}


def test_dim_country_schema_and_values():
    df = _load_or_skip("dim_country")
    required = {
        "country_key", "iso2", "iso3", "name_en", "name_fr", "macro_region",
        "is_eu", "is_mediterranean", "land_area_km2", "forest_area_ha",
    }
    assert required.issubset(df.columns)
    assert df["country_key"].is_unique
    assert set(df["macro_region"]) <= {"EU", "EU_non", "ME_AF"}
    assert df["is_eu"].dtype == bool
    # at least the well-known EU Mediterranean core must resolve correctly
    core = df.set_index("country_key")
    for cc in ("FR", "ES", "IT", "PT", "GR"):
        assert core.loc[cc, "is_eu"], cc
        assert core.loc[cc, "is_mediterranean"], cc


def test_dim_landcover_matches_effis_schema():
    df = _load_or_skip("dim_landcover")
    assert set(df["landcover_key"]) == {
        "broadleaf", "conifer", "mixed", "sclerophyllous", "transitional",
        "other_natural", "agricultural", "artificial", "other",
    }


def test_dim_firecause_groups_are_known():
    df = _load_or_skip("dim_firecause")
    assert set(df["cause_group"]) <= {"natural", "accidental", "malicious", "involuntary", "unknown"}
    assert df["cause_key"].is_unique


# --- fact_fireevent: schema + referential integrity ------------------------------

def test_fact_fireevent_schema():
    df = _load_or_skip("fact_fireevent")
    required = {
        "fire_event_key", "source_key", "country_key", "commune_key",
        "firedate_key", "cause_key", "landcover_dominant_key", "area_ha",
        "ba_class", "pct_natura2000", "centroid_lat", "centroid_lon",
    }
    assert required.issubset(df.columns)
    assert df["fire_event_key"].is_unique
    assert (df["area_ha"] >= 0).all()
    assert set(df["source_key"]) <= {"EFFIS_RDA", "BDIFF"}


def test_fact_fireevent_country_fk_integrity():
    fact = _load_or_skip("fact_fireevent")
    countries = _load_or_skip("dim_country")
    assert set(fact["country_key"]) <= set(countries["country_key"])


def test_fact_fireevent_cause_fk_integrity():
    fact = _load_or_skip("fact_fireevent")
    causes = _load_or_skip("dim_firecause")
    used = set(fact["cause_key"].dropna())
    assert used <= set(causes["cause_key"])


def test_fact_fireevent_landcover_fk_integrity():
    fact = _load_or_skip("fact_fireevent")
    landcover = _load_or_skip("dim_landcover")
    used = set(fact["landcover_dominant_key"].dropna())
    assert used <= set(landcover["landcover_key"])


def test_fact_fireevent_bdiff_commune_fk_integrity():
    fact = _load_or_skip("fact_fireevent")
    communes = _load_or_skip("dim_commune")
    used = set(fact.loc[fact["source_key"] == "BDIFF", "commune_key"].dropna())
    assert used <= set(communes["commune_key"])


# --- Rollups: schema + cross-check vs the independently-ingested control figure --

def test_rollup_countryyear_matches_rda_stats_control():
    """fact_burntarea_countryyear (EFFIS_RDA, EU countries) rolled up must be
    close to the independently-ingested effis_rda_stats.csv EU total: both
    trace back to EFFIS's Rapid Damage Assessment, so a large divergence
    signals a bug in the country mapping or the rollup, not a real methodology
    gap (docs/data-model.md: rda-stats is a control figure, not a competing
    source, but it must still roughly agree)."""
    cy = _load_or_skip("fact_burntarea_countryyear")
    rda_path = config.DATA_PROCESSED / "effis_rda_stats.csv"
    if not rda_path.exists():
        pytest.skip("effis_rda_stats.csv not built")
    rda = pd.read_csv(rda_path)
    rda_eu = rda[rda["area"] == "EU"].set_index("year")["sum_rda_total_ha"]

    ref = pd.read_csv(Path(config.ROOT) / "src" / "transform" / "reference" / "country_reference.csv")
    eu_countries = set(ref.query("macro_region == 'EU'")["country_key"])

    rollup = cy[(cy["source_key"] == "EFFIS_RDA") & (cy["country_key"].isin(eu_countries))].copy()
    rollup["year"] = pd.to_datetime(rollup["year_key"]).dt.year
    rollup_by_year = rollup.groupby("year")["burnt_area_ha"].sum()

    common_years = sorted(set(rollup_by_year.index) & set(rda_eu.index))
    assert common_years, "no overlapping years between rollup and rda-stats control"
    for year in common_years:
        our, theirs = rollup_by_year[year], rda_eu[year]
        assert our == pytest.approx(theirs, rel=0.02), (
            f"{year}: rollup={our:,.0f} ha vs rda-stats={theirs:,.0f} ha (control figure)"
        )


def test_fact_burntarea_countrymonth_schema():
    df = _load_or_skip("fact_burntarea_countrymonth")
    assert {"country_key", "month_key", "source_key", "burnt_area_ha", "number_of_fires"}.issubset(df.columns)
    assert (df["burnt_area_ha"] >= 0).all()
    assert (df["number_of_fires"] >= 0).all()


def test_fact_burntarea_landcover_schema_and_fk():
    df = _load_or_skip("fact_burntarea_landcover")
    landcover = _load_or_skip("dim_landcover")
    assert set(df["landcover_key"]) <= set(landcover["landcover_key"])
    assert set(df["source_key"]) == {"EFFIS_RDA"}  # BDIFF carries no land cover


# --- Phase-2 facts: schema only when present --------------------------------------

def test_fact_hotspot_countrymonth_schema_if_present():
    df = _load_or_skip("fact_hotspot_countrymonth")
    assert {"country_key", "month_key", "source_key", "hotspot_count", "mean_frp", "max_frp"}.issubset(df.columns)
    assert (df["hotspot_count"] >= 0).all()


def test_fact_fireweather_countrymonth_schema_if_present():
    df = _load_or_skip("fact_fireweather_countrymonth")
    required = {
        "country_key", "month_key", "mean_fwi", "max_fwi", "days_fwi_high",
        "days_fwi_very_high", "mean_temp", "total_precip", "precip_anomaly",
        "mean_rh", "spei",
    }
    assert required.issubset(df.columns)
    assert (df["days_fwi_high"].dropna() >= 0).all()


# --- Phase 3: the climate explanatory layer ---------------------------------------
# Each test below maps to one line of the verification table in
# docs/specs/phase3-climate-explanatory.md.

def test_fireweather_reaches_the_effis_horizon():
    """Lot 1. The FWI series must start no later than 2016, the EFFIS perimeter
    floor: the whole point of the Why page is to put fire weather and burnt area
    on a shared horizon rather than a long series against a short one."""
    df = _load_or_skip("fact_fireweather_countrymonth")
    assert df["month_key"].dt.year.min() <= 2016


def test_fireweather_has_no_monthly_gap_per_country():
    """Lot 1. Every country must have an unbroken run of months between its own
    first and last observation. A hole would silently drop a fire season out of
    any monthly average."""
    df = _load_or_skip("fact_fireweather_countrymonth")
    gaps = {}
    for country, grp in df.groupby("country_key"):
        expected = pd.date_range(grp["month_key"].min(), grp["month_key"].max(), freq="MS")
        missing = set(expected) - set(grp["month_key"])
        if missing:
            gaps[country] = len(missing)
    assert not gaps, f"missing months per country: {gaps}"


def test_fireweather_covers_france():
    """Regression guard. Natural Earth codes France, Norway and Kosovo as
    ISO_A2 = '-99', which silently dropped them from every point-in-polygon
    join until 2026-07-31. France has its own deep-dive page; its absence from
    the climate table was invisible on screen."""
    df = _load_or_skip("fact_fireweather_countrymonth")
    assert {"FR", "NO"} <= set(df["country_key"])


def test_fireweather_climate_columns_are_populated():
    """Lot 4. mean_temp, total_precip and mean_rh come from ERA5 and must cover
    the whole FWI horizon; these were 100% null before phase 3."""
    df = _load_or_skip("fact_fireweather_countrymonth")
    for col in ("mean_temp", "total_precip", "mean_rh"):
        assert df[col].notna().mean() > 0.95, f"{col} only {df[col].notna().mean():.1%} populated"
    assert df["mean_temp"].between(-40, 45).all()
    assert (df["total_precip"].dropna() >= 0).all()
    assert df["mean_rh"].between(0, 100).all()


def test_precip_anomaly_is_a_deviation_not_a_level():
    """Lot 4. precip_anomaly is a percent deviation from the 1991-2020 normal,
    so it is centred near zero and signed. A column of positive levels would
    mean the normal never got subtracted."""
    df = _load_or_skip("fact_fireweather_countrymonth")
    anomaly = df["precip_anomaly"].dropna()
    assert not anomaly.empty
    assert (anomaly < 0).any() and (anomaly > 0).any()
    assert abs(anomaly.median()) < 40


def test_spei_is_present_and_never_imputed():
    """Lot 3. SPEI is populated wherever EDO's mask reaches, and stays null
    elsewhere rather than being filled with a neutral zero.

    Measured 2026-07-31, EDO's domain happens to cover every country in the
    fire-weather table, so this asserts the shape of the values rather than the
    presence of holes: a standardized index is centred on zero, signed, and
    bounded in practice, and a mass of exact zeros would mean imputation."""
    df = _load_or_skip("fact_fireweather_countrymonth")
    spei = df["spei"].dropna()
    assert not spei.empty, "spei is entirely null"
    assert spei.abs().max() <= 5
    assert (spei == 0).mean() < 0.01, "a mass of exact zeros means imputation"
    assert (spei < 0).any() and (spei > 0).any()
    assert abs(spei.median()) < 1


def test_co2_emissions_are_populated_at_country_year():
    """Lot 2. The DAX measure `CO2 Emissions (t)` was always blank before
    phase 3. GWIS/GFAS now fills it on more than 90% of country-year pairs."""
    cy = _load_or_skip("fact_burntarea_countryyear")
    cy = cy.copy()
    cy["year"] = pd.to_datetime(cy["year_key"]).dt.year
    covered = cy.groupby(["country_key", "year"])["co2_emissions_t"].apply(lambda s: s.notna().any())
    assert covered.mean() > 0.90, f"only {covered.mean():.1%} of country-years carry CO2"
    assert (cy["co2_emissions_t"].dropna() >= 0).all()


def test_gwis_emissions_never_carry_burnt_area():
    """Lot 2. GWIS is an independent measurement of the fire, so it gets its own
    source rows. If a GWIS row ever carried burnt_area_ha, summing the column
    across sources would double count against EFFIS."""
    cy = _load_or_skip("fact_burntarea_countryyear")
    gwis = cy[cy["source_key"] == "GWIS_GFAS"]
    assert not gwis.empty
    assert gwis["burnt_area_ha"].isna().all()
    assert gwis["co2_emissions_t"].notna().all()
    others = cy[cy["source_key"] != "GWIS_GFAS"]
    assert others["co2_emissions_t"].isna().all(), (
        "emissions must live only on GWIS rows, never attributed to EFFIS or BDIFF"
    )


def test_gwis_excludes_countries_outside_the_study_domain():
    """Lot 2. GFAS ships national totals that cannot be clipped afterwards.
    Russia's total is Siberian and would outweigh every European country."""
    cy = _load_or_skip("fact_burntarea_countryyear")
    gwis = cy[cy["source_key"] == "GWIS_GFAS"]
    assert "RU" not in set(gwis["country_key"])
    assert {"FR", "ES", "IT", "GR", "PT"} <= set(gwis["country_key"])


def test_dim_source_declares_exactly_the_sources_in_use():
    """Assainissement. A source declared in dim_source but never ingested shows
    up in slicers and filters to nothing (FIRMS_MODIS did, until phase 3).
    Every declared source must either appear in a fact table or be documented
    here as a control-only figure."""
    sources = _load_or_skip("dim_source")
    declared = set(sources["source_key"])
    used: set[str] = set()
    for name in ("fact_fireevent", "fact_burntarea_countryyear",
                 "fact_burntarea_countrymonth", "fact_hotspot_countrymonth"):
        path = PROCESSED / f"{name}.parquet"
        if path.exists():
            used |= set(pd.read_parquet(path, columns=["source_key"])["source_key"])
    # Sources that legitimately carry no fact row of their own: the climate
    # grids feed fact_fireweather (which has no source_key column, one grain
    # one source) and rda-stats is the control figure.
    documented_without_facts = {"EFFIS_STATS", "CDS_FWI", "ERA5", "EDO_SPEI"}
    orphans = declared - used - documented_without_facts
    assert not orphans, f"declared in dim_source but never ingested: {sorted(orphans)}"
