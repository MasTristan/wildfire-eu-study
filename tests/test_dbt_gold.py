"""dbt must reproduce the v1 Python rollups value for value.

A synthetic silver layer (fact_fireevent) and interim GWIS table are written
to a temporary lake, dbt builds the three rollup models and their schema
tests there, and each output is compared with the v1 function from
src/transform/rollups.py on the same input, through the same canonical form
the v1 -> v2 parity test uses. This is the equivalence proof that lets the
real parity test be trusted once data exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import config, contracts
from src.parity import canonical
from src.transform import rollups
from tests.conftest import run_dbt

ROLLUPS = {
    "fact_burntarea_countrymonth": rollups.build_countrymonth,
    "fact_burntarea_countryyear": rollups.build_countryyear,
    "fact_burntarea_landcover": rollups.build_landcover,
}


def _synthetic_fireevent(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    landcover = list(sorted({"broadleaf", "conifer", "mixed", "sclerophyllous", "transitional",
                             "other_natural", "agricultural", "artificial", "other"}))
    df = pd.DataFrame({
        "fire_event_key": [f"F{i:06d}" for i in range(n)],
        "source_key": rng.choice(["EFFIS_RDA", "BDIFF"], n, p=[0.7, 0.3]),
        "country_key": rng.choice(["ES", "PT", "FR", "GR", "IT"], n),
        "commune_key": None,
        "firedate_key": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 365 * 9, n), unit="D"),
        "cause_key": None,
        "landcover_dominant_key": rng.choice(np.array([*landcover, None], dtype=object), n),
        "area_ha": rng.gamma(1.2, 60.0, n).round(2),
        "ba_class": None,
        "pct_natura2000": rng.uniform(0, 100, n),
        "centroid_lat": rng.uniform(36, 48, n),
        "centroid_lon": rng.uniform(-9, 26, n),
    })
    df.loc[df["source_key"] == "BDIFF", "landcover_dominant_key"] = None
    df["commune_key"] = df["commune_key"].astype(object)
    df["cause_key"] = df["cause_key"].astype(object)
    df["ba_class"] = df["ba_class"].astype(object)
    return df


def _synthetic_gwis(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2003-01-01", "2026-06-01", freq="MS")
    rows = [(c, m, float(rng.gamma(2.0, 1e4))) for c in ("ES", "PT", "FR") for m in months]
    return pd.DataFrame(rows, columns=["country_key", "month_key", "co2_t"])


@pytest.fixture(scope="module")
def built_lake(tmp_path_factory, dbt_manifest):
    lake_dir = tmp_path_factory.mktemp("lake")
    silver, interim, gold = lake_dir / "silver", lake_dir / "interim", lake_dir / "gold_star"
    for d in (silver, interim, gold):
        d.mkdir()
    fact = _synthetic_fireevent()
    contracts.validate("fact_fireevent", fact)
    fact.to_parquet(silver / "fact_fireevent.parquet", index=False)
    _synthetic_gwis().to_csv(interim / "gwis_gfas_emissions.csv", index=False)
    env = {
        "WILDFIRE_SILVER": str(silver),
        "WILDFIRE_INTERIM": str(interim),
        "WILDFIRE_GOLD_STAR": str(gold),
        "WILDFIRE_DUCKDB": str(lake_dir / "wildfire.duckdb"),
    }
    res = run_dbt(["build", "--select", "+fact_burntarea_countrymonth +fact_burntarea_countryyear +fact_burntarea_landcover"], env)
    assert res.returncode == 0, res.stdout + res.stderr
    return fact, interim, gold


@pytest.mark.parametrize("name", sorted(ROLLUPS))
def test_dbt_rollup_equals_python_rollup(built_lake, name, monkeypatch):
    fact, interim, gold = built_lake
    monkeypatch.setattr(config, "DATA_INTERIM", interim)  # _gwis_emission_rows reads it
    expected = ROLLUPS[name](fact)
    got = pd.read_parquet(gold / f"{name}.parquet")
    keys = contracts.GOLD_TABLES[name].keys
    columns = list(expected.columns)
    assert set(columns) <= set(got.columns), f"{name}: dbt output lacks {set(columns) - set(got.columns)}"
    assert_frame_equal(
        canonical(expected, keys, columns), canonical(got, keys, columns),
        check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-6, obj=name,
    )


@pytest.mark.parametrize("name", sorted(ROLLUPS))
def test_dbt_rollup_satisfies_contract(built_lake, name):
    _, _, gold = built_lake
    contracts.validate(name, pd.read_parquet(gold / f"{name}.parquet"))
