"""The gold contracts (src/contracts.py) must agree with the semantic model and
must accept the parquets the pipeline produces.

The first group runs without data and keeps the contracts honest against the
TMDL column definitions. The second validates real parquets when present and
skips cleanly otherwise, like tests/test_data_layer.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from src import config, contracts

TMDL_TABLES = Path(config.ROOT) / "powerbi" / "WildfireEU.SemanticModel" / "definition" / "tables"


def _tmdl_source_columns(name: str) -> set[str]:
    text = (TMDL_TABLES / f"{name}.tmdl").read_text(encoding="utf-8")
    return set(re.findall(r"^\s*sourceColumn:\s*(\S+)", text, flags=re.MULTILINE))


@pytest.mark.parametrize("name", sorted(contracts.GOLD_TABLES))
def test_contract_covers_every_column_the_report_reads(name: str):
    """Every physical column the TMDL partition binds must be declared in the
    contract: dropping one in a transform would break the report on refresh."""
    declared = set(contracts.GOLD_TABLES[name].schema.columns)
    bound = _tmdl_source_columns(name)
    assert bound, f"no sourceColumn found in {name}.tmdl"
    missing = bound - declared
    assert not missing, f"{name}: columns read by Power BI but absent from the contract: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(contracts.GOLD_TABLES))
def test_contract_keys_are_declared_columns(name: str):
    table = contracts.GOLD_TABLES[name]
    assert set(table.keys) <= set(table.schema.columns)


def test_contract_rejects_source_mixing():
    """The GWIS invariant must fail loudly, not silently pass on an empty check."""
    df = pd.DataFrame(
        {
            "country_key": ["ES", "ES"],
            "year_key": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "source_key": ["EFFIS_RDA", "GWIS_GFAS"],
            "burnt_area_ha": [100.0, 5.0],  # a GWIS row carrying burnt area
            "number_of_fires": [3, None],
            "co2_emissions_t": [None, 1000.0],
        }
    )
    with pytest.raises(pa.errors.SchemaErrors):
        contracts.validate("fact_burntarea_countryyear", df)


def test_contract_accepts_a_well_formed_countryyear():
    df = pd.DataFrame(
        {
            "country_key": ["ES", "ES"],
            "year_key": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "source_key": ["EFFIS_RDA", "GWIS_GFAS"],
            "burnt_area_ha": [100.0, None],
            "number_of_fires": [3, None],
            "co2_emissions_t": [None, 1000.0],
        }
    )
    out = contracts.validate("fact_burntarea_countryyear", df)
    assert len(out) == 2


@pytest.mark.parametrize("name", sorted(contracts.GOLD_TABLES))
def test_processed_parquet_satisfies_contract(name: str):
    path = config.DATA_PROCESSED / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built (run the pipeline locally)")
    contracts.validate(name, pd.read_parquet(path))
