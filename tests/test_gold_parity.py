"""v1 -> v2 parity: the gold layer the v2 platform produces must equal, value
for value, the gold layer v1 produced and Power BI was built on.

Reference: data/reference/v1/ (gitignored), produced once from the v1 pipeline
by scripts/freeze_v1_reference.py and archived as a release asset on tag
v1.0. Its manifest.json (row counts, columns) is small and may be committed;
the parquets are downloaded when a full comparison is wanted.

Three levels, each skipping cleanly when its input is absent:
  1. manifest vs current parquets: same columns, same row counts;
  2. reference parquets vs current parquets: exact equality on every v1
     column after sorting on the contract keys (extra v2 columns are allowed,
     missing or changed values are not);
  3. the reference itself satisfies the contracts, so a drift in the
     contracts cannot hide behind a drift in the data.

This test is the gate of the whole Dagster/dbt migration (PLAN.md, phase 0):
blocking in CI as soon as the reference exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import config, contracts, lake
from src.parity import canonical

REFERENCE = Path(config.ROOT) / "data" / "reference" / "v1"
MANIFEST = REFERENCE / "manifest.json"


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("no v1 reference manifest (scripts/freeze_v1_reference.py, or download the v1.0 asset)")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _current(name: str) -> pd.DataFrame:
    """The table under test: gold/star once dbt has produced it, else the v1 output."""
    path = lake.GOLD_STAR / f"{name}.parquet"
    if not path.exists():
        path = config.DATA_PROCESSED / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{name} not built in {lake.GOLD_STAR} nor {config.DATA_PROCESSED}")
    return pd.read_parquet(path)


def _reference(name: str) -> pd.DataFrame:
    path = REFERENCE / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not downloaded (release asset gold-v1.zip on tag v1.0)")
    return pd.read_parquet(path)


@pytest.mark.parametrize("name", sorted(contracts.GOLD_TABLES))
def test_manifest_shape_matches_current(name: str):
    manifest = _manifest()
    if name not in manifest["tables"]:
        pytest.skip(f"{name} absent from the v1 reference")
    entry = manifest["tables"][name]
    cur = _current(name)
    missing = set(entry["columns"]) - set(cur.columns)
    assert not missing, f"{name}: v1 columns missing in v2 output: {sorted(missing)}"
    assert len(cur) == entry["rows"], f"{name}: {len(cur)} rows now vs {entry['rows']} in v1"


@pytest.mark.parametrize("name", sorted(contracts.GOLD_TABLES))
def test_values_equal_v1_reference(name: str):
    ref = _reference(name)
    cur = _current(name)
    keys = contracts.GOLD_TABLES[name].keys
    columns = list(ref.columns)
    missing = set(columns) - set(cur.columns)
    assert not missing, f"{name}: v1 columns missing in v2 output: {sorted(missing)}"
    assert_frame_equal(
        canonical(ref, keys, columns),
        canonical(cur, keys, columns),
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
        obj=name,
    )


@pytest.mark.parametrize("name", sorted(contracts.GOLD_TABLES))
def test_reference_satisfies_contract(name: str):
    contracts.validate(name, _reference(name))
