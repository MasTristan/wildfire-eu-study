"""Asset checks: every gold table is validated against its pandera contract
after dbt writes it. The contract is the same object the silver assets and
the tests use (src/contracts.py), so there is one definition of what a table
must look like, enforced in three places.
"""
from __future__ import annotations

import pandas as pd
import pandera as pa
from dagster import AssetCheckResult, AssetKey, asset_check

from src import contracts, lake


def _make(name: str):
    @asset_check(asset=AssetKey(["gold", name]), name="contract", description=f"pandera contract of {name}")
    def _check() -> AssetCheckResult:
        path = lake.GOLD_STAR / f"{name}.parquet"
        if not path.exists():
            return AssetCheckResult(passed=False, metadata={"error": f"{path} missing"})
        df = pd.read_parquet(path)
        try:
            contracts.validate(name, df)
        except pa.errors.SchemaErrors as exc:
            return AssetCheckResult(
                passed=False,
                metadata={"rows": int(len(df)), "failures": exc.failure_cases.head(20).to_string()},
            )
        return AssetCheckResult(passed=True, metadata={"rows": int(len(df))})

    return _check


gold_contract_checks = [_make(name) for name in contracts.GOLD_TABLES]
