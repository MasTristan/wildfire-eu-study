from __future__ import annotations

from dagster import Definitions, load_assets_from_modules

from src.orchestration.assets import bronze, features, gold, silver
from src.orchestration.assets.checks import gold_contract_checks
from src.orchestration.resources import dbt_cli

defs = Definitions(
    assets=load_assets_from_modules([bronze, silver, gold, features]),
    asset_checks=gold_contract_checks,
    resources={"dbt": dbt_cli},
)
