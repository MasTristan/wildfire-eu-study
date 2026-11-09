from __future__ import annotations

from dagster import Definitions, in_process_executor, load_assets_from_modules

from src.orchestration.assets import bronze, features, gold, models, silver
from src.orchestration.assets.checks import gold_contract_checks
from src.orchestration.resources import dbt_cli

defs = Definitions(
    assets=load_assets_from_modules([bronze, silver, gold, features, models]),
    asset_checks=gold_contract_checks,
    resources={"dbt": dbt_cli},
    # In-process execution (ADR 0005): every step runs in the launching
    # process. Dagster's default multiprocess executor spawns a process per
    # step, which is fragile on Windows and buys nothing here, where steps are
    # I/O-bound downloads or single-machine transforms run one after another.
    executor=in_process_executor,
)
