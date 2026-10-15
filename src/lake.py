"""Lakehouse layout for v2 and the bridge from the v1 folders.

Three layers, each a folder of Parquet (and NetCDF or CSV where the source
dictates), overridable by environment variable so tests and CI can point the
whole pipeline at a temporary tree:

    bronze   raw as received, never rewritten        (v1: data/raw)
    silver   one table per source, clean, typed      (v1: data/processed, see below)
    gold     star schema for Power BI, feature store, model outputs

Migration bridge, phase 1 of PLAN.md: the v1 transform functions write their
tables to data/processed. Until they are rewritten, that folder IS the silver
layer: the Dagster silver assets call the v1 builders unchanged, and dbt reads
silver to produce gold/star. Power BI keeps reading data/processed until the
parity gate (tests/test_gold_parity.py) proves gold/star equal, then its
DataFolder parameter moves to gold/star and data/processed becomes silver for
good. No file is relocated by hand at any point.
"""
from __future__ import annotations

import os
from pathlib import Path

from src import config


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


ROOT: Path = config.ROOT
LAKE: Path = _env_path("WILDFIRE_LAKE", ROOT / "data" / "lake")

BRONZE: Path = _env_path("WILDFIRE_BRONZE", config.DATA_RAW)
INTERIM: Path = _env_path("WILDFIRE_INTERIM", config.DATA_INTERIM)
SILVER: Path = _env_path("WILDFIRE_SILVER", config.DATA_PROCESSED)
GOLD: Path = _env_path("WILDFIRE_GOLD", LAKE / "gold")
GOLD_STAR: Path = _env_path("WILDFIRE_GOLD_STAR", GOLD / "star")
GOLD_FEATURES: Path = GOLD / "features"
GOLD_MODEL_OUT: Path = GOLD / "model_out"

DBT_PROJECT_DIR: Path = ROOT / "dbt"


def dbt_env() -> dict[str, str]:
    """Environment the dbt project reads its locations from (profiles.yml, sources.yml)."""
    return {
        "WILDFIRE_SILVER": str(SILVER),
        "WILDFIRE_INTERIM": str(INTERIM),
        "WILDFIRE_GOLD_STAR": str(GOLD_STAR),
    }


def ensure_dirs() -> None:
    for d in (SILVER, GOLD_STAR, GOLD_FEATURES, GOLD_MODEL_OUT):
        d.mkdir(parents=True, exist_ok=True)
