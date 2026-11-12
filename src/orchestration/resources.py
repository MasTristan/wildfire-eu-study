from __future__ import annotations

import os
import sys
from pathlib import Path

from dagster_dbt import DbtCliResource, DbtProject

from src import lake

dbt_project = DbtProject(
    project_dir=lake.DBT_PROJECT_DIR,
    profiles_dir=lake.DBT_PROJECT_DIR,
)
# Under `dagster dev` this runs `dbt parse` so the manifest the assets are
# built from is always current; elsewhere the manifest must already exist
# (make dbt-parse, or the test session fixture).
dbt_project.prepare_if_dev()



def dbt_executable() -> str:
    """The dbt installed next to the running interpreter, so the assets work without an
    activated environment (plain `python -m pytest`, a scheduler); `dbt` on PATH otherwise."""
    exe = Path(sys.executable).with_name("dbt.exe" if os.name == "nt" else "dbt")
    return str(exe) if exe.exists() else "dbt"


dbt_cli = DbtCliResource(project_dir=dbt_project, profiles_dir=str(lake.DBT_PROJECT_DIR),
                         dbt_executable=dbt_executable())
