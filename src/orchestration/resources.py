from __future__ import annotations

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

dbt_cli = DbtCliResource(project_dir=dbt_project, profiles_dir=str(lake.DBT_PROJECT_DIR))
