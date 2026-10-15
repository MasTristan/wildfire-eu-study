"""Gold: the star schema Power BI reads, produced by dbt from silver.

One Dagster asset per dbt model, keyed by the model name, so a gold table has
the same name in the lake, in dbt, in Dagster and in the TMDL partition that
binds it. The dbt lineage (silver sources -> models) is read from the manifest.
"""
from collections.abc import Mapping
from typing import Any

from dagster import AssetExecutionContext, AssetKey
from dagster_dbt import DagsterDbtTranslator, DbtCliResource, dbt_assets

from src.orchestration.resources import dbt_project


class _Translator(DagsterDbtTranslator):
    """Keys carry the lake layer: a dbt source is the silver asset of the same
    name (so the Python and SQL halves form one graph), a model is gold/<name>,
    the file Power BI binds."""

    def get_asset_key(self, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
        name = dbt_resource_props["name"]
        if dbt_resource_props["resource_type"] == "source":
            return AssetKey(["silver", name])
        return AssetKey(["gold", name])

    def get_group_name(self, dbt_resource_props: Mapping[str, Any]) -> str | None:
        return "gold"


@dbt_assets(manifest=dbt_project.manifest_path, dagster_dbt_translator=_Translator())
def gold_star(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()
