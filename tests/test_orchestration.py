"""The Dagster code location must load, and its graph must carry the lake end
to end: bronze sources, silver tables, every gold table, one contract check per
gold table, and the dbt sources joined to the silver assets of the same name.
"""
from __future__ import annotations

import pytest
from dagster import AssetKey

from src import contracts


@pytest.fixture(scope="module")
def defs(dbt_manifest):
    from src.orchestration.definitions import defs

    return defs


def test_definitions_load(defs):
    assert defs.get_repository_def() is not None


def test_every_gold_table_is_a_dbt_asset(defs):
    keys = {spec.key for spec in defs.resolve_all_asset_specs()}
    for name in contracts.GOLD_TABLES:
        assert AssetKey(["gold", name]) in keys, f"{name} has no gold Dagster asset"


def test_dbt_sources_resolve_to_silver_assets(defs):
    """A dbt source and the silver asset that writes it share one key, so the
    graph shows one lineage, not a Python tree and a SQL tree side by side."""
    graph = defs.get_repository_def().asset_graph
    for name in ("fact_fireevent", "dim_country", "dim_date"):
        assert graph.has(AssetKey(["silver", name]))
    gold_node = graph.get(AssetKey(["gold", "fact_burntarea_countrymonth"]))
    assert AssetKey(["silver", "fact_fireevent"]) in gold_node.parent_keys
    # the silver table is produced by Python and only read by dbt
    assert graph.get(AssetKey(["silver", "fact_fireevent"])).group_name == "silver"


def test_dbt_tests_surface_as_asset_checks(defs):
    """dbt schema tests become Dagster asset checks on the gold tables, next to
    the pandera contract check: one place to see a table's health."""
    names = {c.name for c in defs.get_repository_def().asset_graph.asset_check_keys
             if c.asset_key == AssetKey(["gold", "fact_burntarea_countryyear"])}
    assert "contract" in names
    assert any(n.startswith("no_source_mixing") for n in names)


def test_one_contract_check_per_gold_table(defs):
    check_keys = {(c.asset_key, c.name) for c in defs.get_repository_def().asset_graph.asset_check_keys}
    for name in contracts.GOLD_TABLES:
        assert (AssetKey(["gold", name]), "contract") in check_keys


def test_groups_follow_the_lake_layers(defs):
    groups = {spec.key.to_user_string(): spec.group_name for spec in defs.resolve_all_asset_specs()}
    assert groups["bronze/effis_perimeters"] == "bronze"
    assert groups["silver/fact_fireevent"] == "silver"
    assert groups["gold/fact_burntarea_countryyear"] == "gold"


def test_yearly_partitions_cover_the_study(defs):
    graph = defs.get_repository_def().asset_graph
    pdef = graph.get(AssetKey(["bronze", "cds_fwi_raw"])).partitions_def
    assert pdef is not None
    keys = pdef.get_partition_keys()
    assert keys[0] == "2006" and "2026" in keys


def test_feature_store_assets_are_wired(defs):
    graph = defs.get_repository_def().asset_graph
    node = graph.get(AssetKey(["gold", "features_daily"]))
    assert node.partitions_def is not None
    assert {AssetKey(["silver", "grid_cells"]), AssetKey(["bronze", "era5_daily_raw"]),
            AssetKey(["silver", "fact_fireevent"])} <= set(node.parent_keys)
    monthly = graph.get(AssetKey(["bronze", "era5_daily_raw"])).partitions_def
    assert monthly is not None and monthly.get_partition_keys()[0] == "2016-01"


def test_raw_sources_are_partitioned(defs):
    graph = defs.get_repository_def().asset_graph
    for name in ("bdiff_raw", "edo_spei_raw", "cds_fwi_raw", "firms_hotspots_raw"):
        assert graph.get(AssetKey(["bronze", name])).partitions_def is not None, name
    assert graph.get(AssetKey(["bronze", "era5_raw"])).partitions_def.get_partition_keys()[0] == "1991-2000"
    assert AssetKey(["bronze", "bdiff_raw"]) in graph.get(AssetKey(["bronze", "bdiff_incendies"])).parent_keys


def test_model_output_assets_depend_on_the_gold(defs):
    graph = defs.get_repository_def().asset_graph
    for name in ("tail_risk_country", "attribution_countryyear"):
        node = graph.get(AssetKey(["gold", name]))
        assert node.group_name == "model_out"
        assert AssetKey(["gold", "fact_fireevent"]) in node.parent_keys
