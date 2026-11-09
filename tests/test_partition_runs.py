"""Partitioned assets run a partition range as one run, as
`dagster asset materialize --partition-range` requires without a daemon, and
each receives the whole range. The step-5 commands of the materialisation
runbook are run here verbatim, in process, with the downloads replaced."""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
from click.testing import CliRunner
from dagster import AssetKey, BackfillPolicy

warnings.filterwarnings("ignore", message=".*currently in beta.*")

PARTITIONED = [
    ["bronze", "bdiff_raw"], ["bronze", "firms_hotspots_raw"], ["bronze", "cds_fwi_raw"],
    ["bronze", "cds_fwi_full_raw"], ["bronze", "era5_raw"], ["bronze", "edo_spei_raw"],
    ["bronze", "era5_daily_raw"], ["gold", "features_daily"],
]


@pytest.fixture(scope="module")
def defs(dbt_manifest):
    from src.orchestration.definitions import defs

    return defs


def test_every_partitioned_asset_runs_a_range_as_one_run(defs):
    graph = defs.get_repository_def().asset_graph
    partitioned = {k for k in graph.get_all_asset_keys() if graph.get(k).partitions_def is not None}
    assert partitioned == {AssetKey(k) for k in PARTITIONED}
    for key in partitioned:
        assert graph.get(key).backfill_policy == BackfillPolicy.single_run(), key.to_user_string()


def _step5_commands() -> list[tuple[str, str]]:
    text = Path("docs/runbooks/materialise-v2.md").read_text(encoding="utf-8")
    section = text.split("## 5.", 1)[1].split("\n## ", 1)[0]
    return re.findall(r'--select "([^"]+)" --partition-range (\S+)', section)


def test_runbook_step5_commands_run_in_process(tmp_path, monkeypatch, defs):
    from dagster._cli.asset import asset_materialize_command

    from src.ingest import cds_fwi, edo_spei, era5

    monkeypatch.setenv("DAGSTER_HOME", str(tmp_path))
    calls: dict[str, list] = {"fwi": [], "era5": [], "edo": []}
    monkeypatch.setattr(cds_fwi, "build", lambda download_only=False, years=None, agg_only=False: calls["fwi"].append(list(years)))
    def _chunk(start: int, end: int) -> Path:
        calls["era5"].append((start, end))
        return tmp_path / "x.nc"

    monkeypatch.setattr(era5, "download_chunk", _chunk)
    monkeypatch.setattr(edo_spei, "download_year", lambda y, r: calls["edo"].append(y))

    commands = _step5_commands()
    assert [sel for sel, _ in commands] == ["bronze/cds_fwi_raw", "bronze/era5_raw", "bronze/edo_spei_raw"]
    runner = CliRunner()
    for sel, rng in commands:
        res = runner.invoke(asset_materialize_command,
                            ["-m", "src.orchestration.definitions", "--select", sel, "--partition-range", rng])
        assert res.exit_code == 0, f"{sel} {rng}: {res.output[-1500:]}"
    assert calls["fwi"] == [[str(y) for y in range(2016, 2027)]]  # one run, the whole range
    assert calls["era5"] == [(1991, 2000), (2001, 2010), (2011, 2020), (2021, 2026)]
    assert calls["edo"] == [str(y) for y in range(2016, 2027)]


def test_edo_reads_no_index_when_every_year_is_cached(tmp_path, monkeypatch):
    from src import config
    from src.ingest import edo_spei

    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    (tmp_path / "edo").mkdir()
    for y in (2022, 2023):
        (tmp_path / "edo" / f"sem03_m_edo_{y}0101_{y}1221_t.nc").write_bytes(b"x")

    def _no_network(*args, **kwargs):
        raise AssertionError("no network call expected when every year is cached")

    monkeypatch.setattr(edo_spei.requests, "get", _no_network)
    remote = edo_spei._RemoteIndex()
    cached = edo_spei.download_year("2022", remote)
    assert cached is not None and cached.name.startswith("sem03_m_edo_2022")
    assert edo_spei.download_year("2023", remote) is not None
    with pytest.raises(AssertionError):  # a missing year does need the index
        edo_spei.download_year("2024", remote)
