"""Smoke tests: no network required. Validate config and parsing logic."""
from src import config
from src.ingest.effis import parse_rda_payload

# Real-shape sample of a /rda-stats response (2024), trimmed.
_SAMPLE = {
    "total": {"area": "total", "sumPerForest%": 0.33, "estSumTotal": 1969598.95, "sumRDATotal": 1871119.0},
    "EU": {"area": "EU", "sumPerForest%": 0.44, "estSumTotal": 441277.89, "sumRDATotal": 419214.0},
}


def test_paths_exist():
    assert config.DATA_RAW.is_dir()
    assert config.DATA_PROCESSED.is_dir()


def test_scope_is_configured():
    assert "FR" in config.EU_FIRE_DB_COUNTRIES
    assert set(config.MED_CORE).issubset(set(config.EU_FIRE_DB_COUNTRIES))
    assert config.BASE_YEAR_MIN < config.BASE_YEAR_MAX


def test_endpoints_are_https():
    for url in (config.EFFIS_RDA_STATS_URL, config.EFFIS_MAPS_HOST, config.FIRMS_AREA_CSV_URL):
        assert url.startswith("https://")


def test_parse_rda_payload():
    rows = parse_rda_payload(2024, _SAMPLE)
    assert len(rows) == 2
    eu = next(r for r in rows if r["area"] == "EU")
    assert eu["year"] == 2024
    assert eu["sum_rda_total_ha"] == 419214.0
    assert set(rows[0]) == {
        "year",
        "area",
        "sum_per_forest_pct",
        "est_sum_total_ha",
        "sum_rda_total_ha",
    }


def test_firms_window_url_starts_at_the_window_start():
    """Regression guard for the v1 defect: the FIRMS area API wants the first
    day of the window in the path, and v1 passed the last one."""
    from datetime import date

    from src.ingest.firms import window_url

    url = window_url("VIIRS_SNPP_SP", date(2023, 7, 1), 10, map_key="KEY")
    assert url.endswith("/VIIRS_SNPP_SP/-25,34,45,72/10/2023-07-01") or url.endswith("/10/2023-07-01")
    assert "KEY" in url
