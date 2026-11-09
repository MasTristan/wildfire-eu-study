import numpy as np
import pandas as pd

from src.models import extremes


def test_gpd_fit_recovers_known_parameters():
    rng = np.random.default_rng(3)
    shape, scale = 0.35, 120.0
    # numpy's pareto(a) is a Lomax with unit scale: Lomax(a, m) == GPD(shape=1/a, scale=m/a)
    excess = rng.pareto(1 / shape, 20_000) * (scale / shape)
    est_shape, est_scale = extremes.fit_gpd(excess)
    assert abs(est_shape - shape) < 0.04
    assert abs(est_scale / scale - 1) < 0.08


def test_return_level_and_period_are_inverse():
    fit = extremes.GpdFit(threshold=200.0, n_exceedances=300, rate_per_year=30.0, shape=0.3, scale=150.0)
    for T in (2, 10, 50):
        x = fit.return_level(T)
        assert abs(fit.return_period(x) - T) < 1e-6
    assert fit.return_level(10) < fit.return_level(50)
    gumbel = extremes.GpdFit(200.0, 300, 30.0, 0.0, 150.0)
    assert abs(gumbel.return_period(gumbel.return_level(20)) - 20) < 1e-6


def test_tail_risk_table_reports_groups_it_cannot_estimate():
    rng = np.random.default_rng(5)
    big = pd.DataFrame({
        "country_key": "ES", "area_ha": 30 + rng.pareto(2.5, 3000) * 200,
        "firedate_key": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3650, 3000), unit="D"),
    })
    small = pd.DataFrame({
        "country_key": "LU", "area_ha": 30 + rng.pareto(2.5, 40) * 200,
        "firedate_key": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3650, 40), unit="D"),
    })
    table = extremes.tail_risk_table(pd.concat([big, small]), n_boot=50).set_index("country_key")
    assert table.loc["ES", "n_exceedances"] >= 30 and np.isfinite(table.loc["ES", "rl_10y"])
    assert table.loc["ES", "rl_2y"] < table.loc["ES", "rl_20y"]
    # about ten years observed: the horizon is about twenty years, so the 50-year level is null by design
    assert 19 < table.loc["ES", "max_return_period_years"] <= 20
    assert pd.isna(table.loc["ES", "rl_50y"]) and pd.isna(table.loc["ES", "rl_50y_p97.5"])
    assert table.loc["ES", "rl_10y_p2.5"] <= table.loc["ES", "rl_10y"] <= table.loc["ES", "rl_10y_p97.5"]
    assert "shape" not in table.columns or pd.isna(table.loc["LU", "shape"])
    assert table.loc["LU", "n_fires"] == 40
    assert np.isfinite(table.loc["ES", "empirical_rl_2y"])


def test_threshold_stability_table_shape():
    rng = np.random.default_rng(1)
    sizes = 30 + rng.pareto(3.0, 5000) * 100
    tab = extremes.threshold_stability(sizes)
    assert list(tab.columns) == ["quantile", "threshold", "n_exceedances", "shape", "scale", "modified_scale"]
    assert tab["threshold"].is_monotonic_increasing


def test_tail_risk_table_horizon_follows_the_observed_span():
    rng = np.random.default_rng(7)
    ev = pd.DataFrame({
        "country_key": "PT", "area_ha": 30 + rng.pareto(1.5, 2000) * 150,
        "firedate_key": pd.to_datetime("2000-01-01") + pd.to_timedelta(rng.integers(0, 365 * 30, 2000), unit="D"),
    })
    row = extremes.tail_risk_table(ev, n_boot=30).iloc[0]
    assert row["max_return_period_years"] > 50 and np.isfinite(row["rl_50y"])
    short = extremes.tail_risk_table(ev, n_boot=30, max_extrapolation=1.0).iloc[0]
    assert pd.isna(short["rl_50y"]) and np.isfinite(short["rl_20y"])
    assert pd.isna(short["return_period_10k_ha_years"]) or short["return_period_10k_ha_years"] <= short["max_return_period_years"]


def test_tail_risk_table_drops_a_partial_final_year():
    rng = np.random.default_rng(11)
    full = pd.DataFrame({
        "country_key": "ES", "area_ha": 30 + rng.pareto(1.5, 2000) * 150,
        "firedate_key": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3653, 2000), unit="D"),
    })
    partial = pd.DataFrame({  # January to July of the next year, as in the 2026 extraction
        "country_key": "ES", "area_ha": [5_000.0] * 50,
        "firedate_key": pd.date_range("2026-01-15", "2026-07-30", periods=50),
    })
    row = extremes.tail_risk_table(pd.concat([full, partial]), n_boot=20).iloc[0]
    assert row["last_year"] == 2025 and row["years"] == 10.0
    assert row["n_fires"] == int((full["area_ha"] >= 30).sum())
    # an explicit end in December keeps the year
    kept = extremes.tail_risk_table(pd.concat([full, partial]), n_boot=20, data_end=pd.Timestamp("2026-12-31")).iloc[0]
    assert kept["last_year"] == 2026 and kept["years"] == 11.0


def test_last_complete_year_of():
    assert extremes.last_complete_year_of(pd.Timestamp("2026-07-30")) == 2025
    assert extremes.last_complete_year_of(pd.Timestamp("2025-12-28")) == 2025


def test_tail_risk_period_ends_at_the_source_end_not_at_the_last_large_fire():
    rng = np.random.default_rng(13)
    es = pd.DataFrame({
        "country_key": "ES", "area_ha": 30 + rng.pareto(1.5, 1500) * 150,
        "firedate_key": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3653, 1500), unit="D"),
    })
    # SI: large fires stop in 2019, small ones (under the floor) start in 2016 and continue
    si_large = pd.DataFrame({"country_key": "SI", "area_ha": [100.0] * 5,
                             "firedate_key": pd.date_range("2017-03-01", "2019-08-01", periods=5)})
    si_small = pd.DataFrame({"country_key": "SI", "area_ha": [2.0] * 3,
                             "firedate_key": pd.to_datetime(["2016-04-01", "2021-05-01", "2024-07-01"])})
    table = extremes.tail_risk_table(pd.concat([es, si_large, si_small]), n_boot=10).set_index("country_key")
    assert table.loc["SI", "first_year"] == 2016 and table.loc["SI", "last_year"] == 2025
    assert table.loc["SI", "years"] == 10.0


def test_return_levels_long_has_one_row_per_period_and_keeps_the_horizon():
    rng = np.random.default_rng(17)
    ev = pd.DataFrame({
        "country_key": "ES", "area_ha": 30 + rng.pareto(1.5, 1500) * 150,
        "firedate_key": pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 3653, 1500), unit="D"),
    })
    table = extremes.tail_risk_table(ev, n_boot=10)
    long = extremes.return_levels_long(table)
    assert list(long["period_years"]) == list(extremes.RETURN_PERIODS)
    row10 = long.set_index("period_years").loc[10]
    assert row10["return_level_ha"] == table.iloc[0]["rl_10y"] and row10["within_horizon"]
    row50 = long.set_index("period_years").loc[50]
    assert not row50["within_horizon"] and pd.isna(row50["return_level_ha"])
