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
    assert table.loc["ES", "rl_2y"] < table.loc["ES", "rl_50y"]
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
