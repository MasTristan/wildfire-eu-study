import numpy as np
import pandas as pd

from src.models import extremes

EU_MEMBERS = frozenset({"ES", "PT", "SI", "FR", "IT", "GR"})  # what dim_country.is_eu provides in production


def _events(country: str, sizes, dates) -> pd.DataFrame:
    return pd.DataFrame({"country_key": country, "area_ha": np.asarray(sizes, dtype=float),
                         "firedate_key": pd.to_datetime(dates)})


def _simulated(country: str, n_years: int, rate: float, shape: float, scale: float, threshold: float = 500.0,
               seed: int = 0, start_year: int = 2000) -> pd.DataFrame:
    """Episodes of a known GPD, one at least 10 days from the next, plus a small fire each year so
    that every year counts as observed. Dates are spread over days 1 to 300 of each year."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_years):
        n = rng.poisson(rate)
        days = rng.choice(np.arange(1, 300, 10), size=min(n, 29), replace=False)
        for d in days:
            excess = rng.pareto(1 / shape) * (scale / shape) if shape > 0 else rng.exponential(scale)
            rows.append((threshold + excess, pd.Timestamp(f"{start_year + k}-01-01") + pd.Timedelta(days=int(d))))
        rows.append((10.0, pd.Timestamp(f"{start_year + k}-12-15")))
    sizes, dates = zip(*rows, strict=True)
    return _events(country, sizes, dates)


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


def test_fires_one_or_two_days_apart_form_a_single_episode():
    ev = _events("PT", [600, 900, 700, 800, 650, 520], [
        "2017-10-15", "2017-10-16", "2017-10-18",   # 1 day then 2 days apart: one episode, max 900
        "2017-10-21",                                # 3 days after the previous: still the same chain
        "2017-10-25",                                # 4 days after: a new episode
        "2018-07-01",                                # a year later: another one
    ])
    ep = extremes.decluster_episodes(ev, threshold_ha=500.0, gap_days=3)
    assert list(ep["area_ha"]) == [900.0, 650.0, 520.0]
    assert list(ep["n_fires"]) == [4, 1, 1]
    assert ep.loc[0, "date"] == pd.Timestamp("2017-10-16") and ep.loc[0, "year"] == 2017
    assert ep.loc[0, "start_date"] == pd.Timestamp("2017-10-15") and ep.loc[0, "end_date"] == pd.Timestamp("2017-10-21")
    tight = extremes.decluster_episodes(ev, threshold_ha=500.0, gap_days=1)
    assert len(tight) == 5  # 1 day still merges the first two; the 2- and 3-day gaps split


def test_declustering_stays_inside_a_country_and_ignores_fires_under_the_threshold():
    ev = pd.concat([_events("PT", [600, 700], ["2017-06-01", "2017-06-02"]),
                    _events("ES", [800], ["2017-06-02"]),
                    _events("ES", [100, 400], ["2017-06-03", "2017-06-04"])])  # under 500: not exceedances
    ep = extremes.decluster_episodes(ev, threshold_ha=500.0, gap_days=3)
    assert sorted(zip(ep["country_key"], ep["area_ha"], strict=True)) == [("ES", 800.0), ("PT", 700.0)]


def test_gpd_on_episodes_recovers_a_known_shape_and_rate():
    ev = _simulated("ES", n_years=400, rate=6.0, shape=0.4, scale=300.0, seed=1, start_year=1700)
    table = extremes.tail_risk_table(ev, n_boot=40, min_years=8).iloc[0]
    assert table["in_scope"] and table["years"] == 400.0
    assert abs(table["shape"] - 0.4) < 0.07
    assert abs(table["scale"] / 300.0 - 1) < 0.15
    assert abs(table["rate_per_year"] - 6.0) < 0.4
    assert table["shape_p2.5"] < table["shape"] < table["shape_p97.5"]
    assert table["shape_below_one"]
    assert table["rl_10y_p2.5"] <= table["rl_10y"] <= table["rl_10y_p97.5"]


def test_block_bootstrap_propagates_the_uncertainty_on_the_rate():
    # same sizes, but episodes concentrated in a few years: the rate varies a lot across resampled years
    rng = np.random.default_rng(4)
    rows = []
    for k in range(12):
        n = 30 if k in (2, 9) else 1
        for d in range(n):
            rows.append((500 + rng.exponential(300), pd.Timestamp(f"{2010 + k}-03-01") + pd.Timedelta(days=5 * d)))
        rows.append((10.0, pd.Timestamp(f"{2010 + k}-12-20")))
    sizes, dates = zip(*rows, strict=True)
    ev = _events("ES", sizes, dates)
    ep = extremes.decluster_episodes(ev, threshold_ha=500.0)
    boot = extremes.bootstrap_fit(ep, 500.0, np.arange(2010, 2022), n_boot=200, seed=0).set_index("quantity")
    # fixed-rate resampling of the episodes would give a narrow band on the 10-year level; blocks do not
    iid = extremes.tail_risk_table(ev, n_boot=10).iloc[0]
    assert boot.loc["rl_10y", "p97.5"] / boot.loc["rl_10y", "p2.5"] > 1.1
    assert iid["rate_per_year"] == len(ep) / 12


def test_perimeter_and_publication_rule():
    # the three countries below are fitted on the same simulated tail; only the perimeter differs
    big = _simulated("ES", n_years=12, rate=8.0, shape=0.3, scale=300.0, seed=2, start_year=2010)
    short = _simulated("PT", n_years=6, rate=8.0, shape=0.3, scale=300.0, seed=3, start_year=2010)
    short = short.assign(firedate_key=short["firedate_key"] + pd.DateOffset(years=6))  # first record in 2016
    outside = _simulated("DZ", n_years=12, rate=8.0, shape=0.3, scale=300.0, seed=4, start_year=2010)
    few = _simulated("SI", n_years=12, rate=1.0, shape=0.3, scale=300.0, seed=5, start_year=2010)
    ev = pd.concat([big, short, outside, few])
    ev = pd.concat([ev, _events("PT", [10.0], ["2010-01-01"])])  # PT has a record since 2010: make the span 16
    t = extremes.tail_risk_table(ev, n_boot=20, data_end=pd.Timestamp("2021-12-31"),
                                 eu_members=EU_MEMBERS).set_index("country_key")
    assert t.loc["ES", "scope_reason"] == "in_scope" and t.loc["ES", "in_scope"]
    assert t.loc["DZ", "scope_reason"] == "not_eu27" and pd.isna(t.loc["DZ", "shape"])
    assert t.loc["SI", "scope_reason"] == "episodes_below_min" and pd.isna(t.loc["SI", "shape"])
    assert t.loc["DZ", "n_exceedances"] > 30  # it is the perimeter that excludes it, not the count
    # a period of fewer than MIN_YEARS years
    young = _simulated("IT", n_years=6, rate=10.0, shape=0.3, scale=300.0, seed=6, start_year=2016)
    ty = extremes.tail_risk_table(young, n_boot=10, data_end=pd.Timestamp("2021-12-31")).iloc[0]
    assert ty["scope_reason"] == "years_below_min" and not ty["in_scope"]


def test_levels_are_withheld_when_the_shape_is_one_or_more_and_beyond_the_horizon():
    heavy = _simulated("PT", n_years=12, rate=10.0, shape=1.4, scale=200.0, seed=7, start_year=2010)
    light = _simulated("ES", n_years=12, rate=10.0, shape=0.2, scale=200.0, seed=8, start_year=2010)
    t = extremes.tail_risk_table(pd.concat([heavy, light]), n_boot=20, data_end=pd.Timestamp("2021-12-31"))
    t = t.set_index("country_key")
    assert t.loc["PT", "shape"] >= 1 and not t.loc["PT", "shape_below_one"]
    assert t.loc["PT", [f"rl_{T}y" for T in extremes.RETURN_PERIODS]].isna().all()
    assert pd.isna(t.loc["PT", "return_period_10k_ha_years"])
    assert t.loc["ES", "shape_below_one"] and np.isfinite(t.loc["ES", "rl_10y"])
    assert pd.isna(t.loc["ES", "rl_50y"])  # 12 observed years: horizon 24 years
    long = extremes.return_levels_long(t.reset_index()).set_index(["country_key", "period_years"])
    assert not long.loc[("PT", 10), "published"] and not long.loc[("PT", 10), "shape_below_one"]
    assert long.loc[("ES", 10), "published"]
    assert not long.loc[("ES", 50), "published"] and not long.loc[("ES", 50), "within_horizon"]


def test_table_schema_is_stable_even_with_nothing_in_scope():
    ev = _events("DZ", [600.0, 700.0], ["2015-05-01", "2016-05-01"])
    t = extremes.tail_risk_table(ev, n_boot=5, data_end=pd.Timestamp("2016-12-31"), eu_members=EU_MEMBERS)
    assert list(t.columns) == extremes.tail_risk_columns()
    assert list(t["scope_reason"]) == ["not_eu27"]
    full = extremes.tail_risk_table(_simulated("ES", 12, 8.0, 0.3, 300.0, seed=9), n_boot=5)
    assert list(full.columns) == extremes.tail_risk_columns()
    assert str(full["shape_below_one"].dtype) == "boolean" and full["in_scope"].dtype == bool
    for col in ("n_fires", "n_exceedances", "first_year", "last_year"):
        assert pd.api.types.is_integer_dtype(full[col])
    long = extremes.return_levels_long(full)
    assert list(long.columns) == ["country_key", "period_years", "return_level_ha", "return_level_p2_5",
                                  "return_level_p97_5", "empirical_level_ha", "within_horizon", "shape_below_one",
                                  "published"]


def test_threshold_stability_schema_and_monotone_thresholds():
    ev = _simulated("ES", n_years=40, rate=15.0, shape=0.3, scale=300.0, threshold=200.0, seed=10)
    tab = extremes.threshold_stability(ev, n_boot=20)
    assert list(tab.columns) == ["country_key", "threshold_ha", "n_fires", "n_exceedances", "rate_per_year", "shape",
                                 "shape_p2.5", "shape_p97.5", "scale", "modified_scale"]
    assert list(tab["threshold_ha"]) == list(extremes.STABILITY_THRESHOLDS_HA)
    assert tab["n_exceedances"].is_monotonic_decreasing
    # the simulated tail is a GPD from 200 ha up: the shape is flat from 200 to 500 ha
    assert tab["shape"].iloc[:3].max() - tab["shape"].iloc[:3].min() < 0.15
    sparse = extremes.threshold_stability(ev, thresholds=(50_000.0,), n_boot=5)
    assert sparse["shape"].isna().all() and sparse["n_exceedances"].iloc[0] < 20


def test_partial_final_year_is_excluded():
    full = _simulated("ES", n_years=10, rate=8.0, shape=0.3, scale=300.0, seed=11, start_year=2016)
    partial = _events("ES", [5_000.0] * 5, pd.date_range("2026-01-15", "2026-07-30", periods=5))
    row = extremes.tail_risk_table(pd.concat([full, partial]), n_boot=10).iloc[0]
    assert row["last_year"] == 2025 and row["years"] == 10.0
    kept = extremes.tail_risk_table(pd.concat([full, partial]), n_boot=10, data_end=pd.Timestamp("2026-12-31")).iloc[0]
    assert kept["last_year"] == 2026 and kept["years"] == 11.0


def test_last_complete_year_of():
    assert extremes.last_complete_year_of(pd.Timestamp("2026-07-30")) == 2025
    assert extremes.last_complete_year_of(pd.Timestamp("2025-12-28")) == 2025


def test_period_ends_at_the_source_end_not_at_the_last_large_fire():
    es = _simulated("ES", n_years=10, rate=8.0, shape=0.3, scale=300.0, seed=12, start_year=2016)
    si = _events("SI", [100.0, 2.0, 2.0], ["2017-03-01", "2016-04-01", "2024-07-01"])
    table = extremes.tail_risk_table(pd.concat([es, si]), n_boot=10).set_index("country_key")
    assert table.loc["SI", "first_year"] == 2016 and table.loc["SI", "last_year"] == 2025
    assert table.loc["SI", "years"] == 10.0 and table.loc["SI", "n_fires"] == 0


def test_mark_published_leaves_a_single_published_run(tmp_path, monkeypatch):
    import mlflow

    from src.models import tracking

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "art"))
    ids = []
    for name in ("a", "b"):
        with tracking.run("q2_test", name, {"x": 1}, tags={"published": "false"}) as active:
            ids.append(active.info.run_id)
        tracking.mark_published("q2_test", ids[-1])
    client = mlflow.MlflowClient()
    assert client.get_run(ids[0]).data.tags["published"] == "false"
    assert client.get_run(ids[1]).data.tags["published"] == "true"
