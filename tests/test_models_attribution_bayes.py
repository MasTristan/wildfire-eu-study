"""Bayesian Q3: a known slope is recovered from simulated data, and the published
tables keep their schema. Draws are kept small so the module stays within CI
time; the sampling itself is exercised once (module-scoped fixture).
"""
import numpy as np
import pandas as pd
import pytest

from src.models import attribution_bayes as ab

TRUE_SLOPE = {"days_high_season": 0.8, "spei_spring": -0.4}
METRICS = list(TRUE_SLOPE)


def _panel(seed: int = 0, n_countries: int = 8, n_years: int = 10) -> pd.DataFrame:
    """Hierarchical truth: country slopes scatter around TRUE_SLOPE, residual sd 0.3."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_countries):
        level = rng.normal(8, 1.0)
        slopes = {m: s + rng.normal(0, 0.1) for m, s in TRUE_SLOPE.items()}
        for y in range(n_years):
            x = {m: rng.normal() for m in METRICS}
            log_area = level + sum(slopes[m] * x[m] for m in METRICS) + rng.normal(0, 0.3)
            rows.append({"country_key": f"C{c}", "year": 2010 + y, **x, "log_area": log_area,
                         "burnt_area_ha": float(np.expm1(log_area))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fit() -> ab.BayesResult:
    return ab.fit_attribution_bayes(_panel(), metrics=METRICS, draws=150, tune=250, chains=2, cores=1, seed=3)


def test_known_population_slopes_are_recovered(fit):
    pop = fit.coefficients[fit.coefficients["level"] == "population"].set_index("metric")
    panel = _panel()
    for m, slope in TRUE_SLOPE.items():
        truth = slope * panel[m].std(ddof=0)  # the model works on standardised metrics
        assert pop.loc[m, "hdi_low"] < truth < pop.loc[m, "hdi_high"]
        assert abs(pop.loc[m, "mean"] - truth) < 0.15
        # the frequentist coefficient, on standardised data, lands in the same place
        assert abs(pop.loc[m, "mean"] - pop.loc[m, "freq_coef"]) < 0.15
    assert fit.r2_weather_overall > 0.7


def test_published_tables_schema(fit):
    coefs, cy = fit.coefficients, fit.countryyear
    assert {"level", "country_key", "metric", "metric_name", "mean", "sd", "hdi_low", "hdi_high", "prob_positive",
            "area_multiplier", "r_hat", "ess_bulk", "freq_coef", "freq_se", "freq_ci_low", "freq_ci_high"} <= set(coefs.columns)
    assert set(coefs["level"]) == {"population", "between_country_sd", "country"}
    assert (coefs["hdi_low"] <= coefs["mean"]).all() and (coefs["mean"] <= coefs["hdi_high"]).all()
    assert (coefs[coefs["level"] == "country"]["country_key"].notna()).all()
    assert len(coefs[coefs["level"] == "country"]) == 8 * len(METRICS)
    assert {"country_key", "year", "log_area", "fitted", "weather_component", "weather_component_low",
            "weather_component_high", "residual", "adjusted_area_ha", "r2_weather", "r2_weather_low",
            "r2_weather_high"} <= set(cy.columns)
    assert len(cy) == 80 and (cy["adjusted_area_ha"] > 0).all()
    assert cy["r2_weather"].between(0, 1).all() and (cy["r2_weather_low"] <= cy["r2_weather_high"]).all()
    assert {"rhat_max", "ess_bulk_min", "divergences", "duration_s"} <= set(fit.diagnostics)


def test_prepare_drops_short_countries_and_incomplete_rows():
    panel = _panel(1, n_countries=3, n_years=6)
    short = panel[panel["country_key"] == "C0"].head(2).assign(country_key="SHORT")
    holes = panel.copy()
    holes.loc[0, "spei_spring"] = np.nan
    df, mets, dropped = ab.prepare(pd.concat([holes, short], ignore_index=True), METRICS, min_years=4)
    assert dropped == ["SHORT"] and "SHORT" not in set(df["country_key"])
    assert len(df) == 17 and mets == METRICS
    assert abs(df["z_days_high_season"].mean()) < 1e-9 and abs(df["z_days_high_season"].std(ddof=0) - 1) < 1e-9


def test_hdi_is_the_narrowest_interval():
    draws = np.concatenate([np.zeros(10), np.linspace(1, 2, 90)])[:, None]
    lo, hi = ab._hdi(draws, 0.5)
    assert lo[0] >= 1.0 and hi[0] - lo[0] <= 0.56  # skips the isolated zeros


def test_acceptance_rule_of_the_sampler_diagnostics():
    ok = {"rhat_max": 1.004, "ess_bulk_min": 800.0, "divergences": 0.0, "divergence_rate": 0.0}
    assert ab.check_diagnostics(ok)[0]
    assert not ab.check_diagnostics({**ok, "rhat_max": 1.02})[0]
    assert not ab.check_diagnostics({**ok, "divergences": 50.0, "divergence_rate": 0.0125})[0]
