"""Bayesian Q3: a known slope is recovered from simulated data, the OLS and the Bayesian
fit receive exactly the same panel, and the published tables keep their schema. Draws are
kept small so the module stays within CI time; the sampling itself is exercised once
(module-scoped fixture), plus one short two-variant robustness run.
"""
import numpy as np
import pandas as pd
import pytest

from src.models import attribution as at
from src.models import attribution_bayes as ab

TRUE_SLOPE = {"days_high_season": 0.8, "spei_spring": -0.4}   # per within-country standard deviation
METRICS = list(TRUE_SLOPE)
SHORT = dict(draws=150, tune=250, chains=2, cores=1, seed=3)


def _panel(seed: int = 0, n_countries: int = 8, n_years: int = 10) -> pd.DataFrame:
    """Hierarchical truth: country slopes scatter around TRUE_SLOPE, residual sd 0.3. Countries sit at
    different metric levels, so the panel-wide and within-country standard deviations differ."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_countries):
        level = rng.normal(8, 1.0)
        slopes = {m: s + rng.normal(0, 0.1) for m, s in TRUE_SLOPE.items()}
        for y in range(n_years):
            x = {m: 20.0 * c + rng.normal() for m in METRICS}       # within sd 1, between-country spread large
            z = {m: x[m] - 20.0 * c for m in METRICS}
            log_area = level + sum(slopes[m] * z[m] for m in METRICS) + rng.normal(0, 0.3)
            rows.append({"country_key": f"C{c}", "year": 2010 + y, **x, "log_area": log_area,
                         "burnt_area_ha": float(np.expm1(log_area))})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fit() -> ab.BayesResult:
    return ab.fit_attribution_bayes(_panel(), metrics=METRICS, **SHORT)


def test_known_population_slopes_are_recovered(fit):
    pop = fit.coefficients[fit.coefficients["level"] == "population"].set_index("metric")
    panel = _panel()
    for m, slope in TRUE_SLOPE.items():
        sd_within = at.within_country_sd(panel, m)
        truth = slope * sd_within  # the model works on metrics standardised by the within-country sd
        assert pop.loc[m, "hdi_low"] < truth < pop.loc[m, "hdi_high"]
        assert abs(pop.loc[m, "mean"] - truth) < 0.15
        assert abs(pop.loc[m, "mean_per_unit"] - slope * pop.loc[m, "unit_size"]) < 0.15 * pop.loc[m, "unit_size"] / sd_within
        # the frequentist coefficient, on the same standardised data, lands in the same place
        assert abs(pop.loc[m, "mean"] - pop.loc[m, "freq_coef"]) < 0.15
    assert fit.r2_weather_overall > 0.7


def test_both_models_receive_exactly_the_same_panel(fit):
    panel = _panel()
    short = panel[panel["country_key"] == "C0"].head(2).assign(country_key="SHORT")
    holes = panel.copy()
    holes.loc[0, "spei_spring"] = np.nan
    raw = pd.concat([holes, short], ignore_index=True)
    prep = at.prepare_panel(raw, METRICS, min_years=4)
    freq = at.fit_attribution(raw, METRICS, min_years=4)
    assert freq.fitted[["country_key", "year"]].equals(prep.frame[["country_key", "year"]])
    assert len(prep.frame) == 79 and "SHORT" not in set(prep.frame["country_key"])
    # the Bayesian result carries the OLS fit it is compared with, on the same prepared rows
    assert fit.prepared is not None and fit.freq is not None
    keys = ["country_key", "year"]
    assert fit.countryyear[keys].equals(fit.prepared.frame[keys])
    assert fit.freq.fitted[keys].equals(fit.prepared.frame[keys])
    assert fit.prepared.frame.equals(at.prepare_panel(_panel(), METRICS, min_years=4).frame)


def test_published_tables_schema(fit):
    coefs, cy = fit.coefficients, fit.countryyear
    assert {"level", "country_key", "metric", "metric_name", "mean", "sd", "hdi_low", "hdi_high", "hdi_prob", "prob_positive",
            "area_multiplier", "r_hat", "ess_bulk", "freq_coef", "freq_se", "freq_ci_low", "freq_ci_high",
            "sd_within", "sd_pooled", "unit_size", "unit_label", "mean_per_unit", "hdi_low_per_unit", "hdi_high_per_unit",
            "multiplier_per_unit", "freq_coef_per_unit"} <= set(coefs.columns)
    assert (coefs["hdi_prob"] == 0.95).all() and ab.HDI_PROB == 0.95
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


def test_robustness_variants_have_the_published_schema():
    eu = [f"C{i}" for i in range(6)]
    table, diags = ab.robustness_bayes(_panel(), eu, variants=["year_effects"], metrics=METRICS, draws=100, tune=150, chains=2, cores=1, seed=3)
    assert set(table["variant"]) == {"year_effects"} and set(table["model"]) == {"bayesian"}
    assert len(table) == len(METRICS)
    assert {"coef", "ci_low", "ci_high", "coef_per_unit", "ci_low_per_unit", "ci_high_per_unit", "unit_label", "n_obs",
            "n_countries"} <= set(table.columns)
    assert (table["ci_low"] <= table["coef"]).all() and (table["coef"] <= table["ci_high"]).all()
    assert set(diags["variant"]) == {"year_effects"} and {"rhat_max", "divergences"} <= set(diags.columns)
    # the year-effects variant recovers the sign of the true slopes
    signs = table.set_index(["variant", "metric"])["coef_per_unit"]
    assert (signs.xs("days_high_season", level="metric") > 0).all() and (signs.xs("spei_spring", level="metric") < 0).all()


def test_prior_variant_widens_the_slope_prior():
    df = at.prepare_panel(_panel(), METRICS).frame
    base = ab.build_model(df, METRICS)
    wide = ab.build_model(df, METRICS, priors={"mu_beta": (0.0, 2.0)})
    import pymc as pm

    sd = [float(np.std(pm.draw(m["mu_beta"], draws=4000, random_seed=1))) for m in (base, wide)]
    assert 0.9 < sd[0] < 1.1 and 1.9 < sd[1] < 2.1



def test_hdi_is_the_narrowest_interval():
    draws = np.concatenate([np.zeros(10), np.linspace(1, 2, 90)])[:, None]
    lo, hi = ab._hdi(draws, 0.5)
    assert lo[0] >= 1.0 and hi[0] - lo[0] <= 0.56  # skips the isolated zeros


def test_acceptance_rule_of_the_sampler_diagnostics():
    ok = {"rhat_max": 1.004, "ess_bulk_min": 800.0, "divergences": 0.0, "divergence_rate": 0.0}
    assert ab.check_diagnostics(ok)[0]
    assert not ab.check_diagnostics({**ok, "rhat_max": 1.02})[0]
    assert not ab.check_diagnostics({**ok, "divergences": 50.0, "divergence_rate": 0.0125})[0]
