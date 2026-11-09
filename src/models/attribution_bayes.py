"""Q3, Bayesian hierarchical version of the weather attribution (PyMC).

Same panel as the frequentist fit (`attribution.prepare_panel`, the single
exclusion rule), same standardisation of the metrics (centred on each country's
mean, divided by the within-country standard deviation), same outcome (log
burnt area, country x year).
What changes is the structure on the country effects and on the slopes:

    log_area[c,t] = alpha[c] + sum_k beta[c,k] * z[c,t,k] + eps,   eps ~ N(0, sigma)
    alpha[c]  ~ N(mu_alpha, sigma_alpha)                            (centred)
    beta[c,k] = mu_beta[k] + tau[k] * b_raw[c,k], b_raw ~ N(0, 1)   (non-centred)

Intercepts and slopes are partially pooled across countries: a country with
few years borrows strength from the others, a country with a clear signal
moves away from the common slope. `mu_beta` is the population-level slope the
frequentist model estimates with a single pooled coefficient, so the two are
compared metric by metric.

Priors, all weakly informative and on the log scale (the metrics are
standardised, so one unit is one within-country standard deviation of the
metric; the `prior_n02` robustness variant widens `mu_beta` to N(0, 2)):

* mu_alpha ~ N(8, 3): median country-year around 3 000 ha, 95 % of the prior
  mass between roughly 8 ha and 1.2 million ha, which covers every observed
  country without anchoring on the data.
* mu_beta[k] ~ N(0, 1): one standard deviation of a metric multiplies burnt
  area by at most a few units a priori (exp(2) is about 7.4) and the sign is
  left open.
* tau[k] ~ HalfNormal(0.5): countries may differ by about half a log unit per
  standard deviation, a large heterogeneity on this scale.
* sigma_alpha ~ HalfNormal(2) and sigma ~ HalfNormal(1.5): spread of the
  country levels and of the yearly residual, both wide against the observed
  log burnt area (standard deviation about 2.3).

The slopes are non-centred: they are weakly identified (a country has about
ten observations for four slopes), the regime where the centred form builds a
funnel between `tau` and the country slopes and diverges. The intercepts are
centred on purpose: each country has about ten observations on its own level,
which the data pin down, and the non-centred intercept then left `mu_alpha`
only weakly identified against the mean of the country offsets (R-hat 1.02 to
1.03 at 1000 draws on the real panel, against under 1.01 centred). The model is fitted on countries with at least
`min_years` observations; the others carry no within-country information.

The robustness variants add year effects (`gamma`, centred across years, scale
`sigma_year ~ HalfNormal(1)`) or a common linear trend (`delta ~ N(0, 1)` on
the standardised year).

Attribution is statistical, not causal. The Bayesian R2 follows Gelman et
al. (2019): per posterior draw and per country, the variance of the weather
component over the years divided by that variance plus the residual variance.

Backend note (ADR 0010): on Windows without a C++ compiler PyTensor is told
not to look for one and compiles to Numba instead; with a compiler (CI, WSL)
it keeps the default C backend. `configure_backend` must run before PyTensor
is imported, which `fit_attribution_bayes` guarantees.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models import attribution as at
from src.models.attribution import METRIC_NAMES, Z_95, PreparedPanel
from src.models.attribution import fit_attribution as fit_attribution_freq

HDI_PROB = 0.95
PRIORS = {
    "mu_alpha": (8.0, 3.0),
    "mu_beta": (0.0, 1.0),
    "tau": 0.5,
    "sigma_alpha": 2.0,
    "sigma": 1.5,
    "sigma_year": 1.0,
    "delta": 1.0,
}


def configure_backend() -> str:
    """Pick the PyTensor backend before PyTensor is imported; returns "numba" or "c".

    Without g++ PyTensor falls back to a pure Python mode that samples about
    two orders of magnitude slower, so it is told to skip the C search and the
    model is compiled with Numba. On Python 3.11 the `pip` import done at
    start-up by pip-system-certs disables setuptools' distutils shim, which
    PyTensor imports while looking for a compiler; the standard library
    distutils is then used instead.
    """
    if shutil.which("g++") or shutil.which("gcc"):
        return "c"
    if "pytensor" not in sys.modules:
        flags = os.environ.get("PYTENSOR_FLAGS", "")
        if "cxx=" not in flags:
            os.environ["PYTENSOR_FLAGS"] = (flags + ",cxx=").lstrip(",")
    if sys.version_info < (3, 12) and "pip" in sys.modules:
        os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    return "numba"


@dataclass
class BayesResult:
    coefficients: pd.DataFrame      # level, country_key, metric, posterior mean/sd/94 % interval, frequentist counterpart
    countryyear: pd.DataFrame       # per country x year: weather component, adjusted area, per-country Bayesian R2
    r2_weather_overall: float       # pooled within-country share of variance attributed to weather
    diagnostics: dict[str, float] = field(default_factory=dict)
    idata: Any = None
    dropped_countries: list[str] = field(default_factory=list)
    prepared: PreparedPanel | None = None
    freq: Any = None                # the OLS fit on exactly the same prepared panel


def build_model(df: pd.DataFrame, metrics: list[str], priors: dict[str, Any] | None = None,
                year_effects: bool = False, trend: bool = False):
    """The PyMC model on a prepared frame (`attribution.prepare_panel(...).frame`)."""
    import pymc as pm

    countries = pd.Index(sorted(df["country_key"].unique()), name="country")
    cidx = countries.get_indexer(df["country_key"])
    x = df[[f"z_{m}" for m in metrics]].to_numpy()
    pr = {**PRIORS, **(priors or {})}
    years = pd.Index(sorted(df["year"].unique()), name="year")
    yidx = years.get_indexer(df["year"])
    coords = {"country": countries.to_list(), "feature": metrics, "obs": np.arange(len(df)), "year": years.to_list()}
    with pm.Model(coords=coords) as model:
        mu_alpha = pm.Normal("mu_alpha", *pr["mu_alpha"])
        sigma_alpha = pm.HalfNormal("sigma_alpha", pr["sigma_alpha"])
        alpha = pm.Normal("alpha", mu_alpha, sigma_alpha, dims="country")

        mu_beta = pm.Normal("mu_beta", *pr["mu_beta"], dims="feature")
        tau = pm.HalfNormal("tau", pr["tau"], dims="feature")
        b_raw = pm.Normal("b_raw", 0.0, 1.0, dims=("country", "feature"))
        beta = pm.Deterministic("beta", mu_beta + tau * b_raw, dims=("country", "feature"))

        sigma = pm.HalfNormal("sigma", pr["sigma"])
        mean = alpha[cidx] + (beta[cidx] * x).sum(axis=-1)
        if year_effects:
            sigma_year = pm.HalfNormal("sigma_year", pr["sigma_year"])
            gamma = pm.Normal("gamma", 0.0, sigma_year, dims="year")
            gamma_c = pm.Deterministic("gamma_c", gamma - gamma.mean(), dims="year")  # centred: the level lives in alpha
            mean = mean + gamma_c[yidx]
        if trend:
            t = ((df["year"] - df["year"].mean()) / (df["year"].std(ddof=0) or 1.0)).to_numpy()
            delta = pm.Normal("delta", 0.0, pr["delta"])
            mean = mean + delta * t
        pm.Normal("log_area", mean, sigma, observed=df["log_area"].to_numpy(), dims="obs")
    return model


def sample(model, draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int | None = None,
           target_accept: float = 0.99, seed: int = 20260101):
    """NUTS with the backend picked by `configure_backend`."""
    import pymc as pm

    backend = configure_backend()
    kwargs: dict[str, Any] = {"compile_kwargs": {"mode": "NUMBA"}} if backend == "numba" else {}
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=cores or min(chains, os.cpu_count() or 1),
                         target_accept=target_accept, random_seed=seed, progressbar=False, **kwargs)


def diagnostics(idata, model_vars: list[str]) -> dict[str, float]:
    """Worst-case convergence numbers over the named variables, plus the divergence count."""
    import arviz as az

    summ = az.summary(idata, var_names=model_vars, kind="diagnostics", round_to="none")
    div = int(idata.sample_stats["diverging"].values.sum())
    n_draws = int(idata.sample_stats["diverging"].size)
    return {"rhat_max": float(summ["r_hat"].max()), "ess_bulk_min": float(summ["ess_bulk"].min()),
            "ess_tail_min": float(summ["ess_tail"].min()), "divergences": float(div),
            "divergence_rate": div / n_draws}


RHAT_MAX = 1.01
DIVERGENCE_RATE_MAX = 0.01


def check_diagnostics(diag: dict[str, float]) -> tuple[bool, dict[str, float | str]]:
    """The acceptance rule of ADR 0010: R-hat under 1.01 and divergences under 1 % of the draws."""
    ok = diag["rhat_max"] < RHAT_MAX and diag["divergence_rate"] < DIVERGENCE_RATE_MAX
    return ok, {"rhat_max": diag["rhat_max"], "ess_bulk_min": diag["ess_bulk_min"], "divergences": int(diag["divergences"]),
                "divergence_rate": diag["divergence_rate"], "rule": f"rhat < {RHAT_MAX} and divergence rate < {DIVERGENCE_RATE_MAX:.0%}"}


def _hdi(samples: np.ndarray, prob: float = HDI_PROB) -> tuple[np.ndarray, np.ndarray]:
    """Narrowest interval holding `prob` of the draws, along axis 0 (draws), per remaining column."""
    s = np.sort(samples.reshape(samples.shape[0], -1), axis=0)
    n = s.shape[0]
    k = int(np.ceil(prob * n))
    width = s[k - 1:] - s[: n - k + 1]
    i = width.argmin(axis=0)
    cols = np.arange(s.shape[1])
    return s[i, cols].reshape(samples.shape[1:]), s[i + k - 1, cols].reshape(samples.shape[1:])


def summarise(idata, df: pd.DataFrame, metrics: list[str], freq,
              scales: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Posterior tables: coefficients (population, country-level, heterogeneity) and country x year."""
    import arviz as az

    post = idata.posterior.stack(sample=("chain", "draw"))
    countries = list(post["country"].values)
    mu_beta = post["mu_beta"].transpose("sample", "feature").values       # (S, K)
    tau = post["tau"].transpose("sample", "feature").values               # (S, K)
    beta = post["beta"].transpose("sample", "country", "feature").values  # (S, C, K)
    alpha = post["alpha"].transpose("sample", "country").values          # (S, C)
    sigma = post["sigma"].values                                         # (S,)
    rhat = az.rhat(idata, var_names=["mu_beta", "tau", "beta"])
    ess = az.ess(idata, var_names=["mu_beta", "tau", "beta"], method="bulk")

    def block(level: str, country: str | None, k: int, draws: np.ndarray, rh: float, es: float) -> dict[str, Any]:
        lo, hi = _hdi(draws[:, None])
        return {"level": level, "country_key": country, "metric": metrics[k], "metric_name": METRIC_NAMES.get(metrics[k], metrics[k]),
                "mean": float(draws.mean()), "sd": float(draws.std(ddof=1)), "hdi_low": float(lo[0]), "hdi_high": float(hi[0]),
                "prob_positive": float((draws > 0).mean()), "area_multiplier": float(np.exp(draws.mean())),
                "r_hat": float(rh), "ess_bulk": float(es)}

    rows = []
    fc = freq.coefficients.set_index("metric")
    for k, m in enumerate(metrics):
        r = block("population", None, k, mu_beta[:, k], rhat["mu_beta"].sel(feature=m).item(), ess["mu_beta"].sel(feature=m).item())
        r.update(freq_coef=float(fc.loc[m, "coef"]), freq_se=float(fc.loc[m, "se"]),
                 freq_ci_low=float(fc.loc[m, "coef"] - Z_95 * fc.loc[m, "se"]), freq_ci_high=float(fc.loc[m, "coef"] + Z_95 * fc.loc[m, "se"]))
        rows.append(r)
        rows.append(block("between_country_sd", None, k, tau[:, k], rhat["tau"].sel(feature=m).item(), ess["tau"].sel(feature=m).item()))
        for c, country in enumerate(countries):
            rows.append(block("country", country, k, beta[:, c, k],
                              rhat["beta"].sel(country=country, feature=m).item(), ess["beta"].sel(country=country, feature=m).item()))
    coefficients = pd.DataFrame(rows)
    coefficients["hdi_prob"] = HDI_PROB
    if scales is not None:
        coefficients = coefficients.merge(scales[["metric", "sd_within", "sd_pooled", "unit_size", "unit_label"]], on="metric", how="left")
        k = coefficients["unit_size"] / coefficients["sd_within"]
        for col in ("mean", "hdi_low", "hdi_high", "sd"):
            coefficients[f"{col}_per_unit"] = coefficients[col] * k
        coefficients["multiplier_per_unit"] = np.exp(coefficients["mean_per_unit"])
        coefficients["freq_coef_per_unit"] = coefficients["freq_coef"] * k

    # weather component per draw: slope x deviation of the metric from the country's own mean
    zcols = [f"z_{m}" for m in metrics]
    dev = df[zcols] - df.groupby("country_key")[zcols].transform("mean")
    cidx = pd.Index(countries).get_indexer(df["country_key"])
    comp = np.einsum("sik,ik->si", beta[:, cidx, :], dev.to_numpy())    # (S, N)
    fitted = alpha[:, cidx] + np.einsum("sik,ik->si", beta[:, cidx, :], df[zcols].to_numpy())
    if "gamma_c" in post:
        yidx = pd.Index(post["year"].values).get_indexer(df["year"])
        fitted = fitted + post["gamma_c"].transpose("sample", "year").values[:, yidx]
    if "delta" in post:
        t = ((df["year"] - df["year"].mean()) / (df["year"].std(ddof=0) or 1.0)).to_numpy()
        fitted = fitted + post["delta"].values[:, None] * t[None, :]
    comp_mean = comp.mean(axis=0)
    lo, hi = _hdi(comp)

    # Bayesian R2, per country (within-country variance over the years) and pooled
    r2_country = np.empty((comp.shape[0], len(countries)))
    for c in range(len(countries)):
        v = comp[:, cidx == c].var(axis=1)
        r2_country[:, c] = v / (v + sigma**2)
    ss = (comp**2).sum(axis=1)
    r2_overall_draws = ss / (ss + len(df) * sigma**2)
    r2_lo, r2_hi = _hdi(r2_country)
    out = pd.DataFrame({
        "country_key": df["country_key"].to_numpy(), "year": df["year"].to_numpy(),
        "burnt_area_ha": df["burnt_area_ha"].to_numpy(), "log_area": df["log_area"].to_numpy(),
        "fitted": fitted.mean(axis=0), "weather_component": comp_mean,
        "weather_component_low": lo, "weather_component_high": hi,
        "residual": df["log_area"].to_numpy() - fitted.mean(axis=0),
    })
    out["adjusted_area_ha"] = np.expm1(out["log_area"] - out["weather_component"])
    per_country = pd.DataFrame({"country_key": countries, "r2_weather": r2_country.mean(axis=0),
                                "r2_weather_low": r2_lo, "r2_weather_high": r2_hi})
    out = out.merge(per_country, on="country_key", how="left")
    return coefficients, out, float(r2_overall_draws.mean())


def fit_attribution_bayes(panel: pd.DataFrame, metrics: list[str] | None = None, min_years: int = at.MIN_YEARS, draws: int = 1000,
                          tune: int = 1000, chains: int = 4, cores: int | None = None, target_accept: float = 0.99,
                          seed: int = 20260101, priors: dict[str, Any] | None = None, year_effects: bool = False,
                          trend: bool = False, **prepare_kwargs) -> BayesResult:
    """Fit the hierarchical model and summarise it against the OLS fit on exactly the same prepared panel."""
    configure_backend()
    prep = at.prepare_panel(panel, metrics, min_years=min_years, **prepare_kwargs)
    df, mets = prep.frame, prep.metrics
    freq = fit_attribution_freq(df, mets, min_years=min_years)
    model = build_model(df, mets, priors, year_effects, trend)
    t0 = time.perf_counter()
    idata = sample(model, draws, tune, chains, cores, target_accept, seed)
    duration = time.perf_counter() - t0
    names = ["mu_alpha", "sigma_alpha", "mu_beta", "tau", "sigma", "beta", "alpha"]
    names += [v for v in ("sigma_year", "gamma", "delta") if v in idata.posterior]
    diag = diagnostics(idata, names)
    diag["duration_s"] = duration
    coefficients, countryyear, r2_overall = summarise(idata, df, mets, freq, prep.scales)
    diag["r2_weather_overall"] = r2_overall
    diag["r2_weather_freq"] = freq.r2_within_weather
    diag["n_obs"] = float(len(df))
    diag["n_countries"] = float(df["country_key"].nunique())
    dropped = prep.exclusions.loc[~prep.exclusions["kept"], "country_key"].tolist()
    return BayesResult(coefficients, countryyear, r2_overall, diag, idata, dropped, prep, freq)


def robustness_bayes(panel: pd.DataFrame, eu_countries: list[str], variants: list[str] | None = None,
                     metrics: list[str] | None = None, **fit_kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Posterior population slopes under each variant (`VARIANTS` plus the N(0, 2) prior), long format
    without the `robust` flags, and the sampler diagnostics of every variant."""
    rows, diags = [], []
    for name in variants or [*at.VARIANTS, *at.BAYES_EXTRA_VARIANTS]:
        prep_kw, flags = at.variant_spec(name, eu_countries)
        priors = {"mu_beta": (0.0, 2.0)} if name == "prior_n02" else None
        res = fit_attribution_bayes(panel, metrics, priors=priors, **flags, **prep_kw, **fit_kwargs)
        pop = res.coefficients[res.coefficients["level"] == "population"].rename(columns={"mean": "coef"})
        rows.append(at.variant_rows("bayesian", name, pop, "hdi_low", "hdi_high", int(res.diagnostics["n_obs"]),
                                    int(res.diagnostics["n_countries"])))
        diags.append({"variant": name, **{k: float(v) for k, v in res.diagnostics.items()}})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diags)
