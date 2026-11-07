"""Q3, attribution of the year-to-year variability of burnt area to fire
weather: a country x year panel with country fixed effects, frequentist
version (statsmodels OLS, standard errors clustered by country). The
Bayesian hierarchical version (PyMC, run under WSL2) consumes the same
panel and must agree on the sign and magnitude of every coefficient.

Outputs, per country and year: the fitted log burnt area, the weather
component (sum of coefficient x deviation of each metric from the country's
own mean), the residual, and the weather-adjusted burnt area (what the year
would have burnt with that country's average weather). Attribution is
statistical, not causal; the report says so and shows the intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SEASON_MONTHS = (6, 7, 8, 9)
SPRING_MONTHS = (3, 4, 5)
METRICS = ["days_high_season", "spei_spring", "precip_anom_season", "temp_season"]


def build_panel(fireweather: pd.DataFrame, burntarea: pd.DataFrame, source_key: str = "EFFIS_RDA") -> pd.DataFrame:
    """country x year rows: burnt area of `source_key` and seasonal fire-weather metrics.

    Years are kept only when both sides exist and the fire season (June to
    September) is complete in the weather table; nothing is imputed.
    """
    fw = fireweather.copy()
    fw["year"] = pd.to_datetime(fw["month_key"]).dt.year
    fw["month"] = pd.to_datetime(fw["month_key"]).dt.month
    season = fw[fw["month"].isin(SEASON_MONTHS)]
    complete = season.groupby(["country_key", "year"])["month"].nunique() == len(SEASON_MONTHS)
    agg = season.groupby(["country_key", "year"]).agg(
        days_high_season=("days_fwi_high", "sum"),
        precip_anom_season=("precip_anomaly", "mean"),
        temp_season=("mean_temp", "mean"),
    )
    spring = fw[fw["month"].isin(SPRING_MONTHS)].groupby(["country_key", "year"])["spei"].mean().rename("spei_spring")
    weather = agg.join(spring, how="left").loc[complete[complete].index]
    ba = burntarea[burntarea["source_key"] == source_key].copy()
    ba["year"] = pd.to_datetime(ba["year_key"]).dt.year
    ba = ba.groupby(["country_key", "year"])["burnt_area_ha"].sum().rename("burnt_area_ha")
    panel = weather.join(ba, how="inner").reset_index()
    panel["log_area"] = np.log1p(panel["burnt_area_ha"])
    return panel.dropna(subset=["days_high_season"])


@dataclass
class AttributionResult:
    coefficients: pd.DataFrame      # metric, coef, se, p, in standardised units
    fitted: pd.DataFrame            # country_key, year, log_area, fitted, weather_component, residual, adjusted_area_ha
    r2_within_weather: float        # share of within-country variance explained by the weather metrics
    r2_within_total: float          # with country effects and trend together
    n_obs: int


def fit_attribution(panel: pd.DataFrame, metrics: list[str] | None = None, trend: bool = False) -> AttributionResult:
    metrics = [m for m in (metrics or METRICS) if m in panel.columns and panel[m].notna().any()]
    df = panel.dropna(subset=metrics).copy()
    # standardise each metric within the whole panel so coefficients compare
    stats = {m: (df[m].mean(), df[m].std(ddof=0) or 1.0) for m in metrics}
    for m in metrics:
        df[f"z_{m}"] = (df[m] - stats[m][0]) / stats[m][1]
    df["t"] = df["year"] - df["year"].min()
    rhs = " + ".join([f"z_{m}" for m in metrics] + (["t"] if trend else []))
    model = smf.ols(f"log_area ~ C(country_key) + {rhs}", data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["country_key"]})
    coefs = pd.DataFrame({
        "metric": metrics,
        "coef": [res.params[f"z_{m}"] for m in metrics],
        "se": [res.bse[f"z_{m}"] for m in metrics],
        "p_value": [res.pvalues[f"z_{m}"] for m in metrics],
    })
    # weather component: deviation of each metric from the country's own mean, times its coefficient
    comp = np.zeros(len(df))
    for m in metrics:
        dev = df[f"z_{m}"] - df.groupby("country_key")[f"z_{m}"].transform("mean")
        comp += res.params[f"z_{m}"] * dev.to_numpy()
    fitted = pd.DataFrame({
        "country_key": df["country_key"].to_numpy(),
        "year": df["year"].to_numpy(),
        "burnt_area_ha": df["burnt_area_ha"].to_numpy(),
        "log_area": df["log_area"].to_numpy(),
        "fitted": res.fittedvalues.to_numpy(),
        "weather_component": comp,
        "residual": res.resid.to_numpy(),
    })
    fitted["adjusted_area_ha"] = np.expm1(fitted["log_area"] - fitted["weather_component"])
    # within-country variance shares
    within = df["log_area"] - df.groupby("country_key")["log_area"].transform("mean")
    ss_within = float((within**2).sum())
    r2_weather = 1.0 - float(((within - comp) ** 2).sum()) / ss_within if ss_within > 0 else float("nan")
    r2_total = 1.0 - float((res.resid**2).sum()) / ss_within if ss_within > 0 else float("nan")
    return AttributionResult(coefs, fitted, r2_weather, r2_total, int(res.nobs))


def leave_one_year_out_r2(panel: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Out-of-sample check: fit on all years but one, predict the held-out year."""
    rows = []
    for held in sorted(panel["year"].unique()):
        train, test = panel[panel["year"] != held], panel[panel["year"] == held]
        if train["country_key"].nunique() < 2 or test.empty:
            continue
        res = fit_attribution(train, metrics)
        met = res.coefficients["metric"].tolist()
        # rebuild the prediction with the training fit: country effect + weather terms
        tr = train.dropna(subset=met)
        stats = {m: (tr[m].mean(), tr[m].std(ddof=0) or 1.0) for m in met}
        ce = res.fitted.groupby("country_key").apply(lambda g: (g["fitted"] - g["weather_component"]).mean(), include_groups=False)
        te = test.dropna(subset=met)
        te = te[te["country_key"].isin(ce.index)]
        if te.empty:
            continue
        pred = te["country_key"].map(ce).to_numpy()
        for m, c in zip(met, res.coefficients["coef"], strict=True):
            z_test = (te[m] - stats[m][0]) / stats[m][1]
            z_mean_country = tr.assign(z=(tr[m] - stats[m][0]) / stats[m][1]).groupby("country_key")["z"].mean()
            pred = pred + c * (z_test - te["country_key"].map(z_mean_country)).to_numpy()
        err = te["log_area"].to_numpy() - pred
        naive = te["log_area"].to_numpy() - te["country_key"].map(tr.groupby("country_key")["log_area"].mean()).to_numpy()
        rows.append({"held_year": int(held), "n": len(te), "rmse_model": float(np.sqrt(np.mean(err**2))),
                     "rmse_country_mean": float(np.sqrt(np.mean(naive**2)))})
    return pd.DataFrame(rows)
