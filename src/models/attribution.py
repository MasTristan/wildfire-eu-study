"""Q3, attribution of the year-to-year variability of burnt area to fire
weather: a country x year panel with country fixed effects, frequentist
version (statsmodels OLS, standard errors clustered by country). The
Bayesian hierarchical version (attribution_bayes.py, PyMC) consumes the same
panel, through the same `prepare_panel`, and must agree on the sign and
magnitude of every coefficient.

Outputs, per country and year: the fitted log burnt area, the weather
component (sum of coefficient x deviation of each metric from the country's
own mean), the residual, and the weather-adjusted burnt area (what the year
would have burnt with that country's average weather). Attribution is
statistical, not causal; the report says so and shows the intervals.

Units. Each metric is centred on its country's own mean and divided by its
within-country standard deviation (`within_country_sd`: the standard
deviation of the values once each country's mean is removed). A coefficient
therefore reads "one typical year-to-year swing of that metric inside a
country". The panel-wide standard deviation mixes differences between
countries with differences over time, which the country effects remove, and
overstated the temperature and danger-days multipliers (review Q3-1). The
effect per physical unit (`METRIC_UNITS`) is published next to it.

One panel. `prepare_panel` is the only place where rows and countries are
dropped, for both models and for every published table: complete rows only
(every metric and the log burnt area present), then countries with at least
`MIN_YEARS` complete years. The exclusions are listed by `exclusion_report`.

Robustness rule (written before the variants were run, and not edited since).
A metric is called "robust" if and only if both hold:

1. its 95 % interval in the base specification excludes zero (confidence
   interval for the OLS fit, highest-density interval for the Bayesian one);
2. its sign is the same in every variant of `VARIANTS` (base, year effects,
   linear trend, EU only, Mediterranean only, rows under 100 ha removed, and
   for the Bayesian model also an N(0, 2) prior on the slopes). A variant that
   could not be fitted counts as a failure.

A metric that is not robust is presented as not distinguishable from zero.
`apply_robust_rule` implements the rule, on coefficients per physical unit so
that variants with different panels compare in the same units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SEASON_MONTHS = (6, 7, 8, 9)
SPRING_MONTHS = (3, 4, 5)
METRICS = ["days_high_season", "spei_spring", "precip_anom_season", "temp_season"]
METRIC_NAMES = {
    "days_high_season": "High fire-danger days, June to September",
    "spei_spring": "Spring water balance (SPEI), March to May",
    "precip_anom_season": "Rainfall anomaly, June to September",
    "temp_season": "Mean temperature, June to September",
}

MIN_YEARS = 4
MIN_AREA_HA = 100.0
MEDITERRANEAN = ("PT", "ES", "FR", "IT", "GR", "HR", "CY", "MT", "SI")
Z_95 = 1.959963984540054

# one physical unit per metric: (size of the unit in the metric's own scale, label).
# precip_anomaly is a percentage of the 1991-2020 monthly normal, not millimetres.
METRIC_UNITS = {
    "days_high_season": (10.0, "10 jours de danger"),
    "spei_spring": (1.0, "1 unité de SPEI"),
    "precip_anom_season": (10.0, "10 points de % d'anomalie de pluie"),
    "temp_season": (1.0, "1 °C"),
}
VARIANTS = ["base", "year_effects", "linear_trend", "eu_only", "mediterranean_only", "min_area_100ha"]
BAYES_EXTRA_VARIANTS = ["prior_n02"]
REQUIRED_VARIANTS = {"frequentist": VARIANTS, "bayesian": VARIANTS + BAYES_EXTRA_VARIANTS}


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


def within_country_sd(df: pd.DataFrame, metric: str) -> float:
    """Standard deviation of `metric` once each country's own mean is removed (population form)."""
    dev = df[metric] - df.groupby("country_key")[metric].transform("mean")
    sd = float(np.sqrt((dev**2).mean()))
    return sd if sd > 0 else 1.0


@dataclass
class PreparedPanel:
    frame: pd.DataFrame        # complete rows, kept countries, z_<metric> = (x - country mean) / within-country sd
    metrics: list[str]
    scales: pd.DataFrame       # metric, sd_within, sd_pooled, unit_size, unit_label
    exclusions: pd.DataFrame   # one row per affected country, see `exclusion_report`


EXCLUSION_COLUMNS = ["country_key", "rows_in_panel", "rows_missing_metric", "missing_metrics", "years_kept", "kept", "reason"]


def exclusion_report(panel: pd.DataFrame, metrics: list[str], kept: pd.DataFrame, min_years: int) -> pd.DataFrame:
    """Per country that lost at least one row: which metric was missing and whether the country stayed."""
    rows = []
    need = [*metrics, "log_area"]
    kept_years = kept.groupby("country_key")["year"].nunique()
    for country, g in panel.groupby("country_key"):
        complete = g.dropna(subset=need)
        n_missing = int(len(g) - len(complete))
        years_kept = int(kept_years.get(country, 0))
        stayed = years_kept > 0
        if n_missing == 0 and stayed:
            continue
        missing = [m for m in need if g[m].isna().any()]
        reason = f"fewer than {min_years} complete years ({len(complete)})" if not stayed else "incomplete rows dropped"
        rows.append({"country_key": country, "rows_in_panel": int(len(g)), "rows_missing_metric": n_missing,
                     "missing_metrics": ", ".join(missing), "years_kept": years_kept, "kept": stayed, "reason": reason})
    return pd.DataFrame(rows, columns=EXCLUSION_COLUMNS)


def prepare_panel(panel: pd.DataFrame, metrics: list[str] | None = None, min_years: int = MIN_YEARS,
                  min_area_ha: float | None = None, countries: list[str] | tuple[str, ...] | None = None) -> PreparedPanel:
    """The single exclusion rule shared by the OLS and the Bayesian fit.

    Optional restrictions (`countries`, `min_area_ha`) come first and define a
    robustness variant; then complete rows only; then countries with at least
    `min_years` complete years. Metrics are standardised on the rows that remain.
    """
    metrics = [m for m in (metrics or METRICS) if m in panel.columns and panel[m].notna().any()]
    base = panel.copy()
    if countries is not None:
        base = base[base["country_key"].isin(list(countries))]
    if min_area_ha is not None:
        base = base[base["burnt_area_ha"] >= min_area_ha]
    df = base.dropna(subset=[*metrics, "log_area"])
    counts = df.groupby("country_key")["year"].transform("size")
    df = df[counts >= min_years].sort_values(["country_key", "year"]).reset_index(drop=True)
    scales = []
    for m in metrics:
        sd = within_country_sd(df, m)
        df[f"z_{m}"] = (df[m] - df.groupby("country_key")[m].transform("mean")) / sd
        size, label = METRIC_UNITS.get(m, (1.0, m))
        scales.append({"metric": m, "sd_within": sd, "sd_pooled": float(df[m].std(ddof=0)), "unit_size": size, "unit_label": label})
    return PreparedPanel(df, metrics, pd.DataFrame(scales), exclusion_report(base, metrics, df, min_years))


@dataclass
class AttributionResult:
    coefficients: pd.DataFrame      # metric, coef, se, p_value per within-country SD, scale and per-unit columns
    fitted: pd.DataFrame            # country_key, year, log_area, fitted, weather_component, residual, adjusted_area_ha
    r2_within_weather: float        # share of within-country variance explained by the weather metrics
    n_obs: int
    prepared: PreparedPanel = field(repr=False, default=None)  # type: ignore[assignment]


def fit_attribution(panel: pd.DataFrame, metrics: list[str] | None = None, trend: bool = False, year_effects: bool = False,
                    **prepare_kwargs) -> AttributionResult:
    """OLS with country effects on `prepare_panel(panel, ...)`; `trend` adds a linear year term, `year_effects` year dummies."""
    prep = prepare_panel(panel, metrics, **prepare_kwargs)
    metrics, df = prep.metrics, prep.frame.copy()
    df["t"] = df["year"] - df["year"].min()
    rhs = " + ".join([f"z_{m}" for m in metrics] + (["t"] if trend else []) + (["C(year)"] if year_effects else []))
    model = smf.ols(f"log_area ~ C(country_key) + {rhs}", data=df)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["country_key"]})
    coefs = pd.DataFrame({
        "metric": metrics,
        "coef": [res.params[f"z_{m}"] for m in metrics],
        "se": [res.bse[f"z_{m}"] for m in metrics],
        "p_value": [res.pvalues[f"z_{m}"] for m in metrics],
    })
    coefs = coefs.merge(prep.scales, on="metric", how="left")
    coefs["coef_per_unit"] = coefs["coef"] * coefs["unit_size"] / coefs["sd_within"]
    coefs["se_per_unit"] = coefs["se"] * coefs["unit_size"] / coefs["sd_within"]
    # weather component: the z columns are already deviations from the country's own mean
    comp = np.zeros(len(df))
    for m in metrics:
        comp += res.params[f"z_{m}"] * df[f"z_{m}"].to_numpy()
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
    # Within-country variance share of the weather metrics alone. The pooled R2 with the
    # country effects (about 0.87) is not a weather share and is never published.
    within = df["log_area"] - df.groupby("country_key")["log_area"].transform("mean")
    ss_within = float((within**2).sum())
    r2_weather = 1.0 - float(((within - comp) ** 2).sum()) / ss_within if ss_within > 0 else float("nan")
    return AttributionResult(coefs, fitted, r2_weather, int(res.nobs), prep)


def leave_one_year_out_r2(panel: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Out-of-sample check: fit on all years but one, predict the held-out year."""
    rows = []
    for held in sorted(panel["year"].unique()):
        train, test = panel[panel["year"] != held], panel[panel["year"] == held]
        if train["country_key"].nunique() < 2 or test.empty:
            continue
        res = fit_attribution(train, metrics)
        met = res.coefficients["metric"].tolist()
        tr = res.prepared.frame
        sd = res.prepared.scales.set_index("metric")["sd_within"]
        # prediction with the training fit: country effect + weather terms
        ce = (res.fitted["fitted"] - res.fitted["weather_component"]).groupby(res.fitted["country_key"]).mean()
        te = test.dropna(subset=[*met, "log_area"])
        te = te[te["country_key"].isin(ce.index)]
        if te.empty:
            continue
        pred = te["country_key"].map(ce).to_numpy()
        for m, c in zip(met, res.coefficients["coef"], strict=True):
            z_test = (te[m] - te["country_key"].map(tr.groupby("country_key")[m].mean())) / sd[m]
            pred = pred + c * z_test.to_numpy()
        err = te["log_area"].to_numpy() - pred
        naive = te["log_area"].to_numpy() - te["country_key"].map(tr.groupby("country_key")["log_area"].mean()).to_numpy()
        rows.append({"held_year": int(held), "n": len(te), "rmse_model": float(np.sqrt(np.mean(err**2))),
                     "rmse_country_mean": float(np.sqrt(np.mean(naive**2)))})
    return pd.DataFrame(rows)


def oos_gain(loyo: pd.DataFrame) -> float:
    """Out-of-sample share of the country-mean error removed by the model, 1 - MSE_model / MSE_country_mean,
    weighted by the held-out rows. The meaningful companion of the in-sample within-country R2."""
    w = loyo["n"].to_numpy()
    mse_model = float((loyo["rmse_model"].to_numpy() ** 2 * w).sum())
    mse_naive = float((loyo["rmse_country_mean"].to_numpy() ** 2 * w).sum())
    return 1.0 - mse_model / mse_naive if mse_naive > 0 else float("nan")


def report_coefficients(coefficients: pd.DataFrame, z: float = Z_95) -> pd.DataFrame:
    """Coefficients as the report reads them: a readable name, a 95 % interval, and the
    multiplier on burnt area for one within-country standard deviation of the metric
    (the outcome is log burnt area, so exp(coef) is a ratio). With the scale columns
    present, the same quantities per physical unit are added."""
    df = coefficients.copy()
    df["metric_name"] = df["metric"].map(METRIC_NAMES).fillna(df["metric"])
    df["ci_low"] = df["coef"] - z * df["se"]
    df["ci_high"] = df["coef"] + z * df["se"]
    df["area_multiplier"] = np.exp(df["coef"])
    df["significant"] = df["p_value"] < 0.05
    if "unit_size" in df.columns:
        k = df["unit_size"] / df["sd_within"]
        df["ci_low_per_unit"] = df["ci_low"] * k
        df["ci_high_per_unit"] = df["ci_high"] * k
        df["multiplier_per_unit"] = np.exp(df["coef_per_unit"])
    return df


def variant_spec(name: str, eu_countries: list[str] | None = None) -> tuple[dict, dict]:
    """(prepare_panel kwargs, model flags) of a robustness variant. `prior_n02` only changes the Bayesian prior."""
    prep: dict = {}
    flags: dict = {}
    if name in ("base", "prior_n02"):
        pass
    elif name == "year_effects":
        flags["year_effects"] = True
    elif name == "linear_trend":
        flags["trend"] = True
    elif name == "eu_only":
        prep["countries"] = list(eu_countries or [])
    elif name == "mediterranean_only":
        prep["countries"] = list(MEDITERRANEAN)
    elif name == "min_area_100ha":
        prep["min_area_ha"] = MIN_AREA_HA
    else:
        raise ValueError(f"unknown variant {name!r}")
    return prep, flags


def variant_rows(model: str, variant: str, coefs: pd.DataFrame, lo_col: str, hi_col: str, n_obs: int, n_countries: int) -> pd.DataFrame:
    """Long rows (one per metric) of a fitted variant, in within-SD and per-unit terms.
    `coefs` needs metric, coef, `lo_col`, `hi_col`, unit_size, sd_within, unit_label."""
    k = (coefs["unit_size"] / coefs["sd_within"]).to_numpy()
    return pd.DataFrame({
        "model": model, "variant": variant, "metric": coefs["metric"].to_numpy(),
        "coef": coefs["coef"].to_numpy(), "ci_low": coefs[lo_col].to_numpy(), "ci_high": coefs[hi_col].to_numpy(),
        "coef_per_unit": coefs["coef"].to_numpy() * k, "ci_low_per_unit": coefs[lo_col].to_numpy() * k,
        "ci_high_per_unit": coefs[hi_col].to_numpy() * k, "unit_label": coefs["unit_label"].to_numpy(),
        "n_obs": n_obs, "n_countries": n_countries,
    })


def robustness_freq(panel: pd.DataFrame, eu_countries: list[str], metrics: list[str] | None = None) -> pd.DataFrame:
    """OLS coefficients under each variant of `VARIANTS`, long format, without the `robust` flags."""
    out = []
    for name in VARIANTS:
        prep, flags = variant_spec(name, eu_countries)
        res = fit_attribution(panel, metrics, **flags, **prep)
        c = report_coefficients(res.coefficients)
        out.append(variant_rows("frequentist", name, c, "ci_low", "ci_high", res.n_obs, int(res.fitted["country_key"].nunique())))
    return pd.concat(out, ignore_index=True)


def apply_robust_rule(table: pd.DataFrame, required: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Add `base_ci_excludes_zero`, `same_sign_all_variants` and `robust` per (model, metric); see the module docstring."""
    required = required or REQUIRED_VARIANTS
    out = table.copy()
    flags: dict[tuple[str, str], tuple[bool, bool, bool]] = {}
    for key, g in out.groupby(["model", "metric"]):
        model, metric = str(key[0]), str(key[1])
        base = g[g["variant"] == "base"]
        excl = (not base.empty) and bool((base["ci_low_per_unit"].iloc[0] > 0) or (base["ci_high_per_unit"].iloc[0] < 0))
        signs = np.sign(g["coef_per_unit"].to_numpy())
        same = (set(required.get(model, VARIANTS)) <= set(g["variant"]) and not np.isnan(signs).any()
                and bool((signs != 0).all()) and len(set(signs.tolist())) == 1)
        flags[(model, metric)] = (excl, same, excl and same)
    keys = list(zip(out["model"].astype(str), out["metric"].astype(str), strict=True))
    out["base_ci_excludes_zero"] = [flags[k][0] for k in keys]
    out["same_sign_all_variants"] = [flags[k][1] for k in keys]
    out["robust"] = [flags[k][2] for k in keys]
    return out


def robust_flags(table: pd.DataFrame, model: str) -> dict[str, bool]:
    """`robust` per metric for one model, from a table returned by `apply_robust_rule`."""
    t = table[table["model"] == model].drop_duplicates("metric")
    return {str(m): bool(r) for m, r in zip(t["metric"], t["robust"], strict=True)}
