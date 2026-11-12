"""Model-output assets (gold/model_out): tables the report and the model cards
read, each produced by one tracked run. Q2 tail risk and Q3 attribution depend
only on the gold; Q1 is the spike, whose frozen outputs are reshaped without
any recomputation. The Europe-wide ignition model on the feature store is
defined here but not materialised (reduced GO, ADR 0009).
"""

from datetime import date
from typing import Any

import pandas as pd
from dagster import AssetCheckResult, AssetCheckSpec, MaterializeResult, asset

from src import lake
from src.orchestration.assets import features, gold

GROUP = "model_out"


@asset(key_prefix="gold", group_name=GROUP, kinds={"python"}, deps=[gold.gold_star],
       description="Q2: GPD tail of declustered EFFIS fire episodes above 500 ha per EU27 country, return levels with year-block bootstrap intervals, threshold stability, goodness of fit")
def tail_risk_country() -> MaterializeResult:
    from src.models import extremes, tracking

    events = pd.read_parquet(lake.GOLD_STAR / "fact_fireevent.parquet")
    events = events[events["source_key"] == "EFFIS_RDA"]
    dim = pd.read_parquet(lake.GOLD_STAR / "dim_country.parquet")
    eu_members = frozenset(dim.loc[dim["is_eu"], "country_key"])
    params = {"threshold_ha": extremes.THRESHOLD_HA, "decluster_days": extremes.DECLUSTER_DAYS,
              "min_years": extremes.MIN_YEARS, "min_episodes": extremes.MIN_EPISODES, "n_boot": 500}
    exp = "q2_tail_risk"
    run_params = {**params, "bootstrap": "year_blocks", "perimeter": "EU27",
                  "stability_thresholds_ha": list(extremes.STABILITY_THRESHOLDS_HA), "stability_n_boot": 200,
                  "gof_test": "anderson_darling_parametric_bootstrap", "gof_n_sim": extremes.GOF_N_SIM,
                  "gof_alpha": extremes.GOF_ALPHA}
    with tracking.run(exp, f"country_{date.today():%Y%m%d}", run_params, tags={"published": "false"}) as active:
        full = extremes.tail_risk_table(events, group="country_key", eu_members=eu_members, **params)  # type: ignore[arg-type]
        table = full[full["in_scope"]].reset_index(drop=True)
        stability = extremes.threshold_stability(events, groups=list(table["country_key"]), group="country_key",
                                                 decluster_days=extremes.DECLUSTER_DAYS)
        gof, episodes = extremes.goodness_of_fit(events, table, group="country_key")
        run_id = active.info.run_id
        table["run_id"] = run_id
        tracking.log_table(table, "tail_risk_country")
        tracking.log_table(gof, "goodness_of_fit")
        tracking.log_table(full[["country_key", "years", "n_fires", "n_exceedances", "in_scope", "scope_reason"]], "perimeter")
        tracking.log_table(stability, "threshold_stability")
        tracking.log_metrics({"groups_seen": len(full), "groups_published": len(table),
                              "groups_shape_below_one": int(table["shape_below_one"].fillna(False).astype(bool).sum()),
                              "groups_gpd_rejected": int(gof["gpd_rejected"].sum()),
                              **{f"episodes_{k}": n for k, n in zip(full["country_key"], full["n_exceedances"], strict=True)}})
    lake.ensure_dirs()
    out = lake.GOLD_MODEL_OUT / "tail_risk_country.parquet"
    table.to_parquet(out, index=False)
    extremes.return_levels_long(table).assign(run_id=run_id).to_parquet(
        lake.GOLD_MODEL_OUT / "tail_risk_levels.parquet", index=False)
    stability.assign(run_id=run_id).to_parquet(lake.GOLD_MODEL_OUT / "tail_risk_threshold_stability.parquet", index=False)
    gof.assign(run_id=run_id).to_parquet(lake.GOLD_MODEL_OUT / "tail_risk_gof.parquet", index=False)
    episodes.assign(run_id=run_id).to_parquet(lake.GOLD_MODEL_OUT / "tail_risk_episodes.parquet", index=False)
    tracking.mark_published(exp, run_id)  # only once every table is written
    return MaterializeResult(metadata={"rows": int(len(table)), "run_id": run_id, "path": str(out)})


Q3_BAYES_SETTINGS: dict[str, Any] = {"draws": 1000, "tune": 1000, "chains": 4, "target_accept": 0.99}
ROBUSTNESS_FILE = "attribution_robustness.parquet"


def _q3_inputs() -> tuple[pd.DataFrame, list[str]]:
    """The country x year panel (before the shared exclusion rule) and the EU country keys."""
    from src.models import attribution

    fw = pd.read_parquet(lake.GOLD_STAR / "fact_fireweather_countrymonth.parquet")
    ba = pd.read_parquet(lake.GOLD_STAR / "fact_burntarea_countryyear.parquet")
    dim = pd.read_parquet(lake.GOLD_STAR / "dim_country.parquet")
    return attribution.build_panel(fw, ba), dim.loc[dim["is_eu"], "country_key"].tolist()


def _robust_column(coefficients: pd.DataFrame, model: str, metric_col: str = "metric") -> pd.Series:
    """`robust` per row from the published robustness table; missing until that asset has run."""
    path = lake.GOLD_MODEL_OUT / ROBUSTNESS_FILE
    if not path.exists():
        return pd.Series(pd.NA, index=coefficients.index, dtype="boolean")
    from src.models import attribution

    flags = attribution.robust_flags(pd.read_parquet(path), model)
    return coefficients[metric_col].map(flags).astype("boolean")


@asset(key_prefix="gold", group_name=GROUP, kinds={"python", "pymc"}, deps=[gold.gold_star],
       description="Q3 robustness: base, year effects, linear trend, EU only, Mediterranean only, rows under 100 ha removed "
                   "(plus an N(0, 2) prior for the Bayesian model); the `robust` rule of src/models/attribution.py")
def attribution_robustness() -> MaterializeResult:
    from src.models import attribution, tracking
    from src.models import attribution_bayes as bayes

    panel, eu = _q3_inputs()
    params = {"variants": attribution.VARIANTS + attribution.BAYES_EXTRA_VARIANTS, **Q3_BAYES_SETTINGS}
    with tracking.run("q3_attribution_robustness", f"variants_{date.today():%Y%m%d}", params) as active:
        freq = attribution.robustness_freq(panel, eu)
        bay, diags = bayes.robustness_bayes(panel, eu, **Q3_BAYES_SETTINGS)
        table = attribution.apply_robust_rule(pd.concat([freq, bay], ignore_index=True))
        table = table.merge(diags[["variant", "rhat_max", "divergences", "duration_s"]].assign(model="bayesian"),
                            on=["model", "variant"], how="left")
        table["run_id"] = active.info.run_id
        tracking.log_metrics({"rhat_max_variants": float(diags["rhat_max"].max()), "divergences_variants": float(diags["divergences"].sum())})
        tracking.log_table(table, "robustness")
    lake.ensure_dirs()
    table.to_parquet(lake.GOLD_MODEL_OUT / ROBUSTNESS_FILE, index=False)
    robust = table.drop_duplicates(["model", "metric"]).groupby("model")["robust"].sum().to_dict()
    return MaterializeResult(metadata={"rows": int(len(table)), "run_id": active.info.run_id,
                                       "robust_metrics": str(robust), "rhat_max_variants": float(diags["rhat_max"].max()),
                                       "divergences_variants": int(diags["divergences"].sum())})


@asset(key_prefix="gold", group_name=GROUP, kinds={"python"}, deps=[gold.gold_star, attribution_robustness],
       description="Q3 (frequentist): burnt area attributed to seasonal fire weather, country fixed effects, weather-adjusted area")
def attribution_countryyear() -> MaterializeResult:
    from src.models import attribution, tracking

    panel, _ = _q3_inputs()
    with tracking.run("q3_attribution", f"ols_fe_{date.today():%Y%m%d}", {"metrics": attribution.METRICS, "min_years": attribution.MIN_YEARS}) as active:
        res = attribution.fit_attribution(panel)
        loyo = attribution.leave_one_year_out_r2(panel)
        oos = attribution.oos_gain(loyo)
        tracking.log_metrics({"r2_within_weather": res.r2_within_weather, "r2_oos_vs_country_mean": oos, "n_obs": res.n_obs})
        tracking.log_table(res.coefficients, "coefficients")
        tracking.log_table(loyo, "leave_one_year_out")
        fitted = res.fitted.assign(run_id=active.info.run_id)
    lake.ensure_dirs()
    out = lake.GOLD_MODEL_OUT / "attribution_countryyear.parquet"
    fitted.to_parquet(out, index=False)
    coefs = attribution.report_coefficients(res.coefficients)
    coefs["robust"] = _robust_column(coefs, "frequentist")
    coefs.assign(run_id=active.info.run_id).to_parquet(lake.GOLD_MODEL_OUT / "attribution_coefficients.parquet", index=False)
    res.prepared.exclusions.assign(run_id=active.info.run_id).to_parquet(lake.GOLD_MODEL_OUT / "attribution_exclusions.parquet", index=False)
    pd.DataFrame([{"r2_within_weather": res.r2_within_weather, "r2_oos_vs_country_mean": oos,
                   "n_obs": res.n_obs, "n_countries": int(fitted["country_key"].nunique()),
                   "first_year": int(fitted["year"].min()), "last_year": int(fitted["year"].max()),
                   "run_id": active.info.run_id}]).to_parquet(lake.GOLD_MODEL_OUT / "attribution_summary.parquet", index=False)
    return MaterializeResult(metadata={"rows": int(len(fitted)), "r2_within_weather": res.r2_within_weather, "run_id": active.info.run_id})


@asset(key_prefix="gold", group_name=GROUP, kinds={"python", "pymc"}, deps=[gold.gold_star, attribution_robustness],
       check_specs=[AssetCheckSpec("sampler_diagnostics", asset=["gold", "attribution_bayes"],
                                   description="R-hat below 1.01 and divergences under 1 % of the draws")],
       description="Q3 (Bayesian): hierarchical PyMC model on the same panel, partially pooled slopes, posterior coefficients and per-country weather R2")
def attribution_bayes():
    from src.models import attribution, tracking
    from src.models import attribution_bayes as bayes

    panel, _ = _q3_inputs()
    min_years = attribution.MIN_YEARS
    params = {"metrics": attribution.METRICS, **Q3_BAYES_SETTINGS, "min_years": min_years, "priors": bayes.PRIORS}
    with tracking.run("q3_attribution_bayes", f"hier_{date.today():%Y%m%d}", params) as active:
        res = bayes.fit_attribution_bayes(panel, min_years=min_years, **Q3_BAYES_SETTINGS)
        diag = res.diagnostics
        tracking.log_metrics({"rhat_max": diag["rhat_max"], "ess_bulk_min": diag["ess_bulk_min"], "ess_tail_min": diag["ess_tail_min"],
                              "divergences": diag["divergences"], "duration_s": diag["duration_s"],
                              "r2_weather_overall": diag["r2_weather_overall"], "r2_weather_freq": diag["r2_weather_freq"],
                              "n_obs": diag["n_obs"], "n_countries": diag["n_countries"]})
        tracking.log_table(res.coefficients[res.coefficients["level"] != "country"], "coefficients_population")
    run_id = active.info.run_id
    lake.ensure_dirs()
    coefs = res.coefficients.assign(run_id=run_id)
    coefs["robust"] = _robust_column(coefs, "bayesian").where(coefs["level"] == "population", pd.NA)
    coefs.to_parquet(lake.GOLD_MODEL_OUT / "attribution_bayes_coefficients.parquet", index=False)
    res.countryyear.assign(run_id=run_id).to_parquet(lake.GOLD_MODEL_OUT / "attribution_bayes_countryyear.parquet", index=False)
    passed, detail = bayes.check_diagnostics(diag)
    yield MaterializeResult(metadata={"rows": int(len(res.countryyear)), "run_id": run_id, "rhat_max": diag["rhat_max"],
                                      "divergences": int(diag["divergences"]), "duration_s": round(diag["duration_s"], 1),
                                      "dropped_countries": ", ".join(res.dropped_countries)})
    yield AssetCheckResult(check_name="sampler_diagnostics", passed=passed, metadata=detail)


@asset(key_prefix="gold", group_name=GROUP, kinds={"python"}, deps=[features.features_daily, features.grid_cells],
       description="Q1: ignition occurrence, leave-one-year-out over the feature store with the pre-registered protocol")
def ignition_eval() -> MaterializeResult:
    from src.models import ignition, tracking

    files = sorted(lake.GOLD_FEATURES.glob("daily_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cells = pd.read_parquet(features.CELLS_PATH)
    df = df.merge(cells[["cell", "country_key"]], on="cell", how="left")
    years = sorted(int(y) for y in df["year"].unique())
    test_years = years[2:]  # every year with at least two earlier years to train on
    p = ignition.Protocol()
    with tracking.run("q1_ignition", f"loyo_{date.today():%Y%m%d}", {"test_years": test_years, "neg_rate": p.neg_rate, "block_deg": p.block_deg}) as active:
        res = ignition.run_protocol(df, test_years, p, out_dir=lake.GOLD_MODEL_OUT / "ignition")
        for name, g in res["metrics"].groupby("model"):
            tracking.log_metrics({f"{name}_pr_auc_mean": g["pr_auc"].mean(), f"{name}_brier_mean": g["brier"].mean()})
        tracking.log_metrics({"all_pass": float(res["criterion"]["all_pass"])})
        tracking.log_table(res["metrics"], "metrics")
        summary = ignition.monthly_country_summary(res["predictions"], cells).assign(run_id=active.info.run_id)
    summary.to_parquet(lake.GOLD_MODEL_OUT / "ignition_monthly_country.parquet", index=False)
    res["metrics"].assign(run_id=active.info.run_id).to_parquet(lake.GOLD_MODEL_OUT / "ignition_metrics.parquet", index=False)
    return MaterializeResult(metadata={"test_years": str(test_years), "all_pass": res["criterion"]["all_pass"], "run_id": active.info.run_id})


@asset(key_prefix="gold", group_name=GROUP, kinds={"python"},
       description="Q1 spike results for the report: metrics, criterion, calibration and feature gain, read from the frozen spike outputs")
def q1_spike_results() -> MaterializeResult:
    from src.models import spike_results

    lake.ensure_dirs()
    run_id = spike_results.log_spike_run()
    tables = spike_results.all_tables(run_id=run_id)
    for name, df in tables.items():
        df.to_parquet(lake.GOLD_MODEL_OUT / f"{name}.parquet", index=False)
    crit = tables["q1_criterion"]
    return MaterializeResult(metadata={**{name: int(len(df)) for name, df in tables.items()},
                                       "all_pass": bool(crit["all_pass"].iloc[0]), "run_id": run_id})
