"""Model-output assets (gold/model_out): tables the report reads, each
produced by one tracked run. Q1 (ignition) joins once the spike verdict
fixes its scope; Q2 tail risk and Q3 attribution depend only on the gold.
"""

from datetime import date

import pandas as pd
from dagster import AssetCheckResult, AssetCheckSpec, MaterializeResult, asset

from src import lake
from src.orchestration.assets import features, gold

GROUP = "model_out"


@asset(key_prefix="gold", group_name=GROUP, kinds={"python"}, deps=[gold.gold_star],
       description="Q2: GPD tail of EFFIS fire sizes per country, return levels with bootstrap intervals")
def tail_risk_country() -> MaterializeResult:
    from src.models import extremes, tracking

    events = pd.read_parquet(lake.GOLD_STAR / "fact_fireevent.parquet")
    events = events[events["source_key"] == "EFFIS_RDA"]
    params = {"floor_ha": 30.0, "threshold_quantile": 0.90, "min_exceedances": 30, "n_boot": 500}
    with tracking.run("q2_tail_risk", f"country_{date.today():%Y%m%d}", params) as active:
        table = extremes.tail_risk_table(events, group="country_key", **params)  # type: ignore[arg-type]
        table["run_id"] = active.info.run_id
        tracking.log_table(table, "tail_risk_country")
        tracking.log_metrics({"groups": len(table), "groups_estimated": int(table.get("shape", pd.Series(dtype=float)).notna().sum())})
    lake.ensure_dirs()
    out = lake.GOLD_MODEL_OUT / "tail_risk_country.parquet"
    table.to_parquet(out, index=False)
    extremes.return_levels_long(table).assign(run_id=active.info.run_id).to_parquet(
        lake.GOLD_MODEL_OUT / "tail_risk_levels.parquet", index=False)
    return MaterializeResult(metadata={"rows": int(len(table)), "run_id": active.info.run_id, "path": str(out)})


@asset(key_prefix="gold", group_name=GROUP, kinds={"python"}, deps=[gold.gold_star],
       description="Q3 (frequentist): burnt area attributed to seasonal fire weather, country fixed effects, weather-adjusted area")
def attribution_countryyear() -> MaterializeResult:
    from src.models import attribution, tracking

    fw = pd.read_parquet(lake.GOLD_STAR / "fact_fireweather_countrymonth.parquet")
    ba = pd.read_parquet(lake.GOLD_STAR / "fact_burntarea_countryyear.parquet")
    panel = attribution.build_panel(fw, ba)
    with tracking.run("q3_attribution", f"ols_fe_{date.today():%Y%m%d}", {"metrics": attribution.METRICS}) as active:
        res = attribution.fit_attribution(panel)
        loyo = attribution.leave_one_year_out_r2(panel)
        tracking.log_metrics({"r2_within_weather": res.r2_within_weather, "r2_within_total": res.r2_within_total, "n_obs": res.n_obs})
        tracking.log_table(res.coefficients, "coefficients")
        tracking.log_table(loyo, "leave_one_year_out")
        fitted = res.fitted.assign(run_id=active.info.run_id)
    lake.ensure_dirs()
    out = lake.GOLD_MODEL_OUT / "attribution_countryyear.parquet"
    fitted.to_parquet(out, index=False)
    attribution.report_coefficients(res.coefficients).assign(run_id=active.info.run_id).to_parquet(
        lake.GOLD_MODEL_OUT / "attribution_coefficients.parquet", index=False)
    pd.DataFrame([{"r2_within_weather": res.r2_within_weather, "r2_within_total": res.r2_within_total,
                   "n_obs": res.n_obs, "n_countries": int(fitted["country_key"].nunique()),
                   "first_year": int(fitted["year"].min()), "last_year": int(fitted["year"].max()),
                   "run_id": active.info.run_id}]).to_parquet(lake.GOLD_MODEL_OUT / "attribution_summary.parquet", index=False)
    return MaterializeResult(metadata={"rows": int(len(fitted)), "r2_within_weather": res.r2_within_weather, "run_id": active.info.run_id})


@asset(key_prefix="gold", group_name=GROUP, kinds={"python", "pymc"}, deps=[gold.gold_star],
       check_specs=[AssetCheckSpec("sampler_diagnostics", asset=["gold", "attribution_bayes"],
                                   description="R-hat below 1.01 and divergences under 1 % of the draws")],
       description="Q3 (Bayesian): hierarchical PyMC model on the same panel, partially pooled slopes, posterior coefficients and per-country weather R2")
def attribution_bayes():
    from src.models import attribution, tracking
    from src.models import attribution_bayes as bayes

    fw = pd.read_parquet(lake.GOLD_STAR / "fact_fireweather_countrymonth.parquet")
    ba = pd.read_parquet(lake.GOLD_STAR / "fact_burntarea_countryyear.parquet")
    panel = attribution.build_panel(fw, ba)
    draws, tune, chains, target_accept, min_years = 1000, 1000, 4, 0.99, 4
    params = {"metrics": attribution.METRICS, "draws": draws, "tune": tune, "chains": chains, "target_accept": target_accept,
              "min_years": min_years, "priors": bayes.PRIORS}
    with tracking.run("q3_attribution_bayes", f"hier_{date.today():%Y%m%d}", params) as active:
        res = bayes.fit_attribution_bayes(panel, min_years=min_years, draws=draws, tune=tune, chains=chains,
                                          target_accept=target_accept)
        diag = res.diagnostics
        tracking.log_metrics({"rhat_max": diag["rhat_max"], "ess_bulk_min": diag["ess_bulk_min"], "ess_tail_min": diag["ess_tail_min"],
                              "divergences": diag["divergences"], "duration_s": diag["duration_s"],
                              "r2_weather_overall": diag["r2_weather_overall"], "r2_weather_freq": diag["r2_weather_freq"],
                              "n_obs": diag["n_obs"], "n_countries": diag["n_countries"]})
        tracking.log_table(res.coefficients[res.coefficients["level"] != "country"], "coefficients_population")
    run_id = active.info.run_id
    lake.ensure_dirs()
    res.coefficients.assign(run_id=run_id).to_parquet(lake.GOLD_MODEL_OUT / "attribution_bayes_coefficients.parquet", index=False)
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
