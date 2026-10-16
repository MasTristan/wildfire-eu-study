"""Model-output assets (gold/model_out): tables the report reads, each
produced by one tracked run. Q1 (ignition) joins once the spike verdict
fixes its scope; Q2 tail risk and Q3 attribution depend only on the gold.
"""

from datetime import date

import pandas as pd
from dagster import MaterializeResult, asset

from src import lake
from src.orchestration.assets import gold

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
    res.coefficients.assign(run_id=active.info.run_id).to_parquet(lake.GOLD_MODEL_OUT / "attribution_coefficients.parquet", index=False)
    return MaterializeResult(metadata={"rows": int(len(fitted)), "r2_within_weather": res.r2_within_weather, "run_id": active.info.run_id})
