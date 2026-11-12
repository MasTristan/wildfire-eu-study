import numpy as np
import pandas as pd

from src.models import attribution as at


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    countries = ["ES", "PT", "GR", "IT", "FR", "HR"]
    years = list(range(2016, 2026))
    rows = []
    effects = {c: rng.normal(9, 1) for c in countries}
    for c in countries:
        for y in years:
            for m in range(1, 13):
                days_high = max(0, rng.normal(12 if m in (6, 7, 8, 9) else 2, 3))
                rows.append({"country_key": c, "month_key": pd.Timestamp(year=y, month=m, day=1),
                             "days_fwi_high": days_high, "precip_anomaly": rng.normal(0, 20),
                             "mean_temp": rng.normal(20 if m in (6, 7, 8, 9) else 8, 2), "spei": rng.normal(0, 1)})
    fw = pd.DataFrame(rows)
    panel_truth = at.build_panel(fw, pd.DataFrame(columns=["country_key", "year_key", "source_key", "burnt_area_ha"]))
    # burnt area generated from the true metrics: strong effect of days_high, weak of spei
    fw["year"] = fw["month_key"].dt.year
    season = fw[fw["month_key"].dt.month.isin(at.SEASON_MONTHS)].groupby(["country_key", "year"])["days_fwi_high"].sum()
    spring = fw[fw["month_key"].dt.month.isin(at.SPRING_MONTHS)].groupby(["country_key", "year"])["spei"].mean()
    ba_rows = []
    for (c, y), dh in season.items():
        log_area = effects[c] + 0.06 * (dh - season.mean()) - 0.3 * (spring.loc[(c, y)]) + rng.normal(0, 0.3)
        ba_rows.append({"country_key": c, "year_key": pd.Timestamp(year=y, month=1, day=1), "source_key": "EFFIS_RDA",
                        "burnt_area_ha": float(np.expm1(log_area))})
    return fw, pd.DataFrame(ba_rows), panel_truth


def test_panel_and_fit_recover_the_weather_signal():
    fw, ba, _ = _synthetic()
    panel = at.build_panel(fw, ba)
    assert len(panel) == 60 and {"days_high_season", "spei_spring", "log_area"} <= set(panel.columns)
    res = at.fit_attribution(panel)
    coefs = res.coefficients.set_index("metric")["coef"]
    assert coefs["days_high_season"] > 0 and coefs["spei_spring"] < 0
    assert abs(coefs["days_high_season"]) > abs(coefs["temp_season"])
    assert 0.3 < res.r2_within_weather <= 1.0
    assert len(res.fitted) == 60 and (res.fitted["adjusted_area_ha"] > 0).all()
    loyo = at.leave_one_year_out_r2(panel)
    assert len(loyo) == 10
    assert loyo["rmse_model"].mean() < loyo["rmse_country_mean"].mean()


def test_incomplete_season_years_are_dropped_not_imputed():
    fw, ba, _ = _synthetic(1)
    fw = fw[~((fw["country_key"] == "ES") & (fw["month_key"] == "2025-09-01"))]  # September missing
    panel = at.build_panel(fw, ba)
    assert not ((panel["country_key"] == "ES") & (panel["year"] == 2025)).any()


def test_report_coefficients_adds_interval_and_multiplier():
    c = pd.DataFrame({"metric": ["days_high_season"], "coef": [1.12], "se": [0.25], "p_value": [0.0001]})
    r = at.report_coefficients(c).iloc[0]
    assert r["metric_name"].startswith("High fire-danger days")
    assert abs(r["ci_low"] - (1.12 - at.Z_95 * 0.25)) < 1e-12 and r["ci_high"] > r["coef"]
    assert abs(r["area_multiplier"] - np.exp(1.12)) < 1e-12 and r["significant"]


def _levels_panel() -> pd.DataFrame:
    """Countries at wildly different levels, each swinging by the same +-s pattern around its own mean."""
    rows = []
    swing = np.array([-1.5, -0.5, 0.5, 1.5])
    for country, (level, s) in {"AA": (0.0, 2.0), "BB": (100.0, 2.0), "CC": (1000.0, 2.0)}.items():
        for i, y in enumerate(range(2016, 2020)):
            rows.append({"country_key": country, "year": y, "temp_season": level + s * swing[i], "log_area": 5.0 + 0.1 * i,
                         "burnt_area_ha": 150.0 + i})
    return pd.DataFrame(rows)


def test_within_country_sd_ignores_differences_between_countries():
    df = _levels_panel()
    expected = 2.0 * np.sqrt(np.mean(np.array([-1.5, -0.5, 0.5, 1.5]) ** 2))
    assert abs(at.within_country_sd(df, "temp_season") - expected) < 1e-12
    assert df["temp_season"].std(ddof=0) > 300          # the pooled spread is dominated by the levels
    prep = at.prepare_panel(df, ["temp_season"])
    z = prep.frame["z_temp_season"]
    assert abs(z.std(ddof=0) - 1.0) < 1e-12              # one unit = one within-country standard deviation
    assert prep.frame.groupby("country_key")["z_temp_season"].mean().abs().max() < 1e-12
    row = prep.scales.iloc[0]
    assert abs(row["sd_within"] - expected) < 1e-12 and row["sd_pooled"] > 300


def test_coefficient_per_unit_is_the_physical_slope():
    rng = np.random.default_rng(5)
    rows = []
    for c in range(6):
        for y in range(2012, 2026):
            days = rng.normal(60 + 40 * c, 8)                 # large differences between countries, 8 days within
            rows.append({"country_key": f"C{c}", "year": y, "days_high_season": days,
                         "log_area": 6 + c + 0.03 * days + rng.normal(0, 0.2)})
    panel = pd.DataFrame(rows).assign(burnt_area_ha=lambda d: np.expm1(d["log_area"]))
    res = at.fit_attribution(panel, ["days_high_season"])
    row = res.coefficients.iloc[0]
    assert abs(row["coef_per_unit"] - 0.3) < 0.05              # 0.03 per day, 10 days
    assert abs(row["coef"] - 0.03 * row["sd_within"]) < 0.03 * 1.0


def test_single_exclusion_rule_and_report():
    panel = _levels_panel()
    short = panel[panel["country_key"] == "AA"].head(2).assign(country_key="SHORT")
    holes = panel.copy()
    holes.loc[0, "temp_season"] = np.nan
    prep = at.prepare_panel(pd.concat([holes, short], ignore_index=True), ["temp_season"], min_years=3)
    assert set(prep.frame["country_key"]) == {"AA", "BB", "CC"}      # SHORT has 2 rows only
    ex = prep.exclusions.set_index("country_key")
    assert not ex.loc["SHORT", "kept"] and "fewer than 3" in ex.loc["SHORT", "reason"]
    assert ex.loc["AA", "kept"] and ex.loc["AA", "rows_missing_metric"] == 1 and ex.loc["AA", "missing_metrics"] == "temp_season"
    # the OLS fit works on exactly the rows of the shared preparation
    res = at.fit_attribution(pd.concat([holes, short], ignore_index=True), ["temp_season"], min_years=3)
    assert res.fitted[["country_key", "year"]].equals(prep.frame[["country_key", "year"]])


def _rows(model, variant, metric, lo, hi):
    mid = (lo + hi) / 2
    return {"model": model, "variant": variant, "metric": metric, "coef_per_unit": mid, "ci_low_per_unit": lo, "ci_high_per_unit": hi}


def test_robust_rule():
    freq_variants = at.VARIANTS
    rows = []
    for v in freq_variants:
        rows.append(_rows("frequentist", v, "stable", 0.2, 0.6))                                   # excludes 0, same sign everywhere
        rows.append(_rows("frequentist", v, "wide_base", -0.1 if v == "base" else 0.1, 0.5))        # base interval contains 0
        rows.append(_rows("frequentist", v, "flips", 0.2 if v == "base" else -0.6, 0.6 if v == "base" else -0.2))  # sign changes
    out = at.apply_robust_rule(pd.DataFrame(rows))
    flags = at.robust_flags(out, "frequentist")
    assert flags == {"stable": True, "wide_base": False, "flips": False}
    g = out[out["metric"] == "wide_base"].iloc[0]
    assert not g["base_ci_excludes_zero"] and g["same_sign_all_variants"]
    # a missing variant counts as a failure
    missing = pd.DataFrame([r for r in rows if r["metric"] == "stable" and r["variant"] != "eu_only"])
    assert not at.robust_flags(at.apply_robust_rule(missing), "frequentist")["stable"]
    # the Bayesian model also needs its prior variant
    bay = pd.DataFrame([_rows("bayesian", v, "m", 0.2, 0.6) for v in at.VARIANTS])
    assert not at.robust_flags(at.apply_robust_rule(bay), "bayesian")["m"]
    bay = pd.DataFrame([_rows("bayesian", v, "m", 0.2, 0.6) for v in [*at.VARIANTS, "prior_n02"]])
    assert at.robust_flags(at.apply_robust_rule(bay), "bayesian")["m"]


def test_schemas_robustness_and_no_pooled_r2():
    fw, ba, _ = _synthetic()
    panel = at.build_panel(fw, ba)
    res = at.fit_attribution(panel)
    assert not hasattr(res, "r2_within_total")
    cols = {"metric", "coef", "se", "p_value", "sd_within", "sd_pooled", "unit_size", "unit_label", "coef_per_unit", "se_per_unit"}
    assert cols <= set(res.coefficients.columns)
    rep = at.report_coefficients(res.coefficients)
    assert {"ci_low", "ci_high", "area_multiplier", "significant", "ci_low_per_unit", "ci_high_per_unit", "multiplier_per_unit"} <= set(rep.columns)
    tab = at.apply_robust_rule(at.robustness_freq(panel, ["ES", "PT", "GR", "IT", "FR"]))
    assert set(tab["variant"]) == set(at.VARIANTS) and set(tab["model"]) == {"frequentist"}
    assert {"coef", "ci_low", "ci_high", "coef_per_unit", "unit_label", "n_obs", "n_countries", "base_ci_excludes_zero",
            "same_sign_all_variants", "robust"} <= set(tab.columns)
    assert len(tab) == len(at.VARIANTS) * len(at.METRICS)
    assert tab.loc[tab["variant"] == "mediterranean_only", "n_countries"].iloc[0] == 6
    flags = at.robust_flags(tab, "frequentist")
    assert flags["days_high_season"] and flags["spei_spring"]       # strong simulated effects


def test_oos_gain_weights_by_rows():
    loyo = pd.DataFrame({"n": [10, 30], "rmse_model": [1.0, 1.0], "rmse_country_mean": [2.0, 1.0]})
    assert abs(at.oos_gain(loyo) - (1 - 40 / (40 + 30))) < 1e-12
