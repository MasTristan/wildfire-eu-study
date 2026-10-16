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
