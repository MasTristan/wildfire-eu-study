"""Q1 spike results reshaped for the report.

The spike (notebooks/spike_q1) is closed: its outputs are small, committed
CSV and JSON files under notebooks/spike_q1/out, frozen with the verdict.
This module turns them into the tables the Power BI validation page reads,
without recomputing anything: every number comes from those files, so the
page and VERDICT.md cannot disagree.

Tables (label (a) EFFIS, the primary label of the protocol):
- q1_metrics: one row per test year and model, with the lift over the base
  rate (PR-AUC divided by the share of positive cell-days);
- q1_criterion: the pre-registered criterion per test year, the observed
  gain, its block-bootstrap interval, the threshold and the outcome;
- q1_reliability: calibration bins per test year and model;
- q1_importance: LightGBM gain of the full model's features, as a share.

Every table carries a `run_id`: the MLflow run `q1_spike` that logs the spike's
output files, the criterion file and the criterion metrics (`log_spike_run`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config

SPIKE_OUT: Path = config.ROOT / "notebooks" / "spike_q1" / "out"
LABEL = "a"

# display order and names, from the weakest reference to the full model
MODELS = {
    "danger_class": (1, "Official danger class"),
    "logit_fwi": (2, "Logistic, fire-weather index"),
    "lgbm_fwi": (3, "LightGBM, fire-weather index"),
    "lgbm_fwi_space": (4, "LightGBM, index + place and season"),
    "lgbm_full": (5, "LightGBM, full model"),
}
# criterion frozen in PROTOCOL.md before training
THRESHOLDS = {"brier_rel_improvement": 0.10, "pr_auc_gain": 0.05}
FEATURES = {
    "laga_7": ("Ignitions in the cell, past 7 days", "Place and season"),
    "laga_30": ("Ignitions in the cell, past 30 days", "Place and season"),
    "clim": ("Ignition climatology of the cell", "Place and season"),
    "doy": ("Day of year", "Place and season"),
    "dc": ("Drought Code", "Fire-weather index"),
    "ffmc": ("Fine Fuel Moisture Code", "Fire-weather index"),
    "dmc": ("Duff Moisture Code", "Fire-weather index"),
    "isi": ("Initial Spread Index", "Fire-weather index"),
    "bui": ("Build-up Index", "Fire-weather index"),
    "fwi": ("Fire Weather Index", "Fire-weather index"),
    "dsr": ("Daily Severity Rating", "Fire-weather index"),
    "tp_7d": ("Precipitation, past 7 days", "Weather and drought"),
    "tp_mm": ("Precipitation, day", "Weather and drought"),
    "spei3": ("Drought index SPEI-3", "Weather and drought"),
    "rh": ("Relative humidity, mean", "Weather and drought"),
    "rh_min": ("Relative humidity, minimum", "Weather and drought"),
    "t2m_c": ("Temperature, mean", "Weather and drought"),
    "t2m_max": ("Temperature, maximum", "Weather and drought"),
    "wind": ("Wind speed, maximum", "Weather and drought"),
    "wind_mean": ("Wind speed, mean", "Weather and drought"),
    "cvh": ("High vegetation cover", "Land"),
    "cvl": ("Low vegetation cover", "Land"),
    "tvh": ("High vegetation type", "Land"),
    "tvl": ("Low vegetation type", "Land"),
    "lsm": ("Land share of the cell", "Land"),
    "elev_m": ("Elevation", "Land"),
}
QUANTITY_NAMES = {"brier_rel_improvement": "Brier score, relative gain", "pr_auc_gain": "PR-AUC, absolute gain"}
EXPERIMENT = "q1_spike"
LOGGED_SUFFIXES = {".csv", ".json", ".png"}


def _with_model_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["model"].isin(list(MODELS))].copy()
    df["model_order"] = df["model"].map(lambda m: MODELS[m][0])
    df["model_name"] = df["model"].map(lambda m: MODELS[m][1])
    return df


def metrics(out: Path = SPIKE_OUT) -> pd.DataFrame:
    df = pd.read_csv(out / f"metrics_{LABEL}.csv")
    df = _with_model_names(df[df["label"] == LABEL])
    df["lift"] = df["pr_auc"] / df["base_rate"]
    cols = ["year", "model", "model_order", "model_name", "n", "positives", "base_rate", "pr_auc", "lift", "brier", "logloss"]
    return df[cols].sort_values(["year", "model_order"]).reset_index(drop=True)


def criterion(out: Path = SPIKE_OUT) -> pd.DataFrame:
    crit = json.loads((out / f"criterion_{LABEL}.json").read_text())
    boot = pd.read_csv(out / f"bootstrap_{LABEL}.csv").set_index(["year", "quantity"])
    rows = []
    for year in sorted(k for k in crit if k.isdigit()):
        for quantity, threshold in THRESHOLDS.items():
            value = float(crit[year][quantity])
            b = boot.loc[(int(year), quantity)]
            rows.append({
                "year": int(year), "quantity": quantity, "quantity_name": QUANTITY_NAMES[quantity],
                "value": value, "p2_5": float(b["p2.5"]), "p97_5": float(b["p97.5"]), "threshold": threshold,
                "meets_threshold": value >= threshold, "year_passes": bool(crit[year]["passes"]),
            })
    df = pd.DataFrame(rows)
    df["all_pass"] = bool(crit["all_pass"])
    return df


def reliability(out: Path = SPIKE_OUT) -> pd.DataFrame:
    df = _with_model_names(pd.read_csv(out / f"reliability_{LABEL}.csv"))
    df["bin"] = df.groupby(["year", "model"]).cumcount() + 1
    return df[["year", "model", "model_order", "model_name", "bin", "mean_pred", "obs_rate", "n"]].reset_index(drop=True)


def importance(out: Path = SPIKE_OUT) -> pd.DataFrame:
    df = pd.read_csv(out / f"importance_{LABEL}.csv")
    df["share"] = df["gain"] / df["gain"].sum()
    df["rank"] = df["gain"].rank(ascending=False, method="first").astype(int)
    df["feature_name"] = df["feature"].map(lambda f: FEATURES.get(f, (f, "Other"))[0])
    df["feature_group"] = df["feature"].map(lambda f: FEATURES.get(f, (f, "Other"))[1])
    return df.sort_values("rank").reset_index(drop=True)


def all_tables(out: Path = SPIKE_OUT, run_id: str | None = None) -> dict[str, pd.DataFrame]:
    tables = {"q1_metrics": metrics(out), "q1_criterion": criterion(out),
              "q1_reliability": reliability(out), "q1_importance": importance(out)}
    return {name: df.assign(run_id=run_id) for name, df in tables.items()}


def log_spike_run(out: Path = SPIKE_OUT) -> str:
    """Log the frozen spike outputs as one MLflow run and return its id.

    Logged: every CSV, JSON and PNG of `out` (artefact folder `out/`), the
    pre-registered criterion quantities per test year, the PR-AUC of every
    model and the selected hyper-parameters. Nothing is recomputed.
    """
    from src.models import tracking

    crit = json.loads((out / f"criterion_{LABEL}.json").read_text())
    params: dict[str, object] = {"label": LABEL, "train_years": "2019-2021", "test_years": "2022,2023", "seed": 42,
                                 **{f"threshold_{k}": v for k, v in THRESHOLDS.items()}}
    for model, hp in crit["tuned"].items():
        params.update({f"{model}_{k}": v for k, v in hp.items()})
    with tracking.run(EXPERIMENT, "spike_q1", params, tags={"scope": "ES PT GR 2019-2023"}) as active:
        values: dict[str, float] = {"all_pass": float(crit["all_pass"])}
        for year in sorted(k for k in crit if k.isdigit()):
            for quantity in THRESHOLDS:
                values[f"{quantity}_{year}"] = float(crit[year][quantity])
            values[f"passes_{year}"] = float(crit[year]["passes"])
        for row in metrics(out).itertuples():
            values[f"pr_auc_{row.model}_{row.year}"] = float(row.pr_auc)
            values[f"brier_{row.model}_{row.year}"] = float(row.brier)
        tracking.log_metrics(values)
        for path in sorted(out.iterdir()):
            if path.suffix in LOGGED_SUFFIXES:
                tracking.log_artifact(path, artifact_path="out")
        return str(active.info.run_id)
