"""Q1, ignition occurrence at cell x day: the production version of the Q1
spike (notebooks/spike_q1/s06_train_evaluate.py), run on the feature store
over the whole grid with the pre-registered protocol.

For each held-out year:
- hyper-parameters are chosen on the last training year (fit on the years
  before it), then the model is refitted on all training years;
- the climatology feature is rebuilt from the training years only, and for
  training rows leaves the row's own year out (REVIEW.md, leak 1);
- five models: official EFFIS danger class, logistic on log(1+FWI), LightGBM
  on the seven FWI components, LightGBM adding where/when (static fields,
  calendar, past ignitions), LightGBM with everything (weather, drought);
- metrics on the complete, unsampled held-out year; weekly block bootstrap
  on the two criterion quantities; spatial block cross-validation inside
  the held-out year as the autocorrelation control;
- feature contributions from LightGBM's own SHAP values (pred_contrib),
  aggregated by feature and by country.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression

from src.models import validation as v

FWI_FEATS = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "dsr"]
WEATHER_FEATS = ["t2m_c", "t2m_max", "rh", "rh_min", "wind", "wind_mean", "tp_mm", "tp_7d", "spei3"]
STATIC_FEATS = ["cvh", "cvl", "tvh", "tvl", "lsm", "elev_m"]
EFFIS_BREAKS = [5.2, 11.2, 21.3, 38.0, 50.0]
DEFAULT_GRID = {"num_leaves": [15, 63], "min_data_in_leaf": [50, 300]}
MODELS = ("danger_class", "logit_fwi", "lgbm_fwi", "lgbm_fwi_space", "lgbm_full")


@dataclass
class Protocol:
    label: str = "y_a"
    neg_rate: int = 20
    seed: int = 42
    clim_window: int = 31
    block_deg: float = 2.0
    block_folds: int = 5
    boot_reps: int = 200
    grid: dict = field(default_factory=lambda: dict(DEFAULT_GRID))
    num_threads: int = 6
    learning_rate: float = 0.05
    max_rounds: int = 2000
    early_stopping: int = 100
    criterion_brier: float = 0.10
    criterion_pr_auc: float = 0.05


def present(df: pd.DataFrame, feats: list[str]) -> list[str]:
    return [f for f in feats if f in df.columns]


def space_features(df: pd.DataFrame, tag: str) -> list[str]:
    return present(df, [*FWI_FEATS, *STATIC_FEATS, "doy", f"lag{tag}_7", f"lag{tag}_30", "clim"])


def full_features(df: pd.DataFrame, tag: str) -> list[str]:
    return present(df, [*FWI_FEATS, *WEATHER_FEATS, *STATIC_FEATS, "doy", f"lag{tag}_7", f"lag{tag}_30", "clim"])


# ---------------------------------------------------------------- features
def add_climatology(df: pd.DataFrame, y: str, years: list[int], window: int = 31) -> pd.DataFrame:
    """Per cell and calendar day ignition rate from `years` only; rows of those
    years leave their own year out. `years` never contains the year scored."""
    df = df.copy()
    cells = np.sort(df["cell"].unique())
    cpos = pd.Series(np.arange(len(cells)), index=cells)
    yidx = {yr: i for i, yr in enumerate(years)}
    cnt = np.zeros((len(years), len(cells), 366), dtype="float32")
    tr = df[df["year"].isin(years) & (df[y] > 0)]
    np.add.at(cnt, (tr["year"].map(yidx).to_numpy(), cpos[tr["cell"]].to_numpy(), tr["doy"].to_numpy() - 1), 1.0)
    win = uniform_filter1d(cnt, size=window, axis=2, mode="wrap") * window
    total = win.sum(axis=0)
    ci = cpos[df["cell"]].to_numpy()
    di = df["doy"].to_numpy() - 1
    own = np.zeros(len(df), dtype="float32")
    yr = df["year"].map(yidx)
    m = yr.notna().to_numpy()
    own[m] = win[yr[m].astype(int).to_numpy(), ci[m], di[m]]
    n_used = np.where(m, len(years) - 1, len(years))
    df["clim"] = ((total[ci, di] - own) / np.maximum(n_used, 1) / window).astype("float32")
    return df


def sample_negatives(df: pd.DataFrame, y: str, rate: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    keep = (df[y] > 0).to_numpy() | (rng.random(len(df)) < 1.0 / rate)
    sub = df[keep]
    return sub, np.where(sub[y].to_numpy() > 0, 1.0, float(rate))


# ------------------------------------------------------------------ models
def danger_class(fwi) -> np.ndarray:
    return np.digitize(np.asarray(fwi, dtype="float64"), EFFIS_BREAKS)


def fit_lgbm(train: pd.DataFrame, w: np.ndarray, valid: pd.DataFrame, feats: list[str], y: str, p: Protocol) -> tuple[dict, int, float]:
    """Small grid with early stopping on the validation year; returns (params, iterations, valid logloss)."""
    import lightgbm as lgb

    best: tuple[float, dict, int] | None = None
    for nl in p.grid["num_leaves"]:
        for mdl in p.grid["min_data_in_leaf"]:
            params = {
                "objective": "binary", "learning_rate": p.learning_rate, "num_leaves": nl, "min_data_in_leaf": mdl,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                "seed": p.seed, "verbose": -1, "num_threads": p.num_threads, "metric": "binary_logloss",
            }
            dtr = lgb.Dataset(train[feats], train[y], weight=w)
            dva = lgb.Dataset(valid[feats], valid[y], reference=dtr)
            booster = lgb.train(params, dtr, num_boost_round=p.max_rounds, valid_sets=[dva],
                                callbacks=[lgb.early_stopping(p.early_stopping, verbose=False)])
            score = float(booster.best_score["valid_0"]["binary_logloss"])
            if best is None or score < best[0]:
                best = (score, params, int(booster.best_iteration))
    assert best is not None
    return best[1], best[2], best[0]


def _train_models(df: pd.DataFrame, train_years: list[int], tag: str, p: Protocol) -> dict:
    """All five models for one held-out year; returns predictors and the tuned settings."""
    import lightgbm as lgb

    y = p.label
    rng = np.random.default_rng(p.seed)
    tune_year = max(train_years)
    fit_years = [yr for yr in train_years if yr != tune_year]
    full = add_climatology(df[df["year"].isin(train_years)], y, train_years, p.clim_window)
    tune = add_climatology(df[df["year"].isin(train_years)], y, fit_years, p.clim_window)
    feats_all = {"lgbm_fwi": present(df, FWI_FEATS), "lgbm_fwi_space": space_features(full, tag), "lgbm_full": full_features(full, tag)}
    sub, w = sample_negatives(full, y, p.neg_rate, rng)
    sub_fit, w_fit = sample_negatives(tune[tune["year"].isin(fit_years)], y, p.neg_rate, rng)
    val = tune[tune["year"] == tune_year]
    rates = pd.Series(full[y].to_numpy()).groupby(danger_class(full["fwi"])).mean()
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log1p(sub[["fwi"]]), sub[y], sample_weight=w)
    boosters, tuned = {}, {}
    for name, fs in feats_all.items():
        params, n_iter, score = fit_lgbm(sub_fit, w_fit, val, fs, y, p)
        tuned[name] = {"num_leaves": params["num_leaves"], "min_data_in_leaf": params["min_data_in_leaf"], "iterations": n_iter, "valid_logloss": score}
        boosters[name] = lgb.train(params, lgb.Dataset(sub[fs], sub[y], weight=w), num_boost_round=n_iter)
    return {"rates": rates, "lr": lr, "boosters": boosters, "feats": feats_all, "tuned": tuned, "train_years": train_years}


def _predict(models: dict, te: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {
        "danger_class": models["rates"].reindex(danger_class(te["fwi"])).fillna(0.0).to_numpy(),
        "logit_fwi": models["lr"].predict_proba(np.log1p(te[["fwi"]]))[:, 1],
    }
    for name, booster in models["boosters"].items():
        out[name] = booster.predict(te[models["feats"][name]])
    return out


def shap_summary(models: dict, te: pd.DataFrame, max_rows: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """Mean absolute SHAP contribution of each feature of the full model, overall and by country."""
    booster, fs = models["boosters"]["lgbm_full"], models["feats"]["lgbm_full"]
    sample = te.sample(min(max_rows, len(te)), random_state=seed) if len(te) > max_rows else te
    contrib = booster.predict(sample[fs], pred_contrib=True)[:, : len(fs)]
    rows = [{"scope": "all", "feature": f, "mean_abs_shap": float(np.abs(contrib[:, i]).mean())} for i, f in enumerate(fs)]
    if "country_key" in sample.columns:
        for c, idx in sample.groupby("country_key").indices.items():
            rows += [{"scope": str(c), "feature": f, "mean_abs_shap": float(np.abs(contrib[idx, i]).mean())} for i, f in enumerate(fs)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- protocol
def run_protocol(df: pd.DataFrame, test_years: list[int], p: Protocol | None = None, out_dir: Path | None = None) -> dict:
    """Leave-one-year-out over `test_years` (training on every other year of
    the frame). Returns tables; writes them under out_dir when given."""
    p = p or Protocol()
    tag = p.label.split("_")[-1]
    all_years = sorted(int(y) for y in df["year"].unique())
    metrics_rows, rel_rows, boot_rows, shap_rows, tuned_rows, block_rows, preds = [], [], [], [], [], [], []
    for held in test_years:
        train_years = [yr for yr in all_years if yr != held]
        if len(train_years) < 2:
            raise ValueError(f"held-out year {held}: fewer than two training years")
        models = _train_models(df, train_years, tag, p)
        te = add_climatology(df[df["year"] == held], p.label, train_years, p.clim_window).reset_index(drop=True)
        yt = te[p.label].to_numpy().astype(int)
        pr = _predict(models, te)
        for name in MODELS:
            m = v.metrics(yt, pr[name])
            metrics_rows.append({"held_year": held, "model": name, **m})
            rel = v.reliability(yt, pr[name])
            rel.insert(0, "model", name)
            rel.insert(0, "held_year", held)
            rel_rows.append(rel)
        weeks = te["date"].dt.isocalendar().week.to_numpy()
        boot = v.block_bootstrap_delta(yt, pr["lgbm_fwi"], pr["lgbm_full"], weeks, reps=p.boot_reps, seed=p.seed)
        boot.insert(0, "held_year", held)
        boot_rows.append(boot)
        sh = shap_summary(models, te, seed=p.seed)
        sh.insert(0, "held_year", held)
        shap_rows.append(sh)
        for name, t in models["tuned"].items():
            tuned_rows.append({"held_year": held, "model": name, **t})
        block_rows.append(_spatial_block_control(df, te, held, train_years, tag, p))
        preds.append(pd.DataFrame({"cell": te["cell"], "date": te["date"], "y": yt, "p_fwi": pr["lgbm_fwi"], "p_full": pr["lgbm_full"]}))
    metrics = pd.DataFrame(metrics_rows)
    crit = _criterion(metrics, p)
    result = {
        "metrics": metrics, "reliability": pd.concat(rel_rows, ignore_index=True), "bootstrap": pd.concat(boot_rows, ignore_index=True),
        "shap": pd.concat(shap_rows, ignore_index=True), "tuned": pd.DataFrame(tuned_rows), "spatial_blocks": pd.DataFrame(block_rows),
        "predictions": pd.concat(preds, ignore_index=True), "criterion": crit,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for k in ("metrics", "reliability", "bootstrap", "shap", "tuned", "spatial_blocks"):
            result[k].to_csv(out_dir / f"{k}_{tag}.csv", index=False)
        result["predictions"].to_parquet(out_dir / f"predictions_{tag}.parquet", index=False)
        (out_dir / f"criterion_{tag}.json").write_text(json.dumps(crit, indent=2))
    return result


def _spatial_block_control(df: pd.DataFrame, te: pd.DataFrame, held: int, train_years: list[int], tag: str, p: Protocol) -> dict:
    """Full model scored on spatial folds of the held-out year, trained without those blocks."""
    import lightgbm as lgb

    from src import grid

    lon, lat = grid.cell_centre(df["cell"].to_numpy())
    blocks = v.spatial_blocks(lon, lat, p.block_deg)
    tr_all = add_climatology(df[df["year"].isin(train_years)], p.label, train_years, p.clim_window)
    fs = full_features(tr_all, tag)
    tr_blocks = blocks[df["year"].isin(train_years).to_numpy()]
    te_lon, te_lat = grid.cell_centre(te["cell"].to_numpy())
    te_blocks = v.spatial_blocks(te_lon, te_lat, p.block_deg)
    rng = np.random.default_rng(p.seed)
    ys, ps = [], []
    uniq_blocks = np.unique(np.concatenate([tr_blocks, te_blocks]))
    for _tr_idx, te_idx, _fold in v.block_folds(uniq_blocks, p.block_folds, p.seed):
        held_blocks = set(uniq_blocks[te_idx])
        tr = tr_all[~pd.Series(tr_blocks).isin(held_blocks).to_numpy()]
        tst = te[pd.Series(te_blocks).isin(held_blocks).to_numpy()]
        if tst.empty or tst[p.label].sum() == 0 or tr.empty or tr[p.label].sum() == 0:
            continue
        sub, w = sample_negatives(tr, p.label, p.neg_rate, rng)
        params = {"objective": "binary", "learning_rate": p.learning_rate, "num_leaves": p.grid["num_leaves"][0],
                  "min_data_in_leaf": p.grid["min_data_in_leaf"][0], "feature_fraction": 0.8, "seed": p.seed,
                  "verbose": -1, "num_threads": p.num_threads}
        booster = lgb.train(params, lgb.Dataset(sub[fs], sub[p.label], weight=w), num_boost_round=200)
        ys.append(tst[p.label].to_numpy().astype(int))
        ps.append(booster.predict(tst[fs]))
    if not ys:
        return {"held_year": held, "pr_auc_blocked": float("nan"), "brier_blocked": float("nan"), "n": 0}
    y_all, p_all = np.concatenate(ys), np.concatenate(ps)
    m = v.metrics(y_all, p_all)
    return {"held_year": held, "pr_auc_blocked": m["pr_auc"], "brier_blocked": m["brier"], "n": int(m["n"])}


def _criterion(metrics: pd.DataFrame, p: Protocol) -> dict:
    crit: dict = {}
    for held, g in metrics.groupby("held_year"):
        m3 = g[g["model"] == "lgbm_fwi"].iloc[0]
        m4 = g[g["model"] == "lgbm_full"].iloc[0]
        rel = float((m3["brier"] - m4["brier"]) / m3["brier"]) if m3["brier"] > 0 else float("nan")
        gain = float(m4["pr_auc"] - m3["pr_auc"])
        crit[str(held)] = {
            "brier_rel_improvement": rel, "pr_auc_gain": gain,
            "pr_auc_rel_gain": float(gain / m3["pr_auc"]) if m3["pr_auc"] > 0 else float("nan"),
            "passes": bool(rel >= p.criterion_brier and gain >= p.criterion_pr_auc),
        }
    crit["all_pass"] = all(val["passes"] for k, val in crit.items() if k.isdigit())
    crit["thresholds"] = {"brier_rel_improvement": p.criterion_brier, "pr_auc_gain": p.criterion_pr_auc}
    return crit


def monthly_country_summary(predictions: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """For the report: observed and predicted ignition rates per country and month."""
    df = predictions.merge(cells[["cell", "country_key"]], on="cell", how="left")
    df["month_key"] = df["date"].values.astype("datetime64[M]")
    return df.groupby(["country_key", "month_key"], as_index=False).agg(
        cell_days=("y", "size"), ignitions=("y", "sum"), p_fwi_mean=("p_fwi", "mean"), p_full_mean=("p_full", "mean"),
    )
