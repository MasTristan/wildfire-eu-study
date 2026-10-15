"""Step 3: train the four models and score 2022 and 2023, per PROTOCOL.md.

    uv run python notebooks/spike_q1/s06_train_evaluate.py [a|b ...]   # default: a b

Writes notebooks/spike_q1/out/{metrics,reliability,bootstrap,importance}_<label>.csv,
reliability_<label>.png and criterion_<label>.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import lightgbm as lgb  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
NEG_RATE = 20  # keep 1 negative in 20, weight 20
CLIM_WINDOW = 31  # +/- 15 days
FWI_FEATS = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "dsr"]
WEATHER = ["t2m_c", "d2m_c", "rh", "wind", "u10", "v10", "tp_mm", "tp_7d", "spei3"]
STATIC = ["cvh", "cvl", "tvh", "tvl", "lsm", "elev_m"]
EFFIS_BREAKS = [5.2, 11.2, 21.3, 38.0, 50.0]  # official EFFIS FWI danger classes
GRID = {"num_leaves": [15, 63], "min_data_in_leaf": [50, 300]}  # same for models 3 and 4


def full_features(tag: str) -> list[str]:
    return [*FWI_FEATS, *WEATHER, *STATIC, "doy", f"lag{tag}_7", f"lag{tag}_30", "clim"]


# ------------------------------------------------------------------- climatology
def add_climatology(df: pd.DataFrame, y: str) -> pd.DataFrame:
    """Per cell and calendar day (+/-15 d) ignition rate from TRAIN years only.

    Training rows leave their own year out, so the feature never contains the
    row's own label; test rows use all training years.
    """
    cells = np.sort(df["cell"].unique())
    cpos = pd.Series(np.arange(len(cells)), index=cells)
    years = common.TRAIN_YEARS
    cnt = np.zeros((len(years), len(cells), 366), dtype="float32")
    tr = df[df["year"].isin(years) & (df[y] > 0)]
    np.add.at(cnt, (tr["year"].map({yr: i for i, yr in enumerate(years)}).to_numpy(),
                    cpos[tr["cell"]].to_numpy(), tr["doy"].to_numpy() - 1), 1.0)
    win = uniform_filter1d(cnt, size=CLIM_WINDOW, axis=2, mode="wrap") * CLIM_WINDOW
    total = win.sum(axis=0)  # cell x doy
    ci = cpos[df["cell"]].to_numpy()
    di = df["doy"].to_numpy() - 1
    own = np.zeros(len(df), dtype="float32")
    yr_idx = df["year"].map({yr: i for i, yr in enumerate(years)})
    m = yr_idx.notna().to_numpy()
    own[m] = win[yr_idx[m].astype(int).to_numpy(), ci[m], di[m]]
    n_used = np.where(m, len(years) - 1, len(years))
    df["clim"] = ((total[ci, di] - own) / (n_used * CLIM_WINDOW)).astype("float32")
    return df


# ------------------------------------------------------------------------ models
def danger_class(fwi: pd.Series) -> np.ndarray:
    return np.digitize(fwi.to_numpy(), EFFIS_BREAKS)


def sample_train(df: pd.DataFrame, y: str, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    keep = (df[y] > 0).to_numpy() | (rng.random(len(df)) < 1.0 / NEG_RATE)
    sub = df[keep]
    w = np.where(sub[y].to_numpy() > 0, 1.0, float(NEG_RATE))
    return sub, w


def fit_lgbm(train, w, valid, feats, y, label):
    """Small grid on 2021, early stopping; returns (best_params, best_iteration)."""
    best = None
    for nl in GRID["num_leaves"]:
        for mdl in GRID["min_data_in_leaf"]:
            params = {
                "objective": "binary", "learning_rate": 0.05, "num_leaves": nl, "min_data_in_leaf": mdl,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                "seed": SEED, "verbose": -1, "num_threads": 6, "metric": "binary_logloss",
            }
            dtr = lgb.Dataset(train[feats], train[y], weight=w)
            dva = lgb.Dataset(valid[feats], valid[y], reference=dtr)
            mdl_ = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                             callbacks=[lgb.early_stopping(100, verbose=False)])
            score = mdl_.best_score["valid_0"]["binary_logloss"]
            print(f"    {label} leaves={nl} min_data={mdl}: logloss(2021)={score:.5f} iters={mdl_.best_iteration}")
            if best is None or score < best[0]:
                best = (score, params, mdl_.best_iteration)
    return best[1], best[2]


def run(tag: str) -> None:
    y = f"y_{tag}"
    rng = np.random.default_rng(SEED)
    out = common.OUT
    with common.timed(f"load_{tag}"):
        df = pd.read_parquet(common.INTERIM / "daily.parquet")
    with common.timed(f"features_{tag}"):
        df = add_climatology(df, y)
    feats = full_features(tag)
    train = df[df["year"].isin(common.TRAIN_YEARS)]
    print(f"[{tag}] rows train={len(train):,} positives={int(train[y].sum()):,}; "
          f"test positives={ {yr: int(df.loc[df.year == yr, y].sum()) for yr in common.TEST_YEARS} }")

    preds: dict[str, dict[int, np.ndarray]] = {m: {} for m in ("danger_class", "logit_fwi", "lgbm_fwi", "lgbm_full")}
    importance = {}
    tuned = {}

    with common.timed(f"train_{tag}"):
        # 1. danger class: class -> empirical training ignition rate
        cls_tr = danger_class(train["fwi"])
        rates = pd.Series(train[y].to_numpy()).groupby(cls_tr).mean()
        # 2. logistic on log(1+FWI), sampled+weighted like the trees
        sub, w = sample_train(train, y, rng)
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(np.log1p(sub[["fwi"]]), sub[y], sample_weight=w)
        # 3 and 4. LightGBM: tune on 2021 (fit 2019-2020), refit on 2019-2021
        fit_part = df[df["year"].isin([2019, 2020])]
        val = df[df["year"] == 2021]
        sub_fit, w_fit = sample_train(fit_part, y, rng)
        models = {}
        for name, fs in (("lgbm_fwi", FWI_FEATS), ("lgbm_full", feats)):
            params, n_iter = fit_lgbm(sub_fit, w_fit, val, fs, y, name)
            tuned[name] = {**{k: params[k] for k in ("num_leaves", "min_data_in_leaf")}, "iterations": n_iter}
            dtr = lgb.Dataset(sub[fs], sub[y], weight=w)
            models[name] = (lgb.train(params, dtr, num_boost_round=n_iter), fs)
        importance = pd.DataFrame({
            "feature": feats,
            "gain": models["lgbm_full"][0].feature_importance("gain"),
        }).sort_values("gain", ascending=False)

    rows, rel_rows = [], []
    boot_inputs = {}
    with common.timed(f"predict_{tag}"):
        for yr in common.TEST_YEARS:
            te = df[df["year"] == yr].reset_index(drop=True)
            p = {
                "danger_class": rates.reindex(danger_class(te["fwi"])).fillna(0.0).to_numpy(),
                "logit_fwi": lr.predict_proba(np.log1p(te[["fwi"]]))[:, 1],
            }
            for name, (m, fs) in models.items():
                p[name] = m.predict(te[fs])
            boot_inputs[yr] = (te[y].to_numpy(), te["date"].dt.isocalendar().week.to_numpy(), p)
            yt = te[y].to_numpy()
            for name, pr in p.items():
                preds[name][yr] = pr
                rows.append({
                    "label": tag, "year": yr, "model": name, "n": len(te), "positives": int(yt.sum()),
                    "base_rate": yt.mean(), "pr_auc": average_precision_score(yt, pr),
                    "brier": brier_score_loss(yt, pr), "logloss": log_loss(yt, np.clip(pr, 1e-9, 1 - 1e-9)),
                })
                q = pd.qcut(pd.Series(pr), 10, duplicates="drop")
                g = pd.DataFrame({"p": pr, "y": yt}).groupby(q, observed=True).agg(
                    mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
                g.insert(0, "model", name)
                g.insert(0, "year", yr)
                rel_rows.append(g.reset_index(drop=True))
    metrics = pd.DataFrame(rows)

    with common.timed(f"bootstrap_{tag}"):
        boot = bootstrap_deltas(boot_inputs)

    # decision criterion (PROTOCOL.md): model 4 vs model 3, both test years
    crit = {}
    for yr in common.TEST_YEARS:
        m3 = metrics[(metrics.year == yr) & (metrics.model == "lgbm_fwi")].iloc[0]
        m4 = metrics[(metrics.year == yr) & (metrics.model == "lgbm_full")].iloc[0]
        crit[str(yr)] = {
            "brier_rel_improvement": float((m3.brier - m4.brier) / m3.brier),
            "pr_auc_gain": float(m4.pr_auc - m3.pr_auc),
        }
        crit[str(yr)]["passes"] = bool(crit[str(yr)]["brier_rel_improvement"] >= 0.10
                                       and crit[str(yr)]["pr_auc_gain"] >= 0.05)
    crit["all_pass"] = all(v["passes"] for k, v in crit.items() if k.isdigit())
    crit["tuned"] = tuned

    common.OUT.mkdir(exist_ok=True)
    metrics.to_csv(out / f"metrics_{tag}.csv", index=False)
    pd.concat(rel_rows).to_csv(out / f"reliability_{tag}.csv", index=False)
    boot.to_csv(out / f"bootstrap_{tag}.csv", index=False)
    importance.to_csv(out / f"importance_{tag}.csv", index=False)
    (out / f"criterion_{tag}.json").write_text(json.dumps(crit, indent=2))
    reliability_figure(pd.concat(rel_rows), tag)
    print(metrics.round(5).to_string(index=False))
    print(json.dumps(crit, indent=2))


def bootstrap_deltas(boot_inputs, reps: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for yr, (yt, week, p) in boot_inputs.items():
        groups = [np.flatnonzero(week == w) for w in np.unique(week)]
        d_brier, d_ap = [], []
        for _ in range(reps):
            pick = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            yb = yt[idx]
            if yb.sum() == 0:
                continue
            b3, b4 = (np.mean((p[m][idx] - yb) ** 2) for m in ("lgbm_fwi", "lgbm_full"))
            a3, a4 = (average_precision_score(yb, p[m][idx]) for m in ("lgbm_fwi", "lgbm_full"))
            d_brier.append((b3 - b4) / b3)
            d_ap.append(a4 - a3)
        for name, arr in (("brier_rel_improvement", d_brier), ("pr_auc_gain", d_ap)):
            lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
            rows.append({"year": yr, "quantity": name, "p2.5": lo, "median": med, "p97.5": hi, "reps": len(arr)})
    return pd.DataFrame(rows)


def reliability_figure(rel: pd.DataFrame, tag: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(common.TEST_YEARS), figsize=(11, 4.5), sharey=True)
    for ax, yr in zip(np.atleast_1d(axes), common.TEST_YEARS, strict=True):
        sub = rel[rel.year == yr]
        lim = max(sub["mean_pred"].max(), sub["obs_rate"].max()) * 1.1
        ax.plot([1e-5, lim], [1e-5, lim], "k--", lw=0.8)
        for name, g in sub.groupby("model"):
            ax.plot(g["mean_pred"], g["obs_rate"], marker="o", ms=3, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"label ({tag}), test {yr}")
        ax.set_xlabel("mean predicted probability (decile)")
    np.atleast_1d(axes)[0].set_ylabel("observed ignition rate")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(common.OUT / f"reliability_{tag}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["a", "b"]:
        run(t)
