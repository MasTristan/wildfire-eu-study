"""Exploratory analyses after the verdict (EXPLORATORY.md), label (a).

    uv run python notebooks/spike_q1/s08_exploratory.py

Reuses the verdict pipeline: s06 functions, hyper-parameters from
out/criterion_a.json, training 2019-2021, testing 2022 and 2023. Writes
out/exploratory_spatial_blocks.csv, out/exploratory_ablation.csv,
out/exploratory_new_ignitions.csv, out/exploratory_seeds.csv and
out/exploratory_seeds_summary.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import lightgbm as lgb  # noqa: E402
import s06_train_evaluate as s06  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss  # noqa: E402

from src import grid  # noqa: E402
from src.models import validation  # noqa: E402

Y = "y_a"
TAG = "a"
N_SEEDS = 10


def params_for(tuned: dict) -> dict:
    return {
        "objective": "binary", "learning_rate": 0.05, "num_leaves": tuned["num_leaves"],
        "min_data_in_leaf": tuned["min_data_in_leaf"], "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "seed": s06.SEED, "verbose": -1, "num_threads": 6,
    }


def fit(train: pd.DataFrame, feats: list[str], tuned: dict, rng: np.random.Generator) -> lgb.Booster:
    sub, w = s06.sample_train(train, Y, rng)
    return fit_on(sub, w, feats, tuned)


def fit_on(sub: pd.DataFrame, w: np.ndarray, feats: list[str], tuned: dict, seed: int = s06.SEED) -> lgb.Booster:
    params = {**params_for(tuned), "seed": seed}
    return lgb.train(params, lgb.Dataset(sub[feats], sub[Y], weight=w), num_boost_round=tuned["iterations"])


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """PR-AUC depends on the base rate, so it is reported with its lift
    (PR-AUC / base rate), the only form comparable across subsets."""
    if y.sum() == 0:
        return {"n": int(len(y)), "positives": 0, "base_rate": 0.0, "pr_auc": float("nan"),
                "lift": float("nan"), "brier": float("nan")}
    ap, base = float(average_precision_score(y, p)), float(y.mean())
    return {"n": int(len(y)), "positives": int(y.sum()), "base_rate": base, "pr_auc": ap,
            "lift": ap / base, "brier": float(brier_score_loss(y, p))}


def main() -> None:
    df = pd.read_parquet(common.INTERIM / "daily.parquet")
    s06.WEATHER[:] = [c for c in s06.WEATHER_ALL if c in df.columns]
    df = s06.add_climatology(df, Y, common.TRAIN_YEARS)
    crit = json.loads((common.OUT / "criterion_a.json").read_text())["tuned"]
    feats = {
        "lgbm_fwi": list(s06.FWI_FEATS),
        "lgbm_fwi_space": s06.space_features(TAG),
        "lgbm_full": s06.full_features(TAG),
        "3c_no_lags": [*s06.FWI_FEATS, *s06.STATIC, "doy", "clim"],
        "3d_fwi_lags": [*s06.FWI_FEATS, f"lag{TAG}_7", f"lag{TAG}_30"],
        "3e_no_fwi": [*s06.STATIC, "doy", "clim", f"lag{TAG}_7", f"lag{TAG}_30"],
    }
    tuned_of = {"lgbm_fwi": crit["lgbm_fwi"], "lgbm_full": crit["lgbm_full"]}
    for k in ("lgbm_fwi_space", "3c_no_lags", "3d_fwi_lags", "3e_no_fwi"):
        tuned_of[k] = crit["lgbm_fwi_space"]
    train = df[df["year"].isin(common.TRAIN_YEARS)]

    # ---- E2 and E3: one fit per model on all training years ----------------
    # One negative sample drawn exactly as s06 draws it (first draw of
    # default_rng(SEED) on the training rows), shared by every model, so the
    # refits reproduce section 3 of VERDICT.md. The first version drew a new
    # sample per model, which is why its place-and-time PR-AUCs differed.
    rng = np.random.default_rng(s06.SEED)
    shared_sub, shared_w = s06.sample_train(train, Y, rng)
    boosters = {m: fit_on(shared_sub, shared_w, fs, tuned_of[m]) for m, fs in feats.items()}
    rates = pd.Series(train[Y].to_numpy()).groupby(s06.danger_class(train["fwi"])).mean()
    abl_rows, new_rows = [], []
    lon, lat = grid.cell_centre(df["cell"].to_numpy())
    for yr in common.TEST_YEARS:
        te = df[df["year"] == yr].reset_index(drop=True)
        yt = te[Y].to_numpy()
        preds = {m: b.predict(te[feats[m]]) for m, b in boosters.items()}
        preds["danger_class"] = rates.reindex(s06.danger_class(te["fwi"])).fillna(0.0).to_numpy()
        ref = score(yt, preds["lgbm_fwi_space"])["pr_auc"]
        for m in ("lgbm_fwi", "lgbm_fwi_space", "3c_no_lags", "3d_fwi_lags", "3e_no_fwi", "lgbm_full"):
            s = score(yt, preds[m])
            abl_rows.append({"year": yr, "model": m, **s, "share_of_3b_pr_auc": s["pr_auc"] / ref})
        # neighbourhood lag: any ignition in the 3x3 block over the previous 30 days
        lag30 = df.set_index(["cell", "date"])[f"lag{TAG}_30"]
        nb = np.zeros(len(te))
        for d_cell in grid.neighbours(np.array([0]))[:]:
            key = pd.MultiIndex.from_arrays([te["cell"].to_numpy() + d_cell, te["date"].to_numpy()])
            nb += lag30.reindex(key).fillna(0.0).to_numpy()
        subsets = {"all": np.ones(len(te), bool), "cell_new_30d": (te[f"lag{TAG}_30"] == 0).to_numpy(),
                   "neighbourhood_new_30d": nb == 0}
        for sname, mask in subsets.items():
            for m in ("danger_class", "lgbm_fwi", "lgbm_fwi_space", "lgbm_full"):
                s = score(yt[mask], preds[m][mask])
                new_rows.append({"year": yr, "subset": sname, "model": m, **s,
                                 "share_of_positives": s["positives"] / max(int(yt.sum()), 1)})

    # ---- E1: spatial blocks ---------------------------------------------------
    blocks = validation.spatial_blocks(lon, lat, 2.0)
    df_blocks = pd.Series(blocks, index=df.index)
    uniq = np.unique(blocks)
    blk_rows = []
    pooled: dict[tuple[int, str], list] = {}
    for _tr, te_idx, fold in validation.block_folds(uniq, k=5, seed=s06.SEED):
        held = set(uniq[te_idx])
        in_held = df_blocks.isin(held)
        tr = df[df["year"].isin(common.TRAIN_YEARS) & ~in_held]
        rng = np.random.default_rng(s06.SEED + fold)
        for m in ("lgbm_fwi", "lgbm_fwi_space", "lgbm_full"):
            b = fit(tr, feats[m], tuned_of[m], rng)
            for yr in common.TEST_YEARS:
                te = df[(df["year"] == yr) & in_held]
                if te.empty:  # a fold can hold only blocks without cells in the mesh
                    continue
                pooled.setdefault((yr, m), []).append((te[Y].to_numpy(), b.predict(te[feats[m]])))
        print(f"fold {fold + 1}/5 done ({len(held)} blocks held out)", flush=True)
    unblocked = {(r["year"], r["model"]): r["pr_auc"] for r in abl_rows}
    for (yr, m), parts in sorted(pooled.items()):
        y_all = np.concatenate([p[0] for p in parts])
        p_all = np.concatenate([p[1] for p in parts])
        s = score(y_all, p_all)
        u = unblocked[(yr, m)]
        blk_rows.append({"year": yr, "model": m, **s, "pr_auc_unblocked": u,
                         "relative_drop": (u - s["pr_auc"]) / u if u else float("nan"),
                         "transfers": bool((u - s["pr_auc"]) / u <= 0.30) if u else False})

    # ---- E4: sensitivity to the training sample (EXPLORATORY.md, addendum) ---
    seed_rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        sub, w = s06.sample_train(train, Y, rng)
        models = {m: fit_on(sub, w, feats[m], tuned_of[m], seed=seed) for m in ("lgbm_fwi", "lgbm_fwi_space", "lgbm_full")}
        for yr in common.TEST_YEARS:
            te = df[df["year"] == yr]
            yt = te[Y].to_numpy()
            sc = {m: score(yt, b.predict(te[feats[m]])) for m, b in models.items()}
            m3, m4 = sc["lgbm_fwi"], sc["lgbm_full"]
            rel = (m3["brier"] - m4["brier"]) / m3["brier"]
            gain = m4["pr_auc"] - m3["pr_auc"]
            seed_rows.append({"seed": seed, "year": yr, "pr_auc_m3": m3["pr_auc"], "pr_auc_m3b": sc["lgbm_fwi_space"]["pr_auc"],
                              "pr_auc_m4": m4["pr_auc"], "brier_rel_improvement": rel, "pr_auc_gain": gain,
                              "passes": bool(rel >= 0.10 and gain >= 0.05)})
        print(f"seed {seed + 1}/{N_SEEDS} done", flush=True)
    seeds = pd.DataFrame(seed_rows)
    per_seed = seeds.groupby("seed")["passes"].all()
    summary = seeds.groupby("year").agg(
        brier_rel_min=("brier_rel_improvement", "min"), brier_rel_median=("brier_rel_improvement", "median"),
        brier_rel_max=("brier_rel_improvement", "max"), pr_auc_m4_min=("pr_auc_m4", "min"),
        pr_auc_m4_median=("pr_auc_m4", "median"), pr_auc_m4_max=("pr_auc_m4", "max"),
        share_passing_year=("passes", "mean"),
    ).reset_index()
    summary["share_seeds_passing_both_years"] = float(per_seed.mean())

    out = common.OUT
    seeds.to_csv(out / "exploratory_seeds.csv", index=False)
    summary.to_csv(out / "exploratory_seeds_summary.csv", index=False)
    pd.DataFrame(blk_rows).to_csv(out / "exploratory_spatial_blocks.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(out / "exploratory_ablation.csv", index=False)
    pd.DataFrame(new_rows).to_csv(out / "exploratory_new_ignitions.csv", index=False)
    for name in ("spatial_blocks", "ablation", "new_ignitions", "seeds_summary"):
        print(f"\n== {name} ==")
        print(pd.read_csv(out / f"exploratory_{name}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    with common.timed("exploratory"):
        main()
