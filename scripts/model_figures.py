"""Figures of the model cards (docs/models/*.md), rebuilt from the model outputs.

Reads `gold/model_out` (Q2 tail risk, Q3 attribution, Q1 report tables) and the
spike's committed outputs (`notebooks/spike_q1/out`), writes PNG files to
`docs/img/models/`. Nothing is recomputed: every plotted number is a cell of
one of those tables, so a figure and the card that cites it cannot disagree.

    uv run python scripts/model_figures.py [--only q1|q2|q3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import LogLocator, ScalarFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import lake  # noqa: E402

OUT_DIR: Path = lake.ROOT / "docs" / "img" / "models"
SPIKE_OUT: Path = lake.ROOT / "notebooks" / "spike_q1" / "out"

INK = "#222222"
GREY = "#8a8a8a"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
SKY = "#56B4E9"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9.5, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7, "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
})

# Q2: the five Mediterranean countries of the model card; the others of the published perimeter appear in the shape figure
Q2_COUNTRIES = ["PT", "ES", "FR", "IT", "GR"]
COUNTRY_NAMES = {"PT": "Portugal", "ES": "Espagne", "FR": "France", "IT": "Italie", "GR": "Grèce",
                 "BG": "Bulgarie", "HR": "Croatie", "RO": "Roumanie"}

METRIC_LABELS = {
    "days_high_season": "Jours de danger élevé\n(juin à septembre)",
    "spei_spring": "Bilan hydrique SPEI\n(mars à mai)",
    "precip_anom_season": "Anomalie de pluie\n(juin à septembre)",
    "temp_season": "Température moyenne\n(juin à septembre)",
}

Q1_MODEL_LABELS = {
    "danger_class": "1 Classe de danger",
    "logit_fwi": "2 Logistique sur FWI",
    "lgbm_fwi": "3 LightGBM, composantes FWI",
    "lgbm_fwi_space": "3b LightGBM, FWI + lieu et saison",
    "lgbm_full": "4 LightGBM, modèle complet",
}
Q1_COLORS = {"danger_class": GREY, "lgbm_fwi": ORANGE, "lgbm_fwi_space": GREEN, "lgbm_full": BLUE}


def _save(fig: Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def q2_return_levels(model_out: Path) -> Path:
    """Fitted return level with its 95 % block-bootstrap band where it is published, empirical level as points."""
    levels = pd.read_parquet(model_out / "tail_risk_levels.parquet")
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4), sharex=True)
    for ax, key in zip(axes.ravel(), Q2_COUNTRIES, strict=False):
        d = levels[(levels["country_key"] == key) & levels["return_level_ha"].notna()]
        x = d["period_years"].to_numpy()
        if len(d):
            ax.fill_between(x, d["return_level_p2_5"], d["return_level_p97_5"], color=BLUE, alpha=0.18, lw=0,
                            label="Intervalle à 95 % (bootstrap par années)")
            ax.plot(x, d["return_level_ha"], color=BLUE, marker="o", ms=4, label="Niveau de retour ajusté")
        emp = levels[(levels["country_key"] == key) & levels["empirical_level_ha"].notna()]
        ax.scatter(emp["period_years"], emp["empirical_level_ha"], color=ORANGE, zorder=3, s=26,
                   label="Niveau observé (empirique, épisodes)")
        if not len(d):
            ax.text(0.97, 0.05, "niveaux ajustés non publiés\n(forme ξ ≥ 1)", transform=ax.transAxes, ha="right",
                    va="bottom", color=INK, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20], labels=["2", "5", "10", "20"])
        ax.minorticks_off()
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(labelbottom=True)
        ax.set_title(COUNTRY_NAMES[key], loc="left", fontsize=10.5, color=INK)
    axes[1, 2].axis("off")
    for ax in list(axes[1, :2]) + [axes[0, 0], axes[0, 1], axes[0, 2]]:
        ax.set_xlabel("Période de retour (années)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Taille du feu (hectares, échelle log)")
    handles, labels = [], []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    axes[1, 2].legend(handles, labels, loc="center", fontsize=8.5)
    fig.suptitle("Q2 : taille de l'épisode dépassée en moyenne une fois par période de retour", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "q2_return_levels.png")


def q2_shape(model_out: Path) -> Path:
    """GPD shape parameter with its 95 % interval for every country of the published perimeter."""
    t = pd.read_parquet(model_out / "tail_risk_country.parquet")
    t = t[t["shape"].notna()].sort_values("shape").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [ORANGE if s >= 1 else (BLUE if k in Q2_COUNTRIES else SKY) for k, s in zip(t["country_key"], t["shape"], strict=True)]
    ax.barh(t["country_key"], t["shape"], color=colors, height=0.62)
    ax.errorbar(t["shape"], np.arange(len(t)), xerr=[t["shape"] - t["shape_p2.5"], t["shape_p97.5"] - t["shape"]],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5)
    ax.axvline(0, color=GREY, lw=0.9)
    ax.axvline(1, color=ORANGE, lw=1.1, ls="--")
    ax.text(0.99, 1.0, "forme = 1 : moyenne infinie (niveaux non publiés au-delà)", color=ORANGE, ha="right", va="bottom",
            fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Paramètre de forme de la loi de Pareto généralisée (sans unité)")
    ax.set_ylabel("Pays (code ISO)")
    ax.set_title("Q2 : forme de la queue par pays, intervalle à 95 % (bootstrap par années)", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, "q2_shape.png")


def q2_threshold_stability(model_out: Path) -> Path:
    """Shape against the threshold (200, 300, 500, 1000 ha), five countries, with 95 % intervals."""
    s = pd.read_parquet(model_out / "tail_risk_threshold_stability.parquet")
    thresholds = sorted(s["threshold_ha"].unique())
    palette = [BLUE, ORANGE, GREEN, PURPLE, SKY]
    keys = [k for k in Q2_COUNTRIES if k in set(s["country_key"])]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, (key, color) in enumerate(zip(keys, palette, strict=False)):
        d = s[(s["country_key"] == key) & s["shape"].notna()].sort_values("threshold_ha")
        xpos = np.array([thresholds.index(u) for u in d["threshold_ha"]]) + (i - (len(keys) - 1) / 2) * 0.07
        ax.errorbar(xpos, d["shape"], yerr=[d["shape"] - d["shape_p2.5"], d["shape_p97.5"] - d["shape"]], color=color,
                    marker="o", ms=4, capsize=2.5, lw=1.3, label=COUNTRY_NAMES[key])
    ax.axhline(1, color=ORANGE, lw=1.0, ls="--")
    ax.axhline(0, color=GREY, lw=0.9)
    ax.set_xticks(range(len(thresholds)), labels=[f"{int(u)}" for u in thresholds])
    ax.set_xlabel("Seuil (hectares)")
    ax.set_ylabel("Forme ξ de la loi de Pareto généralisée")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=8.5)
    ax.set_title("Q2 : stabilité de la forme selon le seuil, intervalle à 95 % (bootstrap par années)", loc="left",
                 fontsize=11, color=INK)
    return _save(fig, "q2_threshold_stability.png")


def q2_qq(model_out: Path) -> Path:
    """Quantile-quantile plot of the episodes against the fitted GPD, one panel per country of the perimeter,
    with the Anderson-Darling p-value. Points on the diagonal: the law describes the episodes."""
    ep = pd.read_parquet(model_out / "tail_risk_episodes.parquet")
    gof = pd.read_parquet(model_out / "tail_risk_gof.parquet").set_index("country_key")
    keys = [k for k in Q2_COUNTRIES if k in gof.index] + sorted(k for k in gof.index if k not in Q2_COUNTRIES)
    ncols = 4
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 2.9 * nrows), squeeze=False)
    for ax, key in zip(axes.ravel(), keys, strict=False):
        d = ep[ep["country_key"] == key].sort_values("plotting_position")
        lo = float(min(d["area_ha"].min(), d["model_quantile_ha"].min()))
        hi = float(max(d["area_ha"].max(), d["model_quantile_ha"].max()))
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9)
        rejected = bool(gof.loc[key, "gpd_rejected"])
        ax.scatter(d["model_quantile_ha"], d["area_ha"], s=11, color=ORANGE if rejected else BLUE, alpha=0.8, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        p_value = float(gof.loc[key, "p_value"])
        ax.set_title(f"{COUNTRY_NAMES.get(key, key)} : {int(gof.loc[key, 'n_exceedances'])} épisodes, p = {p_value:.2f}"
                     .replace(".", ","), loc="left", fontsize=9, color=INK)
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Taille attendue par la loi ajustée (ha)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Taille observée (ha)")
    fig.suptitle("Q2 : épisodes observés contre la loi de Pareto généralisée ajustée (échelles log). "
                 "p : test d'Anderson-Darling, orange si la loi est rejetée à 5 %", x=0.01, ha="left", fontsize=10.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "q2_qq.png")


def _bayes_pop(model_out: Path) -> pd.DataFrame:
    b = pd.read_parquet(model_out / "attribution_bayes_coefficients.parquet")
    pop = b[b["level"] == "population"].copy()
    verdict = pop["robust"].map({True: "robuste", False: "non distinguable de zéro"})
    pop["label"] = pop["metric"].map(METRIC_LABELS) + "\n(" + verdict + ")"
    return pop.reset_index(drop=True)


def _q3_size(model_out: Path) -> tuple[int, int]:
    """Observations and countries of the single Q3 panel (both models fit exactly these rows)."""
    cy = pd.read_parquet(model_out / "attribution_bayes_countryyear.parquet")
    return len(cy), int(cy["country_key"].nunique())


def q3_coefficients(model_out: Path) -> Path:
    """Population slope per metric: frequentist estimate (95 % interval) against Bayesian posterior (95 % HDI)."""
    pop = _bayes_pop(model_out)
    y = np.arange(len(pop))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.errorbar(pop["freq_coef"], y + 0.12, xerr=[pop["freq_coef"] - pop["freq_ci_low"], pop["freq_ci_high"] - pop["freq_coef"]],
                fmt="o", color=ORANGE, capsize=3, label="Fréquentiste (MCO, intervalle à 95 %)")
    ax.errorbar(pop["mean"], y - 0.12, xerr=[pop["mean"] - pop["hdi_low"], pop["hdi_high"] - pop["mean"]],
                fmt="s", color=BLUE, capsize=3, label="Bayésien hiérarchique (moyenne, HDI à 95 %)")
    ax.axvline(0, color=GREY, lw=0.9)
    ax.set_yticks(y, labels=pop["label"])
    ax.set_xlabel("Effet sur le log de la surface brûlée, par écart-type intra-pays de la variable (écart d'une année à l'autre)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8.5)
    n_obs, n_c = _q3_size(model_out)
    ax.set_title(f"Q3 : mêmes {n_obs} observations, {n_c} pays, deux méthodes", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, "q3_coefficients.png")


def q3_observed_vs_adjusted(model_out: Path) -> Path:
    """Burnt area summed over the countries of the single panel, observed against weather-adjusted, per year."""
    bay = pd.read_parquet(model_out / "attribution_bayes_countryyear.parquet")
    freq = pd.read_parquet(model_out / "attribution_countryyear.parquet")
    freq = freq[freq["country_key"].isin(bay["country_key"].unique())]
    obs = bay.groupby("year")["burnt_area_ha"].sum() / 1e3
    adj_b = bay.groupby("year")["adjusted_area_ha"].sum() / 1e3
    adj_f = freq.groupby("year")["adjusted_area_ha"].sum() / 1e3
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(obs.index, obs.to_numpy(), color="#d9d9d9", width=0.7, label="Surface brûlée observée")
    ax.plot(adj_f.index, adj_f.to_numpy(), color=ORANGE, marker="o", ms=4, label="Ajustée de la météo, fréquentiste")
    ax.plot(adj_b.index, adj_b.to_numpy(), color=BLUE, marker="s", ms=4, label="Ajustée de la météo, bayésien")
    ax.set_xticks(list(obs.index))
    ax.set_xlabel("Année")
    ax.set_ylabel(f"Surface brûlée, somme des {_q3_size(model_out)[1]} pays (milliers d'hectares)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Q3 : surface observée et surface qu'aurait donnée une météo moyenne", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", visible=False)
    return _save(fig, "q3_observed_vs_adjusted.png")


VARIANT_LABELS = {
    "base": "Base", "year_effects": "Effets année", "linear_trend": "Tendance linéaire", "eu_only": "UE seule",
    "mediterranean_only": "Méditerranée seule", "min_area_100ha": "Sans lignes < 100 ha", "prior_n02": "A priori N(0, 2)",
}
UNIT_AXIS = {
    "days_high_season": "par 10 jours", "spei_spring": "par unité de SPEI",
    "precip_anom_season": "par 10 points de %", "temp_season": "par °C",
}


def q3_robustness(model_out: Path) -> Path:
    """Effect per physical unit of each metric under every variant, both models, with the `robust` verdict in the title."""
    r = pd.read_parquet(model_out / "attribution_robustness.parquet")
    variants = [v for v in VARIANT_LABELS if v in set(r["variant"])]
    y = np.arange(len(variants))[::-1].astype(float)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.6), sharey=True)
    for ax, m in zip(axes, METRIC_LABELS, strict=True):
        for model, color, marker, off in (("frequentist", ORANGE, "o", 0.14), ("bayesian", BLUE, "s", -0.14)):
            d = r[(r["metric"] == m) & (r["model"] == model)].set_index("variant").reindex(variants)
            ok = d["coef_per_unit"].notna().to_numpy()
            ax.errorbar(d["coef_per_unit"][ok], (y + off)[ok], xerr=[(d["coef_per_unit"] - d["ci_low_per_unit"])[ok],
                        (d["ci_high_per_unit"] - d["coef_per_unit"])[ok]], fmt=marker, color=color, ms=4, capsize=2, lw=1.1,
                        label="Fréquentiste (IC 95 %)" if model == "frequentist" else "Bayésien (HDI 95 %)")
        ax.axvline(0, color=GREY, lw=0.9)
        fl = r[(r["metric"] == m)].drop_duplicates("model").set_index("model")["robust"]
        ax.set_title(f"{METRIC_LABELS[m].splitlines()[0]}\nfréq. : {'robuste' if fl.get('frequentist') else 'non robuste'}"
                     f", bayés. : {'robuste' if fl.get('bayesian') else 'non robuste'}", fontsize=9, loc="left", color=INK)
        ax.set_xlabel(f"Effet sur le log de la surface\n{UNIT_AXIS[m]}", fontsize=8.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels=[VARIANT_LABELS[v] for v in variants])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.suptitle("Q3 : l'effet de chaque variable change-t-il quand on change de spécification ?", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return _save(fig, "q3_robustness.png")


def _models(ax: Axes, keys: list[str]) -> None:
    ax.set_yticks(range(len(keys)), labels=[Q1_MODEL_LABELS[k] for k in keys])


def q1_reliability(model_out: Path) -> Path:
    """Predicted against observed ignition rate per bin of predicted probability, test years 2022 and 2023."""
    rel = pd.read_parquet(model_out / "q1_reliability.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), sharex=True, sharey=True)
    for ax, year in zip(axes, (2022, 2023), strict=True):
        d = rel[rel["year"] == year]
        lim = [1e-4, 1.0]
        ax.plot(lim, lim, color=GREY, lw=1.0, ls="--", label="Calibration parfaite")
        for key, color in Q1_COLORS.items():
            m = d[(d["model"] == key) & (d["obs_rate"] > 0)].sort_values("bin")  # a null rate cannot sit on a log axis
            ax.plot(m["mean_pred"], m["obs_rate"], color=color, marker="o", ms=3.5, lw=1.3, label=Q1_MODEL_LABELS[key])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"Test {year}", loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Probabilité prédite moyenne du lot (échelle log)")
    axes[0].set_ylabel("Fréquence observée de départs de feu du lot (échelle log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=8.5)
    fig.suptitle("Q1 : fiabilité des probabilités, une cellule de 0,25 degré un jour donné", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "q1_reliability.png")


def q1_pr_auc(model_out: Path) -> Path:
    """PR-AUC per model and test year, with the base rate of each year."""
    m = pd.read_parquet(model_out / "q1_metrics.parquet")
    keys = list(Q1_MODEL_LABELS)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for year, color, dy in ((2022, BLUE, 0.13), (2023, ORANGE, -0.13)):
        d = m[m["year"] == year].set_index("model").loc[keys]
        ypos = np.arange(len(keys)) + dy
        ax.scatter(d["pr_auc"], ypos, color=color, s=34, zorder=3, label=f"PR-AUC {year}")
        base = float(d["base_rate"].iloc[0])
        ax.axvline(base, color=color, lw=0.9, ls=":", label=f"Taux de base {year} ({base:.4f})".replace(".", ","))
    _models(ax, keys)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("PR-AUC (échelle log, un modèle sans information vaut le taux de base)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Q1 : précision moyenne par modèle, test 2022 et 2023", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, "q1_pr_auc.png")


def q1_spatial_blocks() -> Path:
    """PR-AUC with and without holding out 2 degree spatial blocks (E1 of the spike)."""
    e1 = pd.read_csv(SPIKE_OUT / "exploratory_spatial_blocks.csv")
    e8 = pd.read_csv(SPIKE_OUT / "exploratory_spatial_blocks_clim.csv")
    keys = ["lgbm_fwi", "lgbm_fwi_space", "lgbm_full"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, year in zip(axes, (2022, 2023), strict=True):
        d = e1[e1["year"] == year].set_index("model").loc[keys]
        ypos = np.arange(len(keys))
        for y, (_, r) in zip(ypos, d.iterrows(), strict=True):
            ax.plot([r["pr_auc"], r["pr_auc_unblocked"]], [y, y], color=GREY, lw=1.4, zorder=1)
        ax.scatter(d["pr_auc_unblocked"], ypos, color=BLUE, s=38, zorder=3, label="Sans blocs (verdict)")
        ax.scatter(d["pr_auc"], ypos, color=ORANGE, s=38, zorder=3, marker="s", label="Blocs spatiaux exclus de l'entraînement")
        d8 = e8[e8["year"] == year].set_index("model").loc[keys]
        ax.scatter(d8["pr_auc"], ypos, color=GREEN, s=44, zorder=4, marker="D",
                   label="Idem, climatologie des cellules écartées remplacée par la moyenne régionale")
        ax.set_title(f"Test {year}", loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("PR-AUC")
        ax.grid(axis="y", visible=False)
    _models(axes[0], keys)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    fig.suptitle("Q1 : la compétence se transfère-t-elle à des cellules jamais vues ? (blocs de 2 degrés, 5 plis)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return _save(fig, "q1_spatial_blocks.png")


def build_all(model_out: Path | None = None, only: str | None = None) -> list[Path]:
    """Every figure, or only those of one question (only = "q1", "q2" or "q3")."""
    src = model_out or lake.GOLD_MODEL_OUT
    builders = {
        "q2": [q2_return_levels, q2_shape, q2_threshold_stability, q2_qq],
        "q3": [q3_coefficients, q3_observed_vs_adjusted, q3_robustness],
        "q1": [q1_reliability, q1_pr_auc],
    }
    todo = [only] if only else list(builders)
    out = [fn(src) for q in todo for fn in builders[q]]
    if only in (None, "q1"):
        out.append(q1_spatial_blocks())
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["q1", "q2", "q3"], help="rebuild the figures of one question only")
    for p in build_all(only=parser.parse_args().only):
        print(p.relative_to(lake.ROOT))
