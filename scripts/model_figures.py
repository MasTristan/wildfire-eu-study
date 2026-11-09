"""Figures of the model cards (docs/models/*.md), rebuilt from the model outputs.

Reads `gold/model_out` (Q2 tail risk, Q3 attribution, Q1 report tables) and the
spike's committed outputs (`notebooks/spike_q1/out`), writes PNG files to
`docs/img/models/`. Nothing is recomputed: every plotted number is a cell of
one of those tables, so a figure and the card that cites it cannot disagree.

    uv run python scripts/model_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

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

# Q2: the four countries with the largest return levels among the well-fitted ones
Q2_COUNTRIES = ["PT", "ES", "GR", "IT"]
COUNTRY_NAMES = {"PT": "Portugal", "ES": "Espagne", "GR": "Grèce", "IT": "Italie"}

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
    """Fitted return level with its 95 % bootstrap band, empirical level as points, per country."""
    levels = pd.read_parquet(model_out / "tail_risk_levels.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
    for ax, key in zip(axes.ravel(), Q2_COUNTRIES, strict=True):
        d = levels[(levels["country_key"] == key) & levels["return_level_ha"].notna()]
        x = d["period_years"].to_numpy()
        ax.fill_between(x, d["return_level_p2_5"], d["return_level_p97_5"], color=BLUE, alpha=0.18, lw=0,
                        label="Intervalle à 95 % (bootstrap)")
        ax.plot(x, d["return_level_ha"], color=BLUE, marker="o", ms=4, label="Niveau de retour ajusté")
        emp = levels[(levels["country_key"] == key) & levels["empirical_level_ha"].notna()]
        ax.scatter(emp["period_years"], emp["empirical_level_ha"], color=ORANGE, zorder=3, s=26,
                   label="Niveau observé (empirique)")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20], labels=["2", "5", "10", "20"])
        ax.minorticks_off()
        ax.set_title(COUNTRY_NAMES[key], loc="left", fontsize=10.5, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Période de retour (années)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Taille du feu (hectares, échelle log)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Q2 : taille du feu dépassée en moyenne une fois par période de retour", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, "q2_return_levels.png")


def q2_shape(model_out: Path) -> Path:
    """GPD shape parameter with its 95 % interval for every country whose tail could be fitted."""
    t = pd.read_parquet(model_out / "tail_risk_country.parquet")
    t = t[t["shape"].notna()].sort_values("shape").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    colors = [BLUE if k in Q2_COUNTRIES else SKY for k in t["country_key"]]
    ax.barh(t["country_key"], t["shape"], color=colors, height=0.62)
    ax.errorbar(t["shape"], np.arange(len(t)), xerr=[t["shape"] - t["shape_p2.5"], t["shape_p97.5"] - t["shape"]],
                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5)
    ax.axvline(0, color=GREY, lw=0.9)
    ax.axvline(1, color=ORANGE, lw=1.1, ls="--")
    ax.text(0.99, 1.0, "forme = 1 : moyenne infinie", color=ORANGE, ha="right", va="bottom", fontsize=8.5,
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("Paramètre de forme de la loi de Pareto généralisée (sans unité)")
    ax.set_ylabel("Pays (code ISO)")
    ax.set_title("Q2 : forme de la queue par pays, intervalle à 95 % (bootstrap)", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, "q2_shape.png")


def _bayes_pop(model_out: Path) -> pd.DataFrame:
    b = pd.read_parquet(model_out / "attribution_bayes_coefficients.parquet")
    pop = b[b["level"] == "population"].copy()
    pop["label"] = pop["metric"].map(METRIC_LABELS)
    return pop.reset_index(drop=True)


def q3_coefficients(model_out: Path) -> Path:
    """Population slope per metric: frequentist estimate (95 % interval) against Bayesian posterior (94 % HDI)."""
    pop = _bayes_pop(model_out)
    y = np.arange(len(pop))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.errorbar(pop["freq_coef"], y + 0.12, xerr=[pop["freq_coef"] - pop["freq_ci_low"], pop["freq_ci_high"] - pop["freq_coef"]],
                fmt="o", color=ORANGE, capsize=3, label="Fréquentiste (MCO, intervalle à 95 %)")
    ax.errorbar(pop["mean"], y - 0.12, xerr=[pop["mean"] - pop["hdi_low"], pop["hdi_high"] - pop["mean"]],
                fmt="s", color=BLUE, capsize=3, label="Bayésien hiérarchique (moyenne, HDI à 94 %)")
    ax.axvline(0, color=GREY, lw=0.9)
    ax.set_yticks(y, labels=pop["label"])
    ax.set_xlabel("Effet sur le log de la surface brûlée, par écart-type de la variable (écart-type du panel entier)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8.5)
    ax.set_title("Q3 : mêmes 356 observations, 40 pays, deux méthodes", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, "q3_coefficients.png")


def q3_observed_vs_adjusted(model_out: Path) -> Path:
    """Burnt area summed over the 40 countries of both fits, observed against weather-adjusted, per year."""
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
    ax.set_ylabel("Surface brûlée, somme des 40 pays (milliers d'hectares)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Q3 : surface observée et surface qu'aurait donnée une météo moyenne", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", visible=False)
    return _save(fig, "q3_observed_vs_adjusted.png")


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


def build_all(model_out: Path | None = None) -> list[Path]:
    src = model_out or lake.GOLD_MODEL_OUT
    return [
        q2_return_levels(src), q2_shape(src),
        q3_coefficients(src), q3_observed_vs_adjusted(src),
        q1_reliability(src), q1_pr_auc(src), q1_spatial_blocks(),
    ]


if __name__ == "__main__":
    for p in build_all():
        print(p.relative_to(lake.ROOT))
