"""Q2, tail of the fire-size distribution: peaks over threshold with a
generalised Pareto (GPD) fit, return levels and bootstrap intervals.

Inputs are fire sizes in hectares (fact_fireevent, EFFIS rows). Everything
is per group (country or macro-region) and reported with its diagnostics:
the threshold used, the number of exceedances, the shape parameter and its
interval, and the empirical return levels next to the fitted ones, so a
reader can see when the fit is doing work the data cannot support.

Conventions:
- threshold u: a quantile of the sizes above the detection floor, chosen
  per group; the stability table over a grid of candidate thresholds is
  returned for the report rather than hidden;
- rate: exceedances per year over the observed span;
- return level x_T: the size exceeded on average once every T years,
  x_T = u + sigma / xi * ((T * rate)^xi - 1), the Gumbel limit when xi ~ 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

RETURN_PERIODS = (2, 5, 10, 20, 50)


@dataclass(frozen=True)
class GpdFit:
    threshold: float
    n_exceedances: int
    rate_per_year: float
    shape: float
    scale: float

    def return_level(self, period_years: float) -> float:
        m = period_years * self.rate_per_year
        if m <= 1:
            return float("nan")
        if abs(self.shape) < 1e-6:
            return self.threshold + self.scale * np.log(m)
        return self.threshold + self.scale / self.shape * (m**self.shape - 1.0)

    def return_period(self, size: float) -> float:
        """Years between fires at least this large (inverse of return_level)."""
        if size <= self.threshold:
            return float("nan")
        y = size - self.threshold
        if abs(self.shape) < 1e-6:
            m = np.exp(y / self.scale)
        else:
            base = 1.0 + self.shape * y / self.scale
            if base <= 0:
                return float("inf")
            m = base ** (1.0 / self.shape)
        return float(m / self.rate_per_year)


def fit_gpd(excesses: np.ndarray) -> tuple[float, float]:
    """Maximum-likelihood GPD fit of the excesses above the threshold (location fixed at 0)."""
    x = np.asarray(excesses, dtype="float64")
    x = x[x > 0]
    shape, _, scale = stats.genpareto.fit(x, floc=0.0)
    return float(shape), float(scale)


def fit_pot(sizes: np.ndarray, threshold: float, years: float) -> GpdFit:
    s = np.asarray(sizes, dtype="float64")
    exc = s[s > threshold] - threshold
    shape, scale = fit_gpd(exc)
    return GpdFit(float(threshold), int(len(exc)), float(len(exc) / years), shape, scale)


def threshold_stability(sizes: np.ndarray, quantiles=(0.80, 0.85, 0.90, 0.95, 0.975), years: float = 1.0) -> pd.DataFrame:
    """Shape and modified scale over candidate thresholds: a flat shape means the
    GPD regime is reached; this is the diagnostic a reader of the report expects."""
    s = np.asarray(sizes, dtype="float64")
    rows = []
    for q in quantiles:
        u = float(np.quantile(s, q))
        exc = s[s > u] - u
        if len(exc) < 20:
            continue
        shape, scale = fit_gpd(exc)
        rows.append({"quantile": q, "threshold": u, "n_exceedances": int(len(exc)), "shape": shape,
                     "scale": scale, "modified_scale": scale - shape * u})
    return pd.DataFrame(rows)


def bootstrap_fit(sizes: np.ndarray, threshold: float, years: float, n_boot: int = 500, seed: int = 42,
                  periods=RETURN_PERIODS) -> pd.DataFrame:
    """Percentile intervals on the shape and on each return level, by resampling the exceedances."""
    rng = np.random.default_rng(seed)
    s = np.asarray(sizes, dtype="float64")
    exc = s[s > threshold] - threshold
    n = len(exc)
    shapes: list[float] = []
    levels: dict[float, list[float]] = {T: [] for T in periods}
    for _ in range(n_boot):
        sample = rng.choice(exc, n, replace=True)
        shape, scale = fit_gpd(sample)
        fit = GpdFit(threshold, n, n / years, shape, scale)
        shapes.append(shape)
        for T in periods:
            levels[T].append(fit.return_level(T))
    rows = [{"quantity": "shape", "p2.5": np.percentile(shapes, 2.5), "median": np.median(shapes), "p97.5": np.percentile(shapes, 97.5)}]
    for T in periods:
        arr = np.array(levels[T])
        rows.append({"quantity": f"rl_{T}y", "p2.5": np.nanpercentile(arr, 2.5), "median": np.nanmedian(arr), "p97.5": np.nanpercentile(arr, 97.5)})
    return pd.DataFrame(rows)


def empirical_return_levels(sizes: np.ndarray, years: float, periods=RETURN_PERIODS) -> dict[int, float]:
    """Model-free counterpart: the size whose exceedance count over the span is about years/T."""
    s = np.sort(np.asarray(sizes, dtype="float64"))[::-1]
    out = {}
    for T in periods:
        k = int(round(years / T))  # expected number of exceedances over the span
        out[T] = float(s[k - 1]) if 1 <= k <= len(s) else float("nan")
    return out


def tail_risk_table(events: pd.DataFrame, group: str = "country_key", floor_ha: float = 30.0,
                    threshold_quantile: float = 0.90, min_exceedances: int = 30, n_boot: int = 500,
                    periods=RETURN_PERIODS) -> pd.DataFrame:
    """One row per group: threshold, exceedances, rate, GPD parameters with
    intervals, fitted and empirical return levels. Groups with too few
    exceedances are listed with null estimates rather than dropped, so the
    report shows where the tail cannot be estimated."""
    ev = events.dropna(subset=["area_ha", "firedate_key"])
    ev = ev[ev["area_ha"] >= floor_ha]
    rows = []
    for key, g in ev.groupby(group):
        dates = pd.to_datetime(g["firedate_key"])
        years = max((dates.max() - dates.min()).days / 365.25, 1.0)
        sizes = g["area_ha"].to_numpy(dtype="float64")
        row: dict = {group: key, "n_fires": int(len(sizes)), "years": round(years, 2), "largest_ha": float(sizes.max())}
        emp = empirical_return_levels(sizes, years, periods)
        for T in periods:
            row[f"empirical_rl_{T}y"] = emp[T]
        u = float(np.quantile(sizes, threshold_quantile))
        n_exc = int((sizes > u).sum())
        row.update({"threshold_ha": u, "n_exceedances": n_exc})
        if n_exc < min_exceedances:
            rows.append(row)
            continue
        fit = fit_pot(sizes, u, years)
        boot = bootstrap_fit(sizes, u, years, n_boot=n_boot, periods=periods).set_index("quantity")
        row.update({"rate_per_year": fit.rate_per_year, "shape": fit.shape, "scale": fit.scale,
                    "shape_p2.5": boot.loc["shape", "p2.5"], "shape_p97.5": boot.loc["shape", "p97.5"]})
        for T in periods:
            row[f"rl_{T}y"] = fit.return_level(T)
            row[f"rl_{T}y_p2.5"] = boot.loc[f"rl_{T}y", "p2.5"]
            row[f"rl_{T}y_p97.5"] = boot.loc[f"rl_{T}y", "p97.5"]
        row["return_period_10k_ha_years"] = fit.return_period(10_000.0)
        rows.append(row)
    return pd.DataFrame(rows)
