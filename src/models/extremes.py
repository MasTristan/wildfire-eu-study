"""Q2, tail of the fire-size distribution: peaks over threshold with a
generalised Pareto (GPD) fit on declustered episodes, return levels and
block-bootstrap intervals.

Inputs are fire sizes in hectares (fact_fireevent, EFFIS rows). Everything
is per country and reported with its diagnostics: the threshold, the number of
episodes above it, the shape parameter and its interval, and the empirical
return levels next to the fitted ones, so a reader can see when the fit is
doing work the data cannot support.

Conventions:
- threshold u: an absolute size (THRESHOLD_HA = 500 ha), the same for every
  country. A quantile of the sizes would move with the EFFIS mapping floor
  (15 to 24 ha before 2020, 0 to 1 ha after), which mixes in thousands of small
  fires; the yearly count of fires above 500 ha does not depend on that floor;
- episodes: fires above u in the same country whose dates are at most
  DECLUSTER_DAYS apart (chained: each fire is compared with the previous
  exceedance) form one episode, represented by its largest fire. The GPD is
  fitted on the episodes, not on the fires, because exceedances that follow one
  another within days are one weather event and are not independent;
- rate: episodes per year over the observed span, counted in calendar years:
  from the country's first record of any size to the last complete year of the
  source, inclusive, so a year without a large fire still counts as observed.
  Only complete years enter: the data end is the latest record of the whole
  input, and a final year that ends before December is dropped (the 2026
  extraction stops at the end of July, before most of the southern season);
- return level x_T: the size exceeded on average once every T years,
  x_T = u + sigma / xi * ((T * rate)^xi - 1), the Gumbel limit when xi ~ 0;
- intervals: percentile intervals from a bootstrap by blocks of years: the
  observed years are resampled with replacement, which keeps the dependence
  inside a year and propagates the uncertainty on the episode rate as well as
  on the shape and scale;
- publication of a level: only when the point estimate of xi is below 1 (at
  xi >= 1 the mean excess is infinite and a level is not a stable quantity),
  and only up to MAX_EXTRAPOLATION times the observed span. A level that is not
  published is null and a boolean column says why;
- published perimeter: EU27 countries with at least MIN_YEARS complete years
  and MIN_EPISODES episodes above the threshold. The other countries stay in the
  table with in_scope = False and a reason, and with null estimates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

RETURN_PERIODS = (2, 5, 10, 20, 50)
MAX_EXTRAPOLATION = 2.0  # published return periods stay within this multiple of the observed span
THRESHOLD_HA = 500.0
DECLUSTER_DAYS = 3
MIN_YEARS = 8
MIN_EPISODES = 30
STABILITY_THRESHOLDS_HA = (200.0, 300.0, 500.0, 1000.0)
STABILITY_MIN_EPISODES = 20
# Hard-coded on purpose: dim_country.is_eu is not a safe source (Germany is false there).
EU27 = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU",
    "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
})


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


def decluster_episodes(events: pd.DataFrame, group: str = "country_key", threshold_ha: float = THRESHOLD_HA,
                       gap_days: int = DECLUSTER_DAYS) -> pd.DataFrame:
    """Fires above the threshold merged into episodes, one row per episode.

    Within a group, consecutive exceedances at most gap_days apart (compared with
    the previous exceedance, so a chain of fires every two days is one episode)
    share an episode; the row kept is the largest fire of the episode, with its
    date and year, the number of fires merged and the first and last dates."""
    cols = [group, "date", "year", "area_ha", "n_fires", "start_date", "end_date"]
    ev = events.dropna(subset=["area_ha", "firedate_key"])
    ev = ev.loc[ev["area_ha"] > threshold_ha, [group, "area_ha", "firedate_key"]].copy()
    if ev.empty:
        return pd.DataFrame(columns=cols)
    ev["date"] = pd.to_datetime(ev["firedate_key"]).dt.normalize()
    ev = ev.sort_values([group, "date"], kind="stable").reset_index(drop=True)
    prev = ev.groupby(group)["date"].shift()
    new_episode = prev.isna() | ((ev["date"] - prev).dt.days > gap_days)
    ev["episode"] = new_episode.cumsum()
    top = ev.loc[ev.groupby("episode")["area_ha"].idxmax(), [group, "date", "area_ha", "episode"]]
    agg = ev.groupby("episode")["date"].agg(n_fires="size", start_date="min", end_date="max")
    out = top.merge(agg, left_on="episode", right_index=True)
    out["year"] = out["date"].dt.year.astype(int)
    return out[cols].sort_values([group, "date"]).reset_index(drop=True)


def _block_bootstrap(ep_year: np.ndarray, ep_excess: np.ndarray, years: np.ndarray, threshold: float, n_boot: int,
                     seed: int, periods=RETURN_PERIODS, min_fit: int = 10) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Shapes and return levels of n_boot replicates, each built from the same number
    of years drawn with replacement (a year brings all its episodes, an empty year
    brings none). A replicate with fewer than min_fit episodes or a failed fit is nan."""
    rng = np.random.default_rng(seed)
    n_years = len(years)
    by_year = [ep_excess[ep_year == y] for y in years]
    shapes = np.full(n_boot, np.nan)
    levels = {T: np.full(n_boot, np.nan) for T in periods}
    for b in range(n_boot):
        draw = rng.integers(0, n_years, n_years)
        pooled = np.concatenate([by_year[i] for i in draw])
        if len(pooled) < min_fit:
            continue
        try:
            shape, scale = fit_gpd(pooled)
        except Exception:  # scipy can fail to converge on degenerate resamples
            continue
        fit = GpdFit(threshold, len(pooled), len(pooled) / n_years, shape, scale)
        shapes[b] = shape
        for T in periods:
            levels[T][b] = fit.return_level(T)
    return shapes, levels


def bootstrap_fit(episodes: pd.DataFrame, threshold: float, years: np.ndarray, n_boot: int = 500, seed: int = 42,
                  periods=RETURN_PERIODS) -> pd.DataFrame:
    """Percentile intervals on the shape and on each return level, by resampling the observed years.

    episodes holds the declustered episodes of one group (columns year and area_ha);
    years lists every observed year, including those without an episode."""
    ep_year = episodes["year"].to_numpy()
    excess = episodes["area_ha"].to_numpy(dtype="float64") - threshold
    shapes, levels = _block_bootstrap(ep_year, excess, np.asarray(years), threshold, n_boot, seed, periods)
    rows = [{"quantity": "shape", "p2.5": np.nanpercentile(shapes, 2.5), "median": np.nanmedian(shapes),
             "p97.5": np.nanpercentile(shapes, 97.5)}]
    for T in periods:
        arr = levels[T]
        rows.append({"quantity": f"rl_{T}y", "p2.5": np.nanpercentile(arr, 2.5), "median": np.nanmedian(arr),
                     "p97.5": np.nanpercentile(arr, 97.5)})
    return pd.DataFrame(rows)


def empirical_return_levels(sizes: np.ndarray, years: float, periods=RETURN_PERIODS) -> dict[int, float]:
    """Model-free counterpart: the size whose exceedance count over the span is about years/T."""
    s = np.sort(np.asarray(sizes, dtype="float64"))[::-1]
    out = {}
    for T in periods:
        k = int(round(years / T))  # expected number of exceedances over the span
        out[T] = float(s[k - 1]) if 1 <= k <= len(s) else float("nan")
    return out


def last_complete_year_of(data_end: pd.Timestamp) -> int:
    """Latest calendar year fully covered by data ending at data_end."""
    return int(data_end.year) if data_end.month == 12 else int(data_end.year) - 1


def _prepare(events: pd.DataFrame, group: str, data_end: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.Series, int]:
    """Fires of the complete years, the first year of record of each group and the last complete year."""
    ev = events.dropna(subset=["area_ha", "firedate_key"])
    end = pd.Timestamp(data_end) if data_end is not None else pd.to_datetime(ev["firedate_key"]).max()
    last_complete_year = last_complete_year_of(end)
    ev = ev[pd.to_datetime(ev["firedate_key"]).dt.year <= last_complete_year]
    # the period runs from the group's first record of any size (its coverage start)
    # to the source's last complete year, not to the group's last large fire:
    # years without a large fire are observed years, and dropping them would inflate the rate
    first_year = pd.to_datetime(ev["firedate_key"]).dt.year.groupby(ev[group]).min()
    return ev, first_year, last_complete_year


def _scope(key, years: float, n_episodes: int, eu_members, min_years: int, min_episodes: int) -> str:
    if eu_members is not None and key not in eu_members:
        return "not_eu27"
    if years < min_years:
        return "years_below_min"
    if n_episodes < min_episodes:
        return "episodes_below_min"
    return "in_scope"


def tail_risk_columns(group: str = "country_key", periods=RETURN_PERIODS) -> list[str]:
    cols = [group, "n_fires", "first_year", "last_year", "years", "largest_ha", "max_return_period_years"]
    cols += [f"empirical_rl_{T}y" for T in periods]
    cols += ["threshold_ha", "n_exceedances", "in_scope", "scope_reason", "rate_per_year", "shape", "scale",
             "shape_p2.5", "shape_p97.5", "shape_below_one"]
    for T in periods:
        cols += [f"rl_{T}y", f"rl_{T}y_p2.5", f"rl_{T}y_p97.5"]
    cols += ["return_period_10k_ha_years"]
    return cols


def tail_risk_table(events: pd.DataFrame, group: str = "country_key", threshold_ha: float = THRESHOLD_HA,
                    decluster_days: int = DECLUSTER_DAYS, min_years: int = MIN_YEARS,
                    min_episodes: int = MIN_EPISODES, n_boot: int = 500, periods=RETURN_PERIODS,
                    max_extrapolation: float = MAX_EXTRAPOLATION, data_end: pd.Timestamp | None = None,
                    eu_members: frozenset[str] | None = EU27, seed: int = 42) -> pd.DataFrame:
    """One row per group seen in the input (see the module docstring for the rules).

    n_fires counts the fires above the threshold before declustering; n_exceedances
    counts the episodes, the sample of the GPD fit. Groups outside the published
    perimeter are listed with in_scope = False, the reason in scope_reason and null
    estimates, so the report shows what was left out. Levels beyond the horizon
    max_extrapolation times the observed span, or all levels when the point shape is
    1 or more (shape_below_one false), are null."""
    ev, first_year, last_complete_year = _prepare(events, group, data_end)
    episodes = decluster_episodes(ev, group, threshold_ha, decluster_days)
    raw = ev.loc[ev["area_ha"] > threshold_ha].groupby(group).size()
    rows = []
    for key in sorted(first_year.index):
        start = int(first_year[key])
        years = float(last_complete_year - start + 1)
        ep = episodes[episodes[group] == key]
        sizes = ep["area_ha"].to_numpy(dtype="float64")
        horizon = max_extrapolation * years
        n_ep = int(len(ep))
        scope = _scope(key, years, n_ep, eu_members, min_years, min_episodes)
        row: dict = {group: key, "n_fires": int(raw.get(key, 0)), "first_year": start, "last_year": last_complete_year,
                     "years": round(years, 2), "largest_ha": float(ev.loc[ev[group] == key, "area_ha"].max()),
                     "max_return_period_years": round(horizon, 2), "threshold_ha": float(threshold_ha),
                     "n_exceedances": n_ep, "in_scope": scope == "in_scope", "scope_reason": scope}
        emp = empirical_return_levels(sizes, years, periods)
        for T in periods:
            row[f"empirical_rl_{T}y"] = emp[T]
        if scope != "in_scope":
            rows.append(row)
            continue
        fit = fit_pot(sizes, threshold_ha, years)
        boot = bootstrap_fit(ep, threshold_ha, np.arange(start, last_complete_year + 1), n_boot=n_boot, seed=seed,
                             periods=periods).set_index("quantity")
        below_one = bool(fit.shape < 1.0)
        row.update({"rate_per_year": fit.rate_per_year, "shape": fit.shape, "scale": fit.scale,
                    "shape_p2.5": boot.loc["shape", "p2.5"], "shape_p97.5": boot.loc["shape", "p97.5"],
                    "shape_below_one": below_one})
        nan = float("nan")
        for T in periods:
            published = below_one and T <= horizon
            row[f"rl_{T}y"] = fit.return_level(T) if published else nan
            row[f"rl_{T}y_p2.5"] = boot.loc[f"rl_{T}y", "p2.5"] if published else nan
            row[f"rl_{T}y_p97.5"] = boot.loc[f"rl_{T}y", "p97.5"] if published else nan
        rp = fit.return_period(10_000.0)
        row["return_period_10k_ha_years"] = rp if (below_one and rp <= horizon) else nan
        rows.append(row)
    out = pd.DataFrame(rows).reindex(columns=tail_risk_columns(group, periods))
    out["in_scope"] = out["in_scope"].astype(bool)
    out["shape_below_one"] = out["shape_below_one"].astype("boolean")
    return out


def threshold_stability(events: pd.DataFrame, groups: list[str] | None = None, group: str = "country_key",
                        thresholds=STABILITY_THRESHOLDS_HA, decluster_days: int = DECLUSTER_DAYS,
                        n_boot: int = 200, min_episodes: int = STABILITY_MIN_EPISODES,
                        data_end: pd.Timestamp | None = None, seed: int = 42) -> pd.DataFrame:
    """Shape and modified scale of the declustered GPD fit over candidate thresholds, per group,
    with a block-bootstrap interval on the shape. A flat shape across thresholds means the GPD
    regime is reached; this is the diagnostic a reader of the report expects. Episodes are
    rebuilt at each threshold (declustering depends on which fires exceed it). A threshold with
    fewer than min_episodes episodes keeps its row with null estimates."""
    ev, first_year, last_complete_year = _prepare(events, group, data_end)
    keys = sorted(first_year.index) if groups is None else [k for k in sorted(groups) if k in first_year.index]
    rows = []
    for u in thresholds:
        episodes = decluster_episodes(ev, group, u, decluster_days)
        raw = ev.loc[ev["area_ha"] > u].groupby(group).size()
        for key in keys:
            start = int(first_year[key])
            years = last_complete_year - start + 1
            ep = episodes[episodes[group] == key]
            row: dict = {group: key, "threshold_ha": float(u), "n_fires": int(raw.get(key, 0)),
                         "n_exceedances": int(len(ep)), "rate_per_year": len(ep) / years}
            if len(ep) >= min_episodes:
                sizes = ep["area_ha"].to_numpy(dtype="float64")
                shape, scale = fit_gpd(sizes - u)
                shapes, _ = _block_bootstrap(ep["year"].to_numpy(), sizes - u, np.arange(start, last_complete_year + 1),
                                             u, n_boot, seed, periods=())
                row.update({"shape": shape, "shape_p2.5": np.nanpercentile(shapes, 2.5),
                            "shape_p97.5": np.nanpercentile(shapes, 97.5), "scale": scale,
                            "modified_scale": scale - shape * u})
            rows.append(row)
    cols = [group, "threshold_ha", "n_fires", "n_exceedances", "rate_per_year", "shape", "shape_p2.5", "shape_p97.5",
            "scale", "modified_scale"]
    return pd.DataFrame(rows).reindex(columns=cols)


def return_levels_long(table: pd.DataFrame, group: str = "country_key", periods=RETURN_PERIODS) -> pd.DataFrame:
    """The return levels of tail_risk_table in long form, one row per group and
    period: the shape a report needs to draw a return-level curve with its band.
    Periods beyond the reliable horizon keep their row with null levels, so a
    curve stops visibly where the data stop. within_horizon and shape_below_one
    say why a level is null; published is true when both hold."""
    rows = []
    for _, r in table.iterrows():
        below_one = r.get("shape_below_one", pd.NA)
        below_one = bool(below_one) if not pd.isna(below_one) else False
        for T in periods:
            within = bool(T <= r["max_return_period_years"])
            rows.append({
                group: r[group], "period_years": int(T),
                "return_level_ha": r.get(f"rl_{T}y", np.nan),
                "return_level_p2_5": r.get(f"rl_{T}y_p2.5", np.nan),
                "return_level_p97_5": r.get(f"rl_{T}y_p97.5", np.nan),
                "empirical_level_ha": r.get(f"empirical_rl_{T}y", np.nan),
                "within_horizon": within,
                "shape_below_one": below_one,
                "published": within and below_one,
            })
    return pd.DataFrame(rows)
