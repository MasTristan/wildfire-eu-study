"""The feature store builders on synthetic inputs: shapes, joins and, above
all, the leakage rules (lags exclude day J, SPEI uses only dekads ended
before J, rain-7d excludes J)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src import grid
from src.features import daily

LONS = np.array([-4.0, -3.75, -3.5])
LATS = np.array([40.0, 40.25])
DATES = pd.date_range("2022-01-01", "2022-02-28", freq="D")


@pytest.fixture(scope="module")
def cells() -> pd.DataFrame:
    lon_g, lat_g = np.meshgrid(LONS, LATS)
    df = pd.DataFrame({"lon": lon_g.ravel(), "lat": lat_g.ravel(), "country_key": "ES"})
    df.insert(0, "cell", grid.cell_id(df["lon"], df["lat"]))
    return df


@pytest.fixture(scope="module")
def fwi_file(tmp_path_factory) -> list:
    rng = np.random.default_rng(0)
    data = {name: (("valid_time", "latitude", "longitude"), rng.gamma(2, 5, (len(DATES), len(LATS), len(LONS))))
            for name in daily.FWI_VARS}
    ds = xr.Dataset(data, coords={"valid_time": DATES, "latitude": LATS, "longitude": LONS})
    # one cell has no FWI at all (sea): must disappear from the store
    for name in daily.FWI_VARS:
        ds[name][:, 0, 0] = np.nan
    p = tmp_path_factory.mktemp("fwi") / "fwi_2022.nc"
    ds.to_netcdf(p)
    return [p]


@pytest.fixture(scope="module")
def static_file(tmp_path_factory) -> list:
    data = {v: (("latitude", "longitude"), np.full((len(LATS), len(LONS)), 0.5)) for v in ("cvh", "cvl", "tvh", "tvl", "lsm")}
    data["z"] = (("latitude", "longitude"), np.full((len(LATS), len(LONS)), 9.80665 * 300.0))
    ds = xr.Dataset(data, coords={"latitude": LATS, "longitude": LONS})
    p = tmp_path_factory.mktemp("static") / "static.nc"
    ds.to_netcdf(p)
    return [p]


@pytest.fixture(scope="module")
def edo_file(tmp_path_factory, cells) -> list:
    """Dekads of Jan and Feb 2022 on a tiny EPSG:3035 grid covering the cells, value = dekad index."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform(cells["lon"].to_numpy(), cells["lat"].to_numpy())
    xs = np.linspace(x.min() - 20000, x.max() + 20000, 12)
    ys = np.linspace(y.min() - 20000, y.max() + 20000, 10)
    times = pd.to_datetime(["2022-01-01", "2022-01-11", "2022-01-21", "2022-02-01", "2022-02-11", "2022-02-21"])
    vals = np.stack([np.full((len(ys), len(xs)), float(i), dtype="float32") for i in range(len(times))])
    ds = xr.Dataset({"sem03": (("time", "y", "x"), vals)}, coords={"time": times, "y": ys, "x": xs})
    p = tmp_path_factory.mktemp("edo") / "sem03_m_edo_20220101_20221221_t.nc"
    ds.to_netcdf(p)
    return [p]


def test_fwi_daily_drops_cells_without_fwi(fwi_file, cells):
    f = daily.fwi_daily(fwi_file, cells)
    assert set(f.columns) == {"cell", "date", *daily.FWI_VARS.values()}
    assert f["cell"].nunique() == len(cells) - 1


def test_static_fields_elevation(static_file, cells):
    s = daily.static_fields(static_file, cells)
    assert len(s) == len(cells) and np.allclose(s["elev_m"], 300.0)


def test_spei_uses_only_dekads_ended_before_day(edo_file, cells):
    s = daily.spei_daily(edo_file, cells, DATES)
    one = s[s["cell"] == cells["cell"].iloc[0]].set_index("date")["spei3"]
    # dekad 0 spans Jan 1-10 (ends Jan 10): unknown on Jan 10, known from Jan 11
    assert np.isnan(one.loc["2022-01-10"])
    assert one.loc["2022-01-11"] == 0.0
    # dekad 2 (Jan 21-31) becomes known on Feb 1, dekad 3 (Feb 1-10) on Feb 11
    assert one.loc["2022-01-31"] == 1.0 and one.loc["2022-02-01"] == 2.0 and one.loc["2022-02-11"] == 3.0


def test_lags_exclude_day_j(cells):
    c = cells["cell"].iloc[1]
    events = pd.DataFrame({
        "firedate_key": pd.to_datetime(["2022-01-15", "2022-01-15", "2021-12-30"]),
        "centroid_lon": [LONS[1], LONS[1], LONS[1]], "centroid_lat": [LATS[0], LATS[0], LATS[0]],
    })
    labels = daily.ignition_labels(events, cells)
    assert labels.loc[labels["date"] == "2022-01-15", "n_start"].item() == 2
    lags = daily.lag_counts(labels, cells, DATES, pd.Timestamp("2021-12-01"), tag="a")
    one = lags[lags["cell"] == c].set_index("date")
    assert one.loc["2022-01-15", "laga_7"] == 0      # day J itself never counts
    assert one.loc["2022-01-16", "laga_7"] == 2
    assert one.loc["2022-01-22", "laga_7"] == 2 and one.loc["2022-01-23", "laga_7"] == 0
    assert one.loc["2022-01-01", "laga_30"] == 1     # history before the year is used


def test_assemble_joins_and_excludes_rain_of_day_j(fwi_file, static_file, edo_file, cells):
    fwi = daily.fwi_daily(fwi_file, cells)
    weather = pd.DataFrame({
        "cell": np.tile(cells["cell"].to_numpy(), len(DATES)),
        "date": np.repeat(DATES.to_numpy(), len(cells)),
        "t2m_c": 10.0, "tp_mm": 1.0,
    })
    spei = daily.spei_daily(edo_file, cells, DATES)
    static = daily.static_fields(static_file, cells)
    events = pd.DataFrame({"firedate_key": pd.to_datetime(["2022-02-10"]), "centroid_lon": [LONS[2]], "centroid_lat": [LATS[1]]})
    labels = daily.ignition_labels(events, cells)
    lags = daily.lag_counts(labels, cells, DATES, pd.Timestamp("2021-01-01"), tag="a")
    df = daily.assemble(cells, DATES, fwi, weather, spei, static, labels, lags)
    assert df["cell"].nunique() == len(cells) - 1           # the sea cell is gone
    assert int(df["y_a"].sum()) == 1
    one = df[df["cell"] == cells["cell"].iloc[2]].set_index("date")
    assert np.isnan(one.loc["2022-01-02", "tp_7d"])         # fewer than 3 past days
    assert one.loc["2022-01-10", "tp_7d"] == 7.0            # days 3..9, never day 10
    assert df.select_dtypes("float64").empty
    assert {"doy", "year", "laga_7", "laga_30", "spei3", "elev_m", "fwi", "dsr"} <= set(df.columns)
