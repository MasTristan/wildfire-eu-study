"""ARCO reader against a fake store: crop orientation across Greenwich,
time indexing, and identity of the daily reduction with the CDS path."""
from __future__ import annotations

import json

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr

from src.ingest import era5_arco as arco
from src.ingest import era5_daily

CODEC = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1)
IBERIA = [44.0, -10.0, 35.5, 4.5]


def _field(var: str, t: int) -> np.ndarray:
    """Deterministic global field: value encodes variable, hour, lat and lon."""
    base = {"2m_temperature": 290.0, "2m_dewpoint_temperature": 282.0, "10m_u_component_of_wind": 2.0,
            "10m_v_component_of_wind": 1.0, "total_precipitation": 0.0002}[var]
    lat = arco.LAT[:, None]
    lon = arco.LON[None, :]
    if var == "total_precipitation":
        return np.full((721, 1440), base, dtype="float32")
    return (base + 5 * np.sin(2 * np.pi * (t % 24) / 24) + 0.01 * lat + 0.001 * lon).astype("float32")


class _Resp:
    def __init__(self, status, content=b"", payload=None):
        self.status_code, self.content, self._p = status, content, payload

    def json(self):
        return self._p


class FakeStore:
    def __init__(self):
        self.calls = 0
        meta = {f"{v}/.zarray": {"chunks": [1, 721, 1440], "dtype": "<f4", "compressor": CODEC.get_config()} for v in arco.VARIABLES.values()}
        meta["latitude/.zarray"] = {"dtype": "<f4", "compressor": CODEC.get_config()}
        meta["longitude/.zarray"] = {"dtype": "<f4", "compressor": CODEC.get_config()}
        meta["time/.zarray"] = {"dtype": "<i8", "compressor": CODEC.get_config()}
        meta[".zattrs"] = {"valid_time_stop": "2026-06-30", "last_updated": "test"}
        self.meta = meta

    def __call__(self, url, timeout):
        self.calls += 1
        path = url.split(".zarr-v3/")[1]
        if path == ".zmetadata":
            return _Resp(200, payload={"metadata": self.meta})
        name, chunk = path.split("/")
        if name == "latitude":
            return _Resp(200, CODEC.encode(arco.LAT.astype("float32").tobytes()))
        if name == "longitude":
            return _Resp(200, CODEC.encode(arco.LON.astype("float32").tobytes()))
        if name == "time":
            return _Resp(200, CODEC.encode(np.arange(10, dtype="int64").tobytes()))
        t = int(chunk.split(".")[0])
        return _Resp(200, CODEC.encode(_field(name, t).tobytes()))


def test_time_index_and_store_check():
    assert arco.time_index(pd.Timestamp("1900-01-01 05:00")) == 5
    assert arco.time_index(pd.Timestamp("2022-07-15 12:00")) == 44755 * 24 + 12  # 44 755 days after 1900-01-01
    info = arco.Store(get=FakeStore()).check_store()
    assert info["valid_time_stop"] == "2026-06-30"


def test_crop_matches_a_cds_answer_across_greenwich():
    lat, lon = arco.area_coords(IBERIA)
    assert lat[0] == 44.0 and lat[-1] == 35.5 and np.all(np.diff(lat) < 0)
    assert lon[0] == -10.0 and lon[-1] == 4.5 and np.all(np.diff(lon) > 0)
    assert len(lat) == 35 and len(lon) == 59


def test_month_hourly_values_land_on_the_right_cells():
    store = arco.Store(get=FakeStore())
    ds = arco.month_hourly(store, 2022, 2, {"iberia": IBERIA}, workers=4)["iberia"]
    assert ds.sizes == {"time": 28 * 24, "latitude": 35, "longitude": 59}
    t = arco.time_index(pd.Timestamp("2022-02-01 06:00"))
    expected = _field("2m_temperature", t)
    iy = int(np.argmin(abs(arco.LAT - 40.0)))
    ix = int(np.argmin(abs(arco.LON - 356.0)))  # -4.0 degrees east
    got = float(ds["t2m"].sel(time="2022-02-01 06:00", latitude=40.0, longitude=-4.0))
    assert abs(got - float(expected[iy, ix])) < 1e-5


def test_download_month_writes_the_cds_daily_format_and_compares_identical(tmp_path):
    fake = FakeStore()
    store = arco.Store(get=fake)
    target = tmp_path / "era5_daily_iberia_2022_02.nc"
    arco.download_month(store, 2022, 2, {"iberia": IBERIA}, {"iberia": target}, workers=4)
    daily = xr.open_dataset(target)
    assert set(era5_daily.DAILY_VARS) <= set(daily.data_vars) and daily.sizes["time"] == 28
    daily.close()
    # the same hourly month through the CDS reduction gives an identical file
    hourly = arco.month_hourly(store, 2022, 2, {"iberia": IBERIA}, workers=4)["iberia"]
    ref = tmp_path / "ref.nc"
    era5_daily.hourly_to_daily(hourly).to_netcdf(ref)
    report = arco.compare_daily(ref, target)
    assert report["passes"].all(), report.to_string()
    # a perturbed file fails the tolerance
    bad = xr.open_dataset(ref).load()
    bad["t2m_c"] = bad["t2m_c"] + 0.5
    badp = tmp_path / "bad.nc"
    bad.to_netcdf(badp)
    assert not arco.compare_daily(ref, badp).set_index("variable").loc["t2m_c", "passes"]
    # resumable: nothing is fetched again
    calls = fake.calls
    arco.download_month(store, 2022, 2, {"iberia": IBERIA}, {"iberia": target}, workers=4)
    assert fake.calls == calls
    json.dumps(report.to_dict())
