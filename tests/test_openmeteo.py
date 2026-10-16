"""Open-Meteo ingest against a test double of the HTTP layer: batching,
resumability, probe fallback on rejected variables, quota stop, loading."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ingest import openmeteo as om

DAYS = pd.date_range("2022-01-01", "2022-01-10", freq="D")


class _Resp:
    def __init__(self, status: int, payload=None, text: str = ""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def _record(lat: float, lon: float, variables: list[str]) -> dict:
    daily = {"time": [d.strftime("%Y-%m-%d") for d in DAYS]}
    for v in variables:
        daily[v] = [float(i) for i in range(len(DAYS))]
    return {"latitude": lat, "longitude": lon, "daily": daily}


class FakeAPI:
    """Serves every DAILY_VARS except vapour_pressure_deficit_max; counts calls."""

    def __init__(self, quota: int | None = None):
        self.calls = 0
        self.quota = quota

    def __call__(self, url, params, timeout):
        self.calls += 1
        if self.quota is not None and self.calls > self.quota:
            return _Resp(429, text="Daily API request limit exceeded")
        variables = params["daily"].split(",")
        bad = [v for v in variables if v == "vapour_pressure_deficit_max"]
        if bad:
            return _Resp(400, text=f"Cannot initialize DailyVariable from invalid String value {bad[0]}")
        lats = [float(x) for x in params["latitude"].split(",")]
        lons = [float(x) for x in params["longitude"].split(",")]
        recs = [_record(la, lo, variables) for la, lo in zip(lats, lons, strict=True)]
        return _Resp(200, recs if len(recs) > 1 else recs[0])


def _cells(n: int) -> pd.DataFrame:
    return pd.DataFrame({"cell": range(100, 100 + n), "lat": [40.0 + 0.25 * i for i in range(n)], "lon": [-4.0] * n})


def test_probe_drops_rejected_variables():
    api = FakeAPI()
    served = om.probe(api, 40.0, -4.0, "2022-01-01", "2022-01-10")
    assert "vapour_pressure_deficit_max" not in served
    assert set(served) == set(om.DAILY_VARS) - {"vapour_pressure_deficit_max"}


def test_download_is_batched_and_resumable(tmp_path):
    api = FakeAPI()
    cells = _cells(45)
    n = om.download(cells, tmp_path, "2022-01-01", "2022-01-10", get=api, batch=20, sleep=lambda s: None)
    assert n == 3 and len(list(tmp_path.glob("batch_*.json"))) == 3
    served = json.loads((tmp_path / "variables.json").read_text())["daily"]
    assert "vapour_pressure_deficit_max" not in served
    calls_after_first = api.calls
    again = om.download(cells, tmp_path, "2022-01-01", "2022-01-10", get=api, batch=20, sleep=lambda s: None)
    assert again == 0 and api.calls == calls_after_first  # nothing re-fetched


def test_quota_stop_is_clean_and_keeps_completed_batches(tmp_path):
    api = FakeAPI(quota=4)  # probe takes 2 calls (one rejected variable), then 2 batches, then the daily limit
    with pytest.raises(SystemExit):
        om.download(_cells(100), tmp_path, "2022-01-01", "2022-01-10", get=api, batch=20, sleep=lambda s: None)
    assert len(list(tmp_path.glob("batch_*.json"))) == 2


def test_load_renames_to_feature_columns(tmp_path):
    cells = _cells(5)
    om.download(cells, tmp_path, "2022-01-01", "2022-01-10", get=FakeAPI(), batch=20, sleep=lambda s: None)
    w = om.load(tmp_path, cells)
    assert {"cell", "date", "t2m_c", "rh", "wind", "tp_mm", "et0"} <= set(w.columns)
    assert "vpd_max" not in w.columns
    assert len(w) == 5 * len(DAYS) and w["cell"].nunique() == 5
