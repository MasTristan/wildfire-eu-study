"""Partition-safe ingest paths that read what is already on disk: FIRMS
windows deduplicated and aggregated over every year, BDIFF yearly zips
parsed and deduplicated. Country attribution (network, Natural Earth) is
out of scope here and tested only where the data is."""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from src.ingest import bdiff, firms


def test_firms_dedupe_and_monthly_aggregate():
    pts = pd.DataFrame({
        "latitude": [40.0, 40.0, 41.0, 38.0],
        "longitude": [-4.0, -4.0, -3.0, 23.0],
        "acq_date": ["2023-07-01", "2023-07-01", "2023-07-20", "2023-08-02"],
        "acq_time": [1230, 1230, 1300, 1100],
        "satellite": ["N", "N", "N", "N"],
        "frp": [10.0, 10.0, 30.0, 5.0],
        "country_key": ["ES", "ES", "ES", "GR"],
    })
    dedup = firms.dedupe_detections(pts)
    assert len(dedup) == 3
    m = firms.monthly_from_points(dedup)
    es = m[(m["country_key"] == "ES")].iloc[0]
    assert es["hotspot_count"] == 2 and es["max_frp"] == 30.0 and es["mean_frp"] == 20.0
    assert str(es["month_key"])[:7] == "2023-07"
    assert set(m["source_key"]) == {"FIRMS_VIIRS"}


def _bdiff_zip(rows: list[list[str]]) -> bytes:
    header = ";".join(["Année"] + [f"c{i}" for i in range(1, len(bdiff._COLUMNS))])
    body = "\n".join([";".join(r) for r in rows])
    text = "Export BDIFF\n\n" + header + "\n" + body + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Incendies.csv", text.encode("utf-8"))
    return buf.getvalue()


def test_bdiff_rows_from_raw_dedupes_across_yearly_zips(tmp_path):
    n = len(bdiff._COLUMNS)
    row = lambda year, number: [year, number] + ["x"] * (n - 2)  # noqa: E731
    (tmp_path / "incendies_2022_2022.zip").write_bytes(_bdiff_zip([row("2022", "1"), row("2022", "2")]))
    (tmp_path / "incendies_2023_2023.zip").write_bytes(_bdiff_zip([row("2023", "1"), row("2022", "2")]))  # overlap
    rows = bdiff.rows_from_raw(tmp_path)
    assert [(r["year"], r["fire_number"]) for r in rows] == [("2022", "1"), ("2022", "2"), ("2023", "1")]
