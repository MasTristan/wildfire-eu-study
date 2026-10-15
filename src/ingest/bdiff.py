"""BDIFF ingest: France forest-fire register (commune grain, 2006+).

BDIFF (bdiff.agriculture.gouv.fr) exposes no documented JSON API. The site is a
Symfony form (`name="if"`, method GET) whose fields set server-side session
state; `/incendies/zip` then exports whatever the last search in that session
matched. Verified live 2026-07-30 by driving the form directly:

    GET /incendies?if[periodeAnnees][anneeDeb]=2020&if[periodeAnnees][anneeFin]=2021&if[fr]=1
    GET /incendies/zip   (same cookie jar)  -> zip scoped to 2020-2021 (5333 fires)

Both requests must share a `requests.Session` (cookie-based state). The zip
contains `Incendies.csv` (semicolon-delimited, UTF-8) with 3-4 metadata lines
before the header row; the header is located by scanning for the row that
starts with "Année".

Run:
    python src/ingest/bdiff.py                  # 2006 -> current year
    python src/ingest/bdiff.py 2016 2020         # explicit year range
"""
from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

_TIMEOUT = 60
# The zip export refuses beyond 30000 rows ("Le nombre de resultats maximum a
# ete atteint... le maximum est de 30000"), verified live 2026-07-30. Batches
# are split by year to stay under this with margin.
_MAX_EXPORT_ROWS = 28000

# Position-based column map: BDIFF's header carries accented French text whose
# byte encoding is unreliable across tools, so we match by index, not name.
_COLUMNS = [
    "year", "fire_number", "department_code", "commune_insee", "commune_name",
    "alert_datetime", "area_total_m2", "area_forest_m2", "area_maquis_m2",
    "area_other_natural_m2", "area_agri_m2", "area_other_m2",
    "area_other_wooded_m2", "area_nonwooded_natural_m2",
    "area_nonwooded_artificial_m2", "area_nonwooded_m2", "area_precision",
    "stand_type_code", "cause_raw", "casualties_flag", "deaths_n",
    "buildings_destroyed_n", "buildings_damaged_n", "data_precision",
]


def _session_search(annee_deb: int, annee_fin: int) -> tuple[requests.Session, int]:
    """Run the search and return (session, result_count) so callers can decide
    whether the range needs splitting before hitting /zip (30k row cap)."""
    sess = requests.Session()
    sess.get(f"{config.BDIFF_HOST}/incendies", timeout=_TIMEOUT)
    params: dict[str, str | int] = {
        "if[periodeAnnees][anneeDeb]": annee_deb,
        "if[periodeAnnees][anneeFin]": annee_fin,
        "if[fr]": 1,
        "if[submit]": "",
    }
    resp = sess.get(f"{config.BDIFF_HOST}/incendies", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    m = re.search(r"(\d[\d\s]*) incendies", resp.text)
    count = int(m.group(1).replace(" ", "")) if m else -1
    return sess, count


def fetch_zip(annee_deb: int, annee_fin: int) -> bytes:
    sess, _ = _session_search(annee_deb, annee_fin)
    resp = sess.get(f"{config.BDIFF_HOST}/incendies/zip", timeout=_TIMEOUT)
    resp.raise_for_status()
    if resp.headers.get("content-type", "").startswith("text/html"):
        raise RuntimeError(
            f"BDIFF export for {annee_deb}-{annee_fin} returned HTML, not a zip "
            f"(likely over the 30000-row export cap)"
        )
    return resp.content


def iter_year_batches(deb: int, fin: int):
    """Yield (deb, fin) sub-ranges that each fit under the zip export cap,
    bisecting on result count (checked via the search page, cheap) rather than
    guessing a fixed batch size."""
    _, count = _session_search(deb, fin)
    if count < 0:
        raise RuntimeError(f"BDIFF: could not read result count for {deb}-{fin}")
    if count <= _MAX_EXPORT_ROWS or deb == fin:
        yield deb, fin
        return
    mid = (deb + fin) // 2
    yield from iter_year_batches(deb, mid)
    yield from iter_year_batches(mid + 1, fin)


def parse_incendies_csv(raw_bytes: bytes) -> list[dict]:
    """Locate the header row (starts with the 'year' column, spelled with the
    accented BDIFF header in practice) and parse fixed-position columns."""
    text = raw_bytes.decode("utf-8")
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:10]):
        first_field = line.split(";", 1)[0].strip()
        if first_field and first_field[0].isdigit() is False and len(first_field) < 12:
            # Candidate header row: first cell is a short non-numeric token
            # ("Année"). Confirm by field count.
            if len(next(csv.reader([line], delimiter=";"))) == len(_COLUMNS):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("BDIFF CSV: header row not found in first 10 lines")

    rows = []
    for row in csv.reader(lines[header_idx + 1:], delimiter=";"):
        if not row or not row[0].strip():
            continue
        if len(row) != len(_COLUMNS):
            continue  # malformed/trailing line, skip rather than guess
        rows.append(dict(zip(_COLUMNS, row, strict=False)))
    return rows


def build(annee_deb: int, annee_fin: int) -> Path:
    raw_dir = config.DATA_RAW / "bdiff"
    raw_dir.mkdir(parents=True, exist_ok=True)

    batches = list(iter_year_batches(annee_deb, annee_fin))
    print(f"BDIFF: {annee_deb}-{annee_fin} split into {len(batches)} batch(es): {batches}")

    seen: set[tuple[str, str]] = set()
    all_rows: list[dict] = []
    for b_deb, b_fin in batches:
        print(f"  fetching {b_deb}-{b_fin}...")
        raw = fetch_zip(b_deb, b_fin)
        (raw_dir / f"incendies_{b_deb}_{b_fin}.zip").write_bytes(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_bytes = zf.read("Incendies.csv")
        rows = parse_incendies_csv(csv_bytes)
        added = 0
        for r in rows:
            key = (r["year"], r["fire_number"])
            if key in seen:
                continue
            seen.add(key)
            all_rows.append(r)
            added += 1
        print(f"    {len(rows)} records ({added} new)")

    out = config.DATA_INTERIM / "bdiff_incendies.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {out} ({len(all_rows)} total records)")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        deb, fin = int(args[0]), int(args[1])
    else:
        deb, fin = config.BASE_YEAR_MIN, date.today().year
    build(deb, fin)
