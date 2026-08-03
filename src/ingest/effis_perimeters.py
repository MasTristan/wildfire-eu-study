"""EFFIS ingest: burnt-area perimeters (individual fire events) via WFS.

Pulls the ms:modis.ba.poly layer as GeoJSON with WFS 2.0 pagination
(count + startindex, stable sort on FIREDATE). Raw GeoJSON pages are saved to
data/raw/effis_perimeters/; a flat attributes table (no geometry) is written to
data/interim for quick inspection. Geometry to GeoParquet + centroids happens in
the transform layer (needs geopandas). Stdlib only, runs before env install.

Coverage: 2016 to present (~102k fires). Pre-2016 needs the EFFIS request form.

Run (validation slice, 1 page):
    python src/ingest/effis_perimeters.py
Run (full pull):
    python src/ingest/effis_perimeters.py --full
"""
from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

_TIMEOUT = 90
_ATTR_FIELDS = [
    "id", "FIREDATE", "FINALDATE", "LASTUPDATE", "COUNTRY", "PROVINCE",
    "COMMUNE", "AREA_HA", "BROADLEA", "CONIFER", "MIXED", "SCLEROPH", "TRANSIT",
    "OTHERNATLC", "AGRIAREAS", "ARTIFSURF", "OTHERLC", "PERCNA2K", "CLASS",
]


def _wfs_url(startindex: int, count: int) -> str:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typename": config.EFFIS_WFS_TYPENAME,
        "outputformat": "geojson",
        # FIREDATE alone is not a unique sort key: features sharing the exact
        # page-boundary timestamp can be emitted on both pages (confirmed:
        # 46 exact-duplicate rows in a 102k pull). id is a stable tiebreaker.
        "sortby": "FIREDATE,id",
        "count": count,
        "startindex": startindex,
    }
    return f"{config.EFFIS_MAPS_HOST}?{urlencode(params)}"


def fetch_page(startindex: int, count: int) -> dict:
    url = _wfs_url(startindex, count)
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def build(page_size: int = 5000, max_pages: int | None = None, save_geometry: bool = True) -> int:
    out_dir = config.DATA_RAW / "effis_perimeters"
    out_dir.mkdir(parents=True, exist_ok=True)
    attrs: list[dict] = []
    page = 0
    startindex = 0

    while True:
        gj = fetch_page(startindex, page_size)
        feats = gj.get("features", [])
        if not feats:
            break
        if save_geometry:
            (out_dir / f"page_{page:04d}.geojson").write_text(json.dumps(gj), encoding="utf-8")
        for f in feats:
            props = f.get("properties", {})
            attrs.append({k: props.get(k) for k in _ATTR_FIELDS})
        got = len(feats)
        print(f"  page {page}: {got} features (total {len(attrs)})")
        page += 1
        startindex += got
        if got < page_size:
            break
        if max_pages is not None and page >= max_pages:
            break

    out_csv = config.DATA_INTERIM / "effis_perimeters_attrs.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ATTR_FIELDS)
        writer.writeheader()
        writer.writerows(attrs)
    print(f"Wrote {out_csv} ({len(attrs)} rows)")
    return len(attrs)


if __name__ == "__main__":
    full = "--full" in sys.argv
    if full:
        print("EFFIS perimeters: FULL pull")
        build(page_size=5000, max_pages=None)
    else:
        print("EFFIS perimeters: validation slice (1 page of 2000)")
        build(page_size=2000, max_pages=1)
