"""Central configuration: paths, verified data-source endpoints, study scope.

Only endpoints that have been verified live are declared here. Sources that
require an account key or a form-based export are documented in docs/sources.md
and wired in their own ingest module once the credential is available.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# --- EFFIS (Copernicus EMS / JRC) --------------------------------------------
# VERIFIED live 2026-07-30: aggregate Rapid Damage Assessment burnt-area stats
# by macro-region (EU, EU_non, ME_AF, total) for a given year.
# Params: year=<int>. The `country` param currently returns HTTP 500, do not use.
EFFIS_RDA_STATS_URL = "https://api2.effis.emergency.copernicus.eu/rda-stats"

# EFFIS MapServer WFS: burnt-area perimeters as GeoJSON.
# VERIFIED 2026-07-30: typename ms:modis.ba.poly, 102440 features, coverage
# 2016-02 to present (live). Pre-2016 perimeters need the EFFIS data-request form.
EFFIS_MAPS_HOST = "https://maps.effis.emergency.copernicus.eu/effis"
EFFIS_WFS_TYPENAME = "ms:modis.ba.poly"

# NOTE: the legacy REST API (rest/2/burntareas, used by the R package effisr)
# is deprecated and returns 404. Per-country annual totals are downloaded as
# .xls from the EFFIS statistics portal, see docs/sources.md.

# --- NASA FIRMS --------------------------------------------------------------
# Active-fire hotspots. Requires a free MAP_KEY (set FIRMS_MAP_KEY in .env).
# Area CSV API: /api/area/csv/{MAP_KEY}/{SOURCE}/{W,S,E,N}/{DAY_RANGE}/{DATE}
FIRMS_AREA_CSV_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")
# Sensors: MODIS_NRT/MODIS_SP, VIIRS_SNPP_NRT/_SP, VIIRS_NOAA20_NRT, VIIRS_NOAA21_NRT
FIRMS_DEFAULT_SOURCE = "VIIRS_SNPP_SP"  # SP = standard/archive quality

# --- Copernicus fire danger (FWI) and climate covariates ---------------------
# Two distinct ECMWF data stores are used:
#   EWDS = CEMS Early Warning Data Store, hosts the FWI historical dataset.
#   CDS  = Climate Data Store, hosts raw ERA5 (temperature, precipitation, dewpoint).
# VERIFIED 2026-07-31: a single ECMWF Personal Access Token authenticates against
# BOTH stores (same unified ECMWF account); only the base URL differs. Licences
# are still per-dataset and accepted once online.
EWDS_API_URL = "https://ewds.climate.copernicus.eu/api"
CDS_API_URL = "https://cds.climate.copernicus.eu/api"
CDS_FWI_DATASET = "cems-fire-historical-v1"  # fire danger indices, ERA5-based
CDS_ERA5_DATASET = "reanalysis-era5-single-levels-monthly-means"

# Token resolution order, per store: dedicated env var, then the shared
# ECMWF_API_KEY, then the key line of ~/.cdsapirc. cdsapi only reads a single
# url/key couple from ~/.cdsapirc, so both ingest modules pass url and key
# explicitly instead of relying on that file's `url`.
ECMWF_API_KEY = os.getenv("ECMWF_API_KEY", "")
CDS_API_KEY = os.getenv("CDS_API_KEY", "")
EWDS_API_KEY = os.getenv("EWDS_API_KEY", "")


def _cdsapirc_key() -> str:
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        return ""
    for line in rc.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("key"):
            return line.split(":", 1)[1].strip()
    return ""


def cds_client(store: str = "cds", **kwargs):
    """Return a cdsapi.Client bound to `store` ("cds" or "ewds").

    Raises with an actionable message rather than letting cdsapi fail on a
    missing/implicit ~/.cdsapirc, which is the usual first-run stumble here.
    """
    import cdsapi

    store = store.lower()
    if store == "ewds":
        url, key = EWDS_API_URL, EWDS_API_KEY
    elif store == "cds":
        url, key = CDS_API_URL, CDS_API_KEY
    else:
        raise ValueError(f"unknown store {store!r}, expected 'cds' or 'ewds'")

    key = key or ECMWF_API_KEY or _cdsapirc_key()
    if not key:
        raise RuntimeError(
            f"No ECMWF token for store {store!r}. Set ECMWF_API_KEY (or "
            f"{store.upper()}_API_KEY) in .env, or put a `key:` line in "
            f"~/.cdsapirc. The same token works on both CDS and EWDS."
        )
    return cdsapi.Client(url=url, key=key, **kwargs)

# --- BDIFF (France, commune level) -------------------------------------------
# Form-based CSV export on the government portal. The data.gouv resource is an
# HTML landing page, not a direct file. Export endpoint wired in bdiff module.
BDIFF_HOST = "https://bdiff.agriculture.gouv.fr"

# --- Study scope -------------------------------------------------------------
# EU + associated countries present in the EFFIS harmonised fire database.
EU_FIRE_DB_COUNTRIES = [
    "BG", "HR", "CY", "CZ", "EE", "FI", "FR", "DE", "GR", "HU", "IT", "LV",
    "LB", "LT", "PL", "PT", "RO", "SK", "ES", "SE", "CH", "TR",
]
# Mediterranean core (bulk of burnt area) for focused analysis.
MED_CORE = ["ES", "PT", "IT", "GR", "FR"]

BASE_YEAR_MIN = 2006  # BDIFF starts 2006; EFFIS perimeters 2000, trends 2012
BASE_YEAR_MAX = 2025
