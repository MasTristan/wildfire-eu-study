"""Build the conformed dimension tables (data/processed/dim_*.parquet).

Every dimension here is either pure calendar generation (dim_date) or public
reference nomenclature (ISO/EU membership, French administrative geography,
EFFIS land-cover schema, source documentation) or derived directly from
already-ingested source data (BDIFF cause labels). No business figures are
invented: land/forest area come from the World Bank API, commune geography
from the French government's geo.api.gouv.fr, at run time.

Run:
    python src/transform/dimensions.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_TIMEOUT = 30

_MONTH_NAMES_FR = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]
_SEASON_FR = {12: "Hiver", 1: "Hiver", 2: "Hiver",
              3: "Printemps", 4: "Printemps", 5: "Printemps",
              6: "Été", 7: "Été", 8: "Été",
              9: "Automne", 10: "Automne", 11: "Automne"}


# --- Dim_Date -----------------------------------------------------------------

def build_dim_date() -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", "2026-12-31", freq="D")
    df = pd.DataFrame({"date_key": dates})
    df["year"] = df["date_key"].dt.year
    df["quarter"] = df["date_key"].dt.quarter
    df["month"] = df["date_key"].dt.month
    df["month_name"] = df["month"].map(lambda m: _MONTH_NAMES_FR[m - 1])
    df["week_iso"] = df["date_key"].dt.isocalendar().week.astype("int64")
    df["day_of_year"] = df["date_key"].dt.dayofyear
    df["is_fire_season"] = df["month"].between(6, 9)
    df["season_label"] = df["month"].map(_SEASON_FR)
    return df


# --- Dim_Country ----------------------------------------------------------------

def _worldbank_indicator(iso3_codes: list[str], indicator: str) -> dict[str, float]:
    """Most recent non-null value per country for a World Bank indicator."""
    codes = ";".join(iso3_codes)
    url = f"https://api.worldbank.org/v2/country/{codes}/indicator/{indicator}"
    out: dict[str, float] = {}
    page = 1
    while True:
        for attempt in range(4):
            try:
                params: dict[str, str | int] = {"format": "json", "per_page": 500, "mrv": 10, "page": page}
                resp = requests.get(url, params=params, timeout=90)
                break
            except requests.exceptions.Timeout:
                if attempt == 3:
                    raise
                print(f"    (timeout, retry {attempt + 1}/3)")
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            break
        for rec in payload[1]:
            iso3 = rec["countryiso3code"]
            if iso3 and iso3 not in out and rec.get("value") is not None:
                out[iso3] = rec["value"]
        meta = payload[0]
        if page >= meta.get("pages", 1):
            break
        page += 1
    return out


def build_dim_country() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE_DIR / "country_reference.csv")
    iso3_list = ref["iso3"].tolist()

    print("  World Bank: fetching land area (AG.LND.TOTL.K2)...")
    land = _worldbank_indicator(iso3_list, "AG.LND.TOTL.K2")
    print("  World Bank: fetching forest area (AG.LND.FRST.K2)...")
    forest = _worldbank_indicator(iso3_list, "AG.LND.FRST.K2")

    ref["land_area_km2"] = ref["iso3"].map(land)
    ref["forest_area_ha"] = ref["iso3"].map(forest).mul(100)  # km2 -> ha

    missing = ref[ref["land_area_km2"].isna() | ref["forest_area_ha"].isna()]
    if not missing.empty:
        print(f"  NOTE: no World Bank area data for {missing['country_key'].tolist()} "
              f"(small/disputed territories not in WB catalogue; left null)")

    ref["iso2"] = ref["country_key"]
    return ref[[
        "country_key", "effis_raw_code", "iso2", "iso3", "name_en", "name_fr",
        "macro_region", "is_eu", "is_mediterranean", "land_area_km2", "forest_area_ha",
    ]]


# --- Dim_Commune (France) --------------------------------------------------------

def build_dim_commune() -> pd.DataFrame:
    print("  geo.api.gouv.fr: fetching regions/departements/communes...")
    regions = {r["code"]: r["nom"] for r in requests.get(
        "https://geo.api.gouv.fr/regions", params={"fields": "code,nom"}, timeout=_TIMEOUT
    ).json()}
    depts = requests.get(
        "https://geo.api.gouv.fr/departements",
        params={"fields": "code,nom,codeRegion"}, timeout=_TIMEOUT,
    ).json()
    dept_name = {d["code"]: d["nom"] for d in depts}

    communes = requests.get(
        "https://geo.api.gouv.fr/communes",
        params={"fields": "nom,code,codeDepartement,codeRegion,centre", "format": "json"},
        timeout=60,
    ).json()

    rows = []
    for c in communes:
        lon, lat = (c.get("centre") or {}).get("coordinates", [None, None])
        dep = c.get("codeDepartement")
        reg = c.get("codeRegion")
        rows.append({
            "commune_key": c["code"],
            "name": c["nom"],
            "department_code": dep,
            "department_name": dept_name.get(dep),
            "region_code": reg,
            "region_name": regions.get(reg),
            "country_key": "FR",
            "centroid_lon": lon,
            "centroid_lat": lat,
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} communes")
    return df


# --- Dim_LandCover ----------------------------------------------------------------
# Aligned to the EFFIS perimeter attribute schema (percent-of-area columns);
# corine_mapping documents the CORINE Land Cover level-1/2 classes each bucket
# aggregates (per EFFIS/JRC methodology notes), not an independent source.

def build_dim_landcover() -> pd.DataFrame:
    rows = [
        ("broadleaf", "BROADLEA", "Feuillus", "CLC 311 Forêts de feuillus"),
        ("conifer", "CONIFER", "Conifères", "CLC 312 Forêts de conifères"),
        ("mixed", "MIXED", "Forêt mixte", "CLC 313 Forêt mélangée"),
        ("sclerophyllous", "SCLEROPH", "Végétation sclérophylle", "CLC 323 Végétation sclérophylle"),
        ("transitional", "TRANSIT", "Forêt/végétation en mutation", "CLC 324 Forêt et végétation arbustive en mutation"),
        ("other_natural", "OTHERNATLC", "Autres milieux naturels", "CLC 32x/33x autres milieux semi-naturels"),
        ("agricultural", "AGRIAREAS", "Surfaces agricoles", "CLC 2xx surfaces agricoles"),
        ("artificial", "ARTIFSURF", "Surfaces artificialisées", "CLC 1xx territoires artificialisés"),
        ("other", "OTHERLC", "Autre / non classé", "hors nomenclature CLC dominante"),
    ]
    return pd.DataFrame(rows, columns=["landcover_key", "class_code", "class_name", "corine_mapping"])


# --- Dim_FireCause (BDIFF, France) ------------------------------------------------

_CAUSE_GROUP = {
    "Naturelle": "natural",
    "Accidentelle": "accidental",
    "Malveillance": "malicious",
    "Involontaire (particulier)": "involuntary",
    "Involontaire (travaux)": "involuntary",
    "": "unknown",
}


def cause_key_from_label(detail: str) -> str:
    """Shared slug so fact_fireevent joins on the same key dim_firecause uses."""
    if not detail:
        return "unknown"
    return detail.lower().replace(" ", "_").replace("(", "").replace(")", "")


def build_dim_firecause() -> pd.DataFrame:
    bdiff_csv = config.DATA_INTERIM / "bdiff_incendies.csv"
    if not bdiff_csv.exists():
        raise FileNotFoundError(f"{bdiff_csv} missing: run src/ingest/bdiff.py first")
    causes = set()
    with bdiff_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            causes.add(row["cause_raw"])

    rows = []
    for detail in sorted(causes):
        group = _CAUSE_GROUP.get(detail)
        if group is None:
            raise ValueError(f"Unmapped BDIFF cause label: {detail!r}")
        label = detail if detail else "Non renseignée"
        rows.append({
            "cause_key": cause_key_from_label(detail),
            "cause_group": group,
            "cause_detail": label,
        })
    return pd.DataFrame(rows)


# --- Dim_Source ---------------------------------------------------------------
# Documentation of each source's method, not observational data; mirrors
# docs/sources.md and docs/data-model.md section 2.1.

def build_dim_source() -> pd.DataFrame:
    rows = [
        ("EFFIS_STATS", "burnt_area", "macro-region (EU/EU_non/ME_AF)", 2006,
         "rda-stats: agrégats macro-région, param country cassé (HTTP 500), utilisé pour contrôle de cohérence uniquement"),
        ("EFFIS_RDA", "burnt_area", "polygone (feu individuel)", 2016,
         "Périmètres WFS ms:modis.ba.poly ; sous-compte les petits feux ; rupture méthodologique 2012 (tendances homogènes 2012+ seulement)"),
        ("BDIFF", "burnt_area", "commune (France)", 2006,
         "Export formulaire bdiff.agriculture.gouv.fr, plafond 30000 lignes par export (paginé par année) ; Prométhée fusionné en 2023 ; non comparable en définition à EFFIS"),
        ("FIRMS_VIIRS", "hotspot", "375 m (point)", 2012,
         "Détections thermiques, jamais comparé aux hectares brûlés ; ingestion bornée à la période récente (voir dim_source.min_year effectif dans fact_hotspot)"),
        # FIRMS_MODIS retiré le 2026-07-31 (phase 3) : déclaré ici depuis le MVP
        # mais jamais ingéré, donc invisible dans les faits et trompeur dans un
        # slicer. Le réintroduire le jour où src/ingest/firms.py le télécharge.
        ("GWIS_GFAS", "emissions", "pays (agrégat national)", 2003,
         "GWIS country profile, produit GFAS (CAMS) : masse émise dérivée de la puissance radiative "
         "satellite, indépendante d'EFFIS. Couvre aussi les feux agricoles et les feux sous le seuil "
         "de cartographie EFFIS, donc totaux non réconciliables avec la surface brûlée EFFIS ; "
         "jamais additionnés"),
        ("CDS_FWI", "fire_weather", "grille 0.5°", 1940,
         "cems-fire-historical-v1, EWDS, moyenne spatiale non masquée par la forêt (simplification documentée)"),
        ("ERA5", "climate", "grille 0.5°", 1940,
         "reanalysis-era5-single-levels-monthly-means (CDS) : température 2 m, précipitations totales, "
         "humidité relative dérivée du point de rosée (Magnus). Anomalie de précipitations calculée "
         "contre la normale 1991-2020"),
        ("EDO_SPEI", "drought", "grille 10 km (agrégée pays)", 1991,
         "European Drought Observatory, SPEI-3 MARSMet : indice standardisé précipitations-"
         "évapotranspiration sur 3 mois. Négatif = déficit hydrique. Non imputé hors couverture"),
    ]
    return pd.DataFrame(rows, columns=[
        "source_key", "measure_type", "spatial_resolution", "min_year", "method_note",
    ])


def build_all() -> None:
    builders = {
        "dim_date": build_dim_date,
        "dim_country": build_dim_country,
        "dim_commune": build_dim_commune,
        "dim_landcover": build_dim_landcover,
        "dim_firecause": build_dim_firecause,
        "dim_source": build_dim_source,
    }
    for name, fn in builders.items():
        print(f"Building {name}...")
        df = fn()
        out = config.DATA_PROCESSED / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"  {len(df)} rows -> {out}")


if __name__ == "__main__":
    build_all()
