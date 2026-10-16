# Sources de données, feux de forêt en Europe

Catalogue des sources retenues, avec statut de vérification au 2026-07-30.
Statut : **live** (endpoint testé et fonctionnel), **fichier** (téléchargement
documenté), **compte** (clé/inscription requise), **à câbler** (mécanisme
identifié, module d'ingestion à écrire).

## Socle officiel européen

| Source | Contenu | Granularité | Période | Accès | Statut |
|---|---|---|---|---|---|
| EFFIS `rda-stats` | Surface brûlée agrégée par macro-région (EU, EU_non, ME_AF, total) | Macro-région | 2006-2025 testé | `GET api2.effis.emergency.copernicus.eu/rda-stats?year=YYYY` | **live** |
| EFFIS portail stats | Totaux pays (surface, nombre de feux) | Pays | 2000+ | Téléchargement .xls | fichier |
| EFFIS périmètres (RDA) | Polygones de zones brûlées | Feu (polygone) | 2016+ (WFS), <2016 sur formulaire | WFS GetFeature GeoJSON, `ms:modis.ba.poly`, 102440 feux | **live** |
| EU Fire Database | Données nationales harmonisées, vérifiées JRC | 22 pays | 2000+ | Formulaire de requête / data.europa.eu | fichier |
| GWIS country profiles, émissions | Émissions de feux de végétation (CO2, CH4, PM2.5), produit **GFAS** | Pays (national uniquement, `gid_1` vide) | 2003-2025 | `GET effis-gwis-cms.s3.eu-west-1.amazonaws.com/apps/country.profile/emission_gfas_full_2003_2024.zip` (CSV `;`) | **live** |
| Rapports annuels JRC | Chiffres validés, tables par pays | Pays | 2000-2025 | PDF + jeux de données | fichier |

Note : le param `country` de `rda-stats` renvoie une erreur 500, ne pas
l'utiliser. L'ancienne API `rest/2/burntareas` (client R `effisr`) est
dépréciée (404). Schéma réel des périmètres capturé : `objectid, id,
countryful, province, commune, firedate, area_ha, broadlea, conifer, mixed,
scleroph, transit, othernatlc, agriareas, artifsurf, otherlc, percna2k,
lastupdate, ba_class, country`.

## Satellite brut (granularité, traitement propre)

| Source | Contenu | Résolution | Période | Accès | Statut |
|---|---|---|---|---|---|
| NASA FIRMS | Hotspots actifs (lat/lon, FRP, confiance) | 375 m (VIIRS) | MODIS 2000+, VIIRS 2012+ | API area CSV, MAP_KEY | **compte** |
| MODIS MCD64A1 / VIIRS BA | Surface brûlée maillée | 500 m | mensuel | NASA Earthdata / AppEEARS | compte |

FIRMS : clé gratuite sur firms.modaps.eosdis.nasa.gov/api/map_key, limite
5000 requêtes / 10 min. Sources archive = suffixe `_SP`, temps réel = `_NRT`.

## Covariables explicatives (le cœur analytique)

| Source | Contenu | Période | Accès | Statut |
|---|---|---|---|---|
| CEMS EWDS, FWI (`cems-fire-historical-v1`) | Indices de danger feu journaliers, ERA5 | 1940+, ingéré 2016-2026 | `cdsapi`, store **EWDS** | **live** |
| ERA5 monthly means (`reanalysis-era5-single-levels-monthly-means`) | Température 2 m, précipitations totales, point de rosée 2 m | 1940+, ingéré 1991-2026 | `cdsapi`, store **CDS**, grille 0.5° | **live** |
| European Drought Observatory, SPEI-3 (`sem03`) | Indice sécheresse standardisé 3 mois, grille 10 km EPSG:3035, pas décadaire | 1991-2026 | `GET drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/EDO_MARSMet_Standardized_Precipitation_Evapotranspiration_Index_SPEI3/ver1-1-0/`, netCDF annuels, **sans clé** | **live** |
| CORINE Land Cover | Type de végétation (attribution surface brûlée) | 1990-2018 | Copernicus Land | fichier |
| Open-Meteo, archive historique, modèle `era5` | ERA5 journalier par point (température, humidité, vent, précipitations, VPD, ET0), agrégé côté serveur en UTC, 0,25° | 1940+ | `GET archive-api.open-meteo.com/v1/archive?models=era5&daily=...`, sans clé, CC BY 4.0, usage non commercial | **live** (spike Q1) |

Bornes d'ingestion mesurées le 2026-07-31 : FWI 2016-01 à 2026-04 (la
consolidation EWDS retarde de quelques mois sur le temps réel), ERA5 1991-01 à
2026-06, SPEI 2016-01 à 2026-07. Le plancher 2016 est un choix de périmètre
aligné sur les périmètres EFFIS, pas une limite de source : les trois remontent
plus loin.

L'humidité relative n'est pas publiée telle quelle par ERA5 : elle est dérivée
de la température et du point de rosée par la formule de Magnus-Tetens
(coefficients 17,67 / 243,5 °C), voir le docstring de `src/ingest/era5.py`.

Autres produits d'émissions offerts par GWIS (GFED, FAO) écartés : tous deux
dérivent des émissions d'une surface brûlée, donc les associer à la surface
brûlée EFFIS reviendrait à mesurer deux fois la même chose. GFAS part de la
puissance radiative satellite, c'est la seule observation indépendante des
trois. Ses totaux nationaux dépassent structurellement ce que suggère la
surface EFFIS : GFAS voit aussi les feux agricoles et les feux sous le seuil de
cartographie EFFIS. Jamais additionnés.

## Deep-dive national (France)

| Source | Contenu | Granularité | Période | Accès | Statut |
|---|---|---|---|---|---|
| BDIFF | Feux, causes, surfaces par type | Commune | 2006+ | GET `/incendies/zip` (défaut = année courante, params date pour l'historique) | **live** |

Prométhée (Méditerranée) a fusionné dans BDIFF en janvier 2023. La ressource
data.gouv est une page HTML, pas un fichier direct : l'export CSV passe par le
formulaire de bdiff.agriculture.gouv.fr.

## Comptes et clés

1. **NASA FIRMS MAP_KEY** : validé, dans `.env` (gitignoré).
2. **Un seul token ECMWF pour les deux stores.** Vérifié le 2026-07-31 : le
   Personal Access Token du compte ECMWF unifié s'authentifie aussi bien sur
   EWDS (`ewds.climate.copernicus.eu/api`) que sur le CDS principal
   (`cds.climate.copernicus.eu/api`). Seule l'URL de base change. Il n'y a donc
   pas de second token à créer, contrairement à ce que laissait entendre la
   version précédente de ce document.
3. **Les licences, elles, restent par dataset** et s'acceptent une fois en
   ligne sur la page du dataset : `cems-fire-historical-v1` (EWDS),
   `reanalysis-era5-single-levels-monthly-means` (CDS).
4. **`~/.cdsapirc` ne porte qu'un couple url/clé** et ne peut donc pas servir
   les deux stores. `src/config.py` expose `cds_client("cds" | "ewds")` qui
   passe `url` et `key` explicitement à `cdsapi.Client`. Le token se résout
   dans cet ordre : `CDS_API_KEY` / `EWDS_API_KEY`, puis `ECMWF_API_KEY`, puis
   la ligne `key:` de `~/.cdsapirc`. Aucune URL n'est lue de ce fichier.

EFFIS, GWIS, EDO, BDIFF, rapports JRC : ouverts, sans clé.

## Piège de nomenclature : Natural Earth et le code ISO

L'attribution pays (hotspots FIRMS, mailles FWI, mailles ERA5, mailles SPEI)
passe par du point-in-polygon sur Natural Earth admin-0 1:50m. **Ce jeu code
`ISO_A2 = "-99"` pour la France, la Norvège et le Kosovo.** Filtrer sur
`ISO_A2` supprimait donc silencieusement la France de toutes les tables
climatiques et de hotspots, y compris celle qui alimente la page de deep-dive
français. Corrigé le 2026-07-31 : le code effectif est `ISO_A2_EH` puis
`ISO_A2` en repli, dans le module partagé `src/ingest/geo_ref.py`, seul point
d'entrée autorisé vers ces polygones.
