# Plateforme v2 : lake, orchestration, gold en SQL

État au 2026-10-15 : ossature en place et testée sans données. Ce document
décrit ce qui existe ; PLAN.md (local) décrit ce qui vient.

## Couches

```
bronze   data/raw et data/interim      brut tel que reçu, une table interim par source (modules v1 inchangés)
silver   data/processed                 une table propre par source, validée par contrat pandera à l'écriture
gold     data/lake/gold/star            le schéma en étoile que Power BI lit, écrit par dbt en Parquet
         data/lake/gold/features        feature store maille x jour (phase 1)
         data/lake/gold/model_out       sorties de modèles (phase 2)
```

Les chemins sont centralisés dans `src/lake.py` et surchargeables par
variables d'environnement (`WILDFIRE_SILVER`, `WILDFIRE_GOLD_STAR`,
`WILDFIRE_INTERIM`), ce qui permet aux tests de pointer toute la chaîne sur un
dossier temporaire.

### Le pont de migration

Les fonctions de transformation v1 écrivent dans `data/processed`. Tant
qu'elles ne sont pas réécrites, **ce dossier est la silver** : les assets
silver appellent les fonctions v1 telles quelles, dbt lit la silver et produit
la gold. Power BI continue de lire `data/processed` jusqu'à ce que le test de
parité (`tests/test_gold_parity.py`) prouve que `gold/star` lui est égale
valeur pour valeur ; son paramètre `DataFolder` bascule alors sur `gold/star`.
Aucun fichier n'est déplacé à la main à aucun moment.

## Dagster (`src/orchestration/`)

Une code location, chargée par `make dagster-dev` (configuration dans
`pyproject.toml`, section `[tool.dagster]`). Les clés d'assets portent la
couche : `bronze/effis_perimeters`, `silver/fact_fireevent`,
`gold/fact_burntarea_countryyear`.

| Couche | Assets | Ce qu'ils font aujourd'hui |
|---|---|---|
| bronze | 9, dont `cds_fwi_raw` et `firms_hotspots_raw` partitionnés par année | enveloppent `src/ingest/<source>.build`, reprenables fichier par fichier |
| silver | 9 (6 dimensions, 3 faits) | appellent les fonctions v1, valident le contrat pandera, écrivent la silver |
| gold | 12, un par modèle dbt | `dbt build`, lignage lu dans le manifeste, sources dbt résolues sur les assets silver du même nom |

Chaque table gold porte deux familles de checks visibles dans l'interface :
le contrat pandera (`contract`, le même objet que les assets silver et les
tests utilisent) et les tests dbt du modèle, remontés automatiquement.

Limite assumée de l'ossature : seuls FWI et FIRMS sont réellement
partitionnés ; les autres sources reconstruisent leur étendue complète en un
appel, ce qui est bon marché grâce aux téléchargements reprenables. La phase 1
les réécrit en assets partitionnés sans changer leurs clés.

## dbt (`dbt/`)

Projet `wildfire`, adaptateur dbt-duckdb. Les sources sont les Parquet de la
silver et le CSV interim des émissions GWIS, lus en place par DuckDB. Chaque
modèle gold est matérialisé en Parquet dans `gold/star` (`external_root` du
profil), un fichier par table.

- 9 modèles de passage (`select *` de la silver) : les tables déjà au bon grain.
- 3 rollups en SQL qui remplacent `src/transform/rollups.py` :
  `fact_burntarea_countrymonth`, `fact_burntarea_countryyear` (avec les lignes
  d'émissions GWIS sur leur propre source), `fact_burntarea_landcover`.
- Tests dbt : grain (`unique_combination_of_columns`), valeurs admises,
  `non_negative`, `no_source_mixing` (une ligne GWIS ne porte jamais de
  surface, une ligne EFFIS ou BDIFF jamais d'émissions).

`tests/test_dbt_gold.py` écrit une silver synthétique, lance `dbt build`
dessus et compare chaque rollup SQL à la fonction Python v1 sur la même
entrée, par la forme canonique du test de parité. C'est la preuve
d'équivalence qui rend le test de parité réel crédible une fois les données
présentes.

## Commandes

```bash
make dbt-parse     # manifeste dbt, nécessaire aux assets Dagster hors `dagster dev`
make dbt-build     # gold/star depuis la silver, avec les tests dbt
make dagster-dev   # interface Dagster sur localhost:3000
make check         # ruff, mypy, pytest (dont dbt build sur données synthétiques)
```

## Feature store (`gold/features`)

Code dans `src/features/`, assets dans `src/orchestration/assets/features.py`.
Une année par partition, un Parquet par année (`daily_<année>.parquet`),
grain maille x jour sur la grille commune (`src/grid.py`, ADR 0003).

| Colonne | Source | Règle de non-fuite |
|---|---|---|
| `cell`, `date`, `year`, `doy` | grille, calendrier | |
| `fwi`, `ffmc`, `dmc`, `dc`, `isi`, `bui`, `dsr` | EWDS cems-fire-historical-v1, jour J | réanalyse du jour, comme l'indice opérationnel |
| `t2m_c`, `t2m_max`, `rh`, `rh_min`, `wind`, `wind_mean`, `tp_mm`, `vpd_max`, `et0` | Open-Meteo archive, modèle era5, jour J (ADR 0006) | seules les variables servies sont présentes |
| `tp_7d` | somme de `tp_mm` sur J-7 à J-1 | J exclu, nul sous 3 jours disponibles |
| `spei3` | EDO SPEI-3 décadaire | dernière décade dont la fin est strictement avant J |
| `cvh`, `cvl`, `tvh`, `tvl`, `lsm`, `elev_m` | ERA5 champs statiques | statiques |
| `laga_7`, `laga_30` | départs EFFIS dans la maille sur J-7 à J-1 et J-30 à J-1 | J exclu, historique depuis 2016 |
| `n_a`, `y_a` | label (a) : départs EFFIS dans la maille le jour J | cible, jamais feature |

Le label (b), premier jour d'un cluster de hotspots VIIRS, est construit par
`src/features/hotspots.py` (DBSCAN 2 km par jour, chaînage à J-1) et rejoint
la table dès que les fenêtres FIRMS brutes couvrent l'Europe.

Chaque règle de la colonne de droite est un test dans
`tests/test_features_daily.py` sur des NetCDF synthétiques.

## Ingestion partition-safe

| Asset | Partition | Reprise |
|---|---|---|
| `bronze/cds_fwi_raw` | année | fichier NetCDF par année |
| `bronze/era5_raw` | décennie | fichier NetCDF par décennie |
| `bronze/edo_spei_raw` | année | fichier NetCDF par année, années non publiées sautées |
| `bronze/bdiff_raw` | année | zip par année, `bdiff_incendies` parse et déduplique tous les zips |
| `bronze/firms_hotspots_raw` | année | fenêtres CSV de 5 jours, `fact_hotspot_countrymonth` agrège toutes les années |
| `bronze/openmeteo_daily` | lots de mailles | JSON par lot, quota journalier respecté |

`effis_perimeters`, `effis_rda_stats` et `gwis_gfas_emissions` restent des
appels uniques : leurs sources ne se découpent pas.

## Modèles (`src/models/`, sorties dans `gold/model_out`)

| Module | Question | Ce qu'il produit | Vérifié par |
|---|---|---|---|
| `extremes.py` | Q2, queue des tailles de feux | ajustement Pareto généralisé au-dessus d'un seuil par pays, niveaux de retour à 2, 5, 10, 20, 50 ans avec intervalles bootstrap, niveaux empiriques en regard, table de stabilité du seuil, période de retour d'un feu de 10 000 ha | récupération de paramètres connus sur un échantillon synthétique, inversion niveau / période |
| `attribution.py` | Q3, part de la météo dans la variabilité annuelle | panel pays x année, effets fixes pays, erreurs groupées par pays, composante météo et surface ajustée du climat par pays et année, R² intra-pays, contrôle leave-one-year-out | signal synthétique retrouvé (signe, ordre de grandeur), années à saison incomplète exclues et non imputées |
| `validation.py` | protocole commun | découpes par année tenue et par blocs spatiaux de 2°, métriques du critère pré-enregistré, fiabilité par décile, bootstrap par blocs sur les différences | partition exacte des lignes, blocs jamais scindés, intervalles qui séparent deux modèles de qualité connue |
| `tracking.py` | traçabilité | MLflow en SQLite local sans serveur, un identifiant de run sur chaque table publiée, gabarit de carte de modèle | run relu depuis le store |

La version bayésienne de Q3 (PyMC, WSL2) consomme le même panel et doit
retrouver le signe et l'ordre de grandeur de chaque coefficient. Q1 rejoint
ce dossier une fois le périmètre fixé par le verdict du spike.
