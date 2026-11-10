# Plateforme : lake, orchestration, gold en SQL

État au 2026-11-10 : la chaîne est matérialisée de bout en bout sur les données
réelles. La gold reproduit la v1 valeur pour valeur (test de parité), le rapport
Power BI la lit, et trois modèles écrivent leurs sorties à côté. Ce qui n'est
pas matérialisé est dit comme tel : le feature store Europe et le modèle
d'occurrence à l'échelle de l'Europe, abandonnés après le verdict du spike
([ADR 0009](adr/0009-q1-scope-after-reduced-go.md)). Leur code reste dans le
dépôt, testé sur données synthétiques.

## Couches

```
bronze   data/raw et data/interim      brut tel que reçu, une table interim par source (modules v1 inchangés)
silver   data/processed                 une table propre par source, validée par contrat pandera à l'écriture
gold     data/lake/gold/star            le schéma en étoile que Power BI lit, écrit par dbt en Parquet
         data/lake/gold/features        feature store maille x jour (code testé, non matérialisé)
         data/lake/gold/model_out       sorties de modèles, un run MLflow par table
```

Les chemins sont centralisés dans `src/lake.py` et surchargeables par
variables d'environnement (`WILDFIRE_SILVER`, `WILDFIRE_GOLD_STAR`,
`WILDFIRE_INTERIM`), ce qui permet aux tests de pointer toute la chaîne sur un
dossier temporaire.

### Le pont de migration

Les fonctions de transformation v1 écrivent dans `data/processed`. Tant
qu'elles ne sont pas réécrites, **ce dossier est la silver** : les assets
silver appellent les fonctions v1 telles quelles, dbt lit la silver et produit
la gold. Power BI a lu `data/processed` jusqu'à ce que le test de
parité (`tests/test_gold_parity.py`) prouve que `gold/star` lui est égale
valeur pour valeur ; son paramètre `DataFolder` a basculé sur `gold/star` le
2026-11-09, sans qu'aucun chiffre du rapport ne change. Aucun fichier n'est
déplacé à la main à aucun moment.

## Dagster (`src/orchestration/`)

Une code location, chargée par `make dagster-dev` (configuration dans
`pyproject.toml`, section `[tool.dagster]`). Les clés d'assets portent la
couche : `bronze/effis_perimeters`, `silver/fact_fireevent`,
`gold/fact_burntarea_countryyear`.

| Couche | Assets | Ce qu'ils font aujourd'hui |
|---|---|---|
| bronze | 16, dont les téléchargements FWI, ERA5, SPEI, BDIFF et FIRMS partitionnés | enveloppent `src/ingest/<source>.build`, reprenables fichier par fichier |
| silver | 10 (6 dimensions, 3 faits, la grille commune) | appellent les fonctions v1, valident le contrat pandera, écrivent la silver |
| gold, schéma en étoile | 13, un par modèle dbt | `dbt build`, lignage lu dans le manifeste, sources dbt résolues sur les assets silver du même nom |
| gold, modèles | 5 matérialisés (`tail_risk_country`, `attribution_countryyear`, `attribution_bayes`, `attribution_robustness`, `q1_spike_results`) et 2 non matérialisés (`features_daily`, `ignition_eval`) | ajustent, journalisent dans MLflow, écrivent `gold/model_out` |

Chaque table gold porte deux familles de checks visibles dans l'interface :
le contrat pandera (`contract`, le même objet que les assets silver et les
tests utilisent) et les tests dbt du modèle, remontés automatiquement.

Limite assumée : les sources qui ne se découpent pas (EFFIS, GWIS)
reconstruisent leur étendue complète en un appel, ce qui est bon marché grâce
aux téléchargements reprenables. Le tableau « Ingestion partition-safe » plus
bas dit quelle source est partitionnée et comment.

## dbt (`dbt/`)

Projet `wildfire`, adaptateur dbt-duckdb. Les sources sont les Parquet de la
silver et le CSV interim des émissions GWIS, lus en place par DuckDB. Chaque
modèle gold est matérialisé en Parquet dans `gold/star` (`external_root` du
profil), un fichier par table.

- 9 modèles de passage (`select *` de la silver) : les tables déjà au bon grain.
- 3 rollups en SQL qui remplacent `src/transform/rollups.py` :
  `fact_burntarea_countrymonth`, `fact_burntarea_countryyear` (avec les lignes
  d'émissions GWIS sur leur propre source), `fact_burntarea_landcover`.
- 1 table propre à la v2, `fact_season_countryyear`, absente de la référence
  de parité puisque la v1 ne l'avait pas.
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

> **Non matérialisé.** Le spike Q1 a rendu un GO réduit : le modèle
> d'occurrence reste à l'échelle du spike (Ibérie et Grèce), et le feature
> store Europe, qui ne servait que son extension, n'est pas construit
> ([ADR 0009](adr/0009-q1-scope-after-reduced-go.md)). Cette section décrit du
> code testé sur des NetCDF synthétiques, pas une table qui existe.

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
| `extremes.py` | Q2, queue des tailles de feux | épisodes de feux de plus de 500 ha (feux à 3 jours ou moins fusionnés), ajustement Pareto généralisé par pays, niveaux de retour à 2, 5, 10, 20 ans avec intervalles par bootstrap d'années entières, publiés seulement si la forme est sous 1 et jusqu'à deux fois la durée observée, périmètre UE avec au moins 8 années complètes et 30 épisodes, niveaux empiriques en regard, table de stabilité du seuil | épisodes formés comme attendu, paramètres connus retrouvés sur un échantillon simulé, règles de périmètre et de publication |
| `attribution.py` | Q3, part de la météo dans la variabilité annuelle | panel pays x année, variables en écart-type intra-pays et par unité physique, effets fixes pays, erreurs groupées par pays, composante météo et surface ajustée par pays et année, R² intra-pays, gain hors échantillon (leave-one-year-out), table de robustesse et colonne `robust` | écart-type intra-pays et pente par unité retrouvés sur un panel simulé, règle `robust`, années à saison incomplète exclues et non imputées |
| `attribution_bayes.py` | Q3, version bayésienne hiérarchique (PyMC) | même panel, intercepts et pentes partiellement mis en commun entre pays (pentes non centrées), a priori faiblement informatifs, coefficients a posteriori (moyenne, intervalle HDI à 95 %) en regard des coefficients fréquentistes, R² bayésien de la météo par pays, diagnostics (R-hat, ESS, divergences) | pente connue retrouvée sur un jeu simulé, schéma des tables, règle d'acceptation du sampler |
| `spike_results.py` | Q1, restitution du spike | remet en forme les sorties gelées de `notebooks/spike_q1/out` en tables `q1_*`, journalisées dans un run MLflow, sans rien recalculer | tables égales aux fichiers gelés |
| `ignition.py` | Q1 à l'échelle de l'Europe | cinq modèles en escalier sur le feature store, non exécuté (GO réduit) | fixtures synthétiques |
| `validation.py` | protocole commun | découpes par année tenue et par blocs spatiaux de 2°, métriques du critère pré-enregistré, fiabilité par décile, bootstrap par blocs sur les différences | partition exacte des lignes, blocs jamais scindés, intervalles qui séparent deux modèles de qualité connue |
| `tracking.py` | traçabilité | MLflow en SQLite local sans serveur, un identifiant de run sur chaque table publiée, gabarit de carte de modèle | run relu depuis le store |

### Tables lues par le rapport

Trois pages Power BI lisent `gold/model_out`, dans le dossier voisin de celui
que désigne `DataFolder` (le chemin est dérivé dans chaque requête, il reste
un seul paramètre à régler).

| Page | Tables | Asset |
|---|---|---|
| Tail risk | `tail_risk_country`, `tail_risk_levels` (niveaux de retour en format long, une ligne par pays et période), `tail_risk_threshold_stability` (forme selon le seuil, par pays) | `gold/tail_risk_country` |
| Attribution | `attribution_countryyear`, `attribution_coefficients` (nom lisible, intervalle à 95 %, multiplicateur par écart-type intra-pays, effet par unité physique, `robust`), `attribution_summary` (R² intra-pays, gain hors échantillon), `attribution_exclusions`, `attribution_robustness` | `gold/attribution_countryyear` |
| (pas de page) | `attribution_bayes_coefficients` (niveaux population, écart-type entre pays et pays : moyenne, écart-type, intervalle à 95 %, effet par unité physique, `robust`, coefficient fréquentiste en regard, R-hat, ESS), `attribution_bayes_countryyear` (composante météo avec intervalle, surface ajustée, R² bayésien par pays) ; `run_id` sur chaque ligne | `gold/attribution_bayes` |
| Validation | `q1_metrics`, `q1_criterion`, `q1_importance` ; `q1_reliability` est écrite pour le site | `gold/q1_spike_results` |

`gold/q1_spike_results` ne recalcule rien : il remet en forme les sorties
gelées et commitées du spike (`notebooks/spike_q1/out`), si bien que la page et
`VERDICT.md` ne peuvent pas diverger. Il se matérialise sans aucune donnée
locale.

La version bayésienne de Q3 (`gold/attribution_bayes`, expérience MLflow
`q3_attribution_bayes`) consomme le même panel et retrouve le signe et
l'ordre de grandeur des coefficients fréquentistes ([ADR 0010](adr/0010-pymc-windows-native.md)).
Elle tourne en Windows natif : sans compilateur C++, PyTensor compile vers
Numba (`configure_backend` le règle avant l'import de PyMC). L'asset check
`sampler_diagnostics` échoue si R-hat atteint 1,01 ou si les divergences
dépassent 1 % des tirages.

Les cartes de modèle ([Modèles](models/index.md)) disent ce que chaque sortie
permet d'affirmer et ce qu'elle ne permet pas.
