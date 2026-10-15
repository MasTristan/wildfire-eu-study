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
