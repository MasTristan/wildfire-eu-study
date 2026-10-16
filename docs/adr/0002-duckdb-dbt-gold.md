# 0002. DuckDB et dbt pour la gold, Parquet comme contrat

Statut : proposé le 2026-10-15.

## Contexte

La gold v1 (douze tables) est produite par des fonctions pandas. Power BI
lit des Parquet. Les rollups sont de l'agrégation pure, ce que SQL exprime
plus court et teste mieux que pandas.

## Décision

dbt avec l'adaptateur dbt-duckdb. Les sources sont les Parquet de la silver,
lus en place. Chaque modèle gold est matérialisé en Parquet dans
`gold/star` (matérialisation `external`), un fichier par table : le contrat
que Power BI lit ne change pas de nature. Tests dbt sur le grain, les
valeurs admises, la non-négativité et le non-mélange des sources. dbt
n'intervient que sur la gold ; la silver (géospatial, raster) reste en
Python.

## Alternatives écartées

- SQL via DuckDB appelé depuis Python sans dbt : perd les tests déclaratifs,
  la documentation générée et le lignage colonne.
- Spark : surdimensionné pour quelques dizaines de millions de lignes sur un
  poste unique.
- Garder pandas : fonctionne, mais chaque invariant reste un test pytest
  séparé plutôt qu'une propriété du modèle.

## Conséquences

- `tests/test_dbt_gold.py` prouve l'équivalence SQL contre Python sur une
  silver synthétique ; le test de parité v1 contre v2 la prouve sur les
  données réelles.
- Le fichier DuckDB est jetable, seul le Parquet compte.
