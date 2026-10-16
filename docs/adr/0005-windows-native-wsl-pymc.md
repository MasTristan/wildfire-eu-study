# 0005. Pipeline sous Windows natif, WSL2 réservé à PyMC

Statut : proposé le 2026-10-15.

## Contexte

La machine est sous Windows 11 avec un seul disque. Power BI Desktop lit
la gold par un chemin Windows. DuckDB, LightGBM, dbt-duckdb et Dagster
fonctionnent nativement sous Windows ; PyMC exige une chaîne C++ que
conda-forge sous WSL2 fournit sans friction.

## Décision

Tout le pipeline (ingestion, lake, dbt, Dagster, LightGBM) tourne sous
Windows natif. WSL2 ne sert qu'à PyMC pour Q3, qui lit une table pays x
année de quelques centaines de lignes et écrit ses sorties dans
`gold/model_out`. `.wslconfig` plafonne WSL2 à 12 Go.

## Alternatives écartées

- Tout sous WSL2 : impose soit des entrées-sorties lentes via /mnt/c, soit
  une copie de la gold vers un chemin Windows à chaque matérialisation.
- Tout sous Windows, PyMC inclus : installation fragile du compilateur,
  temps perdu pour un seul modèle.

## Conséquences

- Dagster en exécuteur in-process en développement (multiprocess fragile
  sous Windows).
- Le disque virtuel WSL2 ne rétrécit pas seul : surveillé et compacté.
