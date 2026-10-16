# Architecture decision records

Une décision par fichier, numérotée, jamais réécrite : une décision qui
change donne un nouvel ADR qui remplace l'ancien. Statuts : proposé,
accepté, remplacé par NNNN.

| ADR | Décision | Statut |
|---|---|---|
| [0001](0001-dagster-orchestration.md) | Dagster comme orchestrateur, assets par couche | proposé, confirmation au verdict du spike Q1 |
| [0002](0002-duckdb-dbt-gold.md) | DuckDB et dbt pour la couche gold, Parquet comme contrat | proposé |
| [0003](0003-common-grid.md) | Grille commune 0,25°, celle du FWI | proposé |
| [0004](0004-mlflow-file-dvc.md) | MLflow en store fichier, DVC pour les données, pas de serveur | proposé |
| [0005](0005-windows-native-wsl-pymc.md) | Pipeline sous Windows natif, WSL2 réservé à PyMC | proposé |
| [0006](0006-era5-daily-via-open-meteo.md) | ERA5 journalier via l'archive Open-Meteo, pas le CDS | accepté |
