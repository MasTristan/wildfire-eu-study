# Architecture decision records

Une décision par fichier, numérotée, jamais réécrite : une décision qui
change donne un nouvel ADR qui remplace l'ancien. Statuts : proposé,
accepté, remplacé par NNNN.

| ADR | Décision | Statut |
|---|---|---|
| [0001](0001-dagster-orchestration.md) | Dagster comme orchestrateur, assets par couche | accepté (verdict du spike : plateforme inchangée, ADR 0009) |
| [0002](0002-duckdb-dbt-gold.md) | DuckDB et dbt pour la couche gold, Parquet comme contrat | proposé |
| [0003](0003-common-grid.md) | Grille commune 0,25°, celle du FWI | proposé |
| [0004](0004-mlflow-file-dvc.md) | MLflow en store fichier, DVC pour les données, pas de serveur | proposé |
| [0005](0005-windows-native-wsl-pymc.md) | Pipeline sous Windows natif, WSL2 réservé à PyMC | proposé, partie WSL2 remplacée par 0010 |
| [0006](0006-era5-daily-via-open-meteo.md) | ERA5 journalier via l'archive Open-Meteo | remplacé par 0007 à l'échelle du maillage |
| [0007](0007-era5-hourly-cds-monthly.md) | ERA5 journalier réduit localement depuis l'horaire CDS, une requête par mois | accepté |
| [0008](0008-arco-era5-fallback.md) | ARCO-ERA5 en repli du CDS pour l'horaire ERA5, sous vérification d'identité | accepté |
| [0009](0009-q1-scope-after-reduced-go.md) | Périmètre de Q1 et de la phase 1 après le GO réduit du spike | accepté |
| [0010](0010-pymc-windows-native.md) | PyMC en Windows natif (backend Numba), sans WSL2 | accepté |
