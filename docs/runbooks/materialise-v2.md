# Procédure : matérialiser la v2 de bout en bout

Exécutant : une session Claude Code locale, Windows natif, worktree de la
branche `ccr-9f718573-2htm0s`, après la procédure du spike Q1. Même règle :
commande, attendu, vérification, conduite en cas d'écart ; une relance à
l'identique, puis arrêt et rapport avec l'erreur exacte. Les chemins de
données sont ceux de `src/lake.py` : silver = `data/processed`, gold =
`data/lake/gold`.

## 1. Environnement

```powershell
git pull --ff-only
uv sync
uv run pytest -q
make dbt-parse
```

Attendu : pytest sans `failed` ; `dbt parse` termine par `Done` et crée
`dbt/target/manifest.json`. Si `uv sync` échoue sur un paquet (dagster,
mlflow, statsmodels, dvc), rapporter le message complet.

## 2. Gold par dbt sur la silver réelle

```powershell
make dbt-build
```

Attendu : `Completed successfully`, avec 13 modèles `OK created external` et
tous les tests `PASS`. Dossier `data/lake/gold/star/` contenant 13 Parquet :
les 12 tables v1 plus `fact_season_countryyear`.

Vérification :

```powershell
uv run python -m src.contracts
uv run pytest -q tests/test_gold_parity.py
```

Attendu : `ok` sur chaque table listée par `src.contracts` (les 12 v1 lues
depuis `data/processed`), et le test de parité entièrement vert : il lit
désormais `data/lake/gold/star` et le compare à la référence v1. Un échec
de parité sur une table : rapporter la sortie `assert_frame_equal` complète,
ne pas corriger.

Écart connu : un test dbt `no_source_mixing` ou `unique_combination` en
`FAIL` signifie une anomalie de la silver, pas de dbt ; rapporter la requête
d'échec affichée par dbt.

## 3. Dagster, première matérialisation

```powershell
$env:DAGSTER_HOME = (Resolve-Path .).Path + "\.dagster"
uv run dagster asset list -m src.orchestration.definitions
```

Attendu : la liste des assets `bronze/...`, `silver/...`, `gold/...`,
sans erreur de chargement.

Matérialiser dans cet ordre, une commande par ligne, en vérifiant
`RUN_SUCCESS` à la fin de chaque :

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "silver/dim_date,silver/dim_country,silver/dim_landcover,silver/dim_source,silver/dim_firecause"
uv run dagster asset materialize -m src.orchestration.definitions --select "silver/dim_commune"
uv run dagster asset materialize -m src.orchestration.definitions --select "silver/fact_fireevent"
uv run dagster asset materialize -m src.orchestration.definitions --select "silver/fact_fireweather_countrymonth"
uv run dagster asset materialize -m src.orchestration.definitions --select "silver/fact_hotspot_countrymonth"
uv run dagster asset materialize -m src.orchestration.definitions --select "gold/*"
```

Les assets silver appellent les fonctions v1 sur les sources déjà sur disque
(`data/raw`, `data/interim`) : aucun téléchargement nouveau, sauf
`dim_country` qui interroge la Banque mondiale et `dim_commune` qui lit
BDIFF interim. Après `gold/*`, relancer :

```powershell
uv run pytest -q tests/test_gold_parity.py
```

Attendu : toujours vert. C'est la preuve que la chaîne Dagster, dbt et
contrats reproduit la v1 à l'identique.

## 4. Interface Dagster

```powershell
make dagster-dev
```

Ouvrir `http://localhost:3000`, menu Assets : les trois groupes bronze,
silver, gold et le groupe model_out, le lignage joint de bout en bout, les
assets gold avec leurs checks `contract` et dbt en vert. Prendre une capture
d'écran du graphe complet (`docs/img/dagster-lineage.png`, à commiter).
Arrêter avec Ctrl+C.

## 5. Bronze partitionnée, enregistrement des années déjà sur disque

Les fichiers existent déjà ; matérialiser enregistre les partitions sans
retélécharger.

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "bronze/cds_fwi_raw" --partition-range 2016...2026
uv run dagster asset materialize -m src.orchestration.definitions --select "bronze/era5_raw" --partition-range 1991-2000...2021-2026
uv run dagster asset materialize -m src.orchestration.definitions --select "bronze/edo_spei_raw" --partition-range 2016...2026
```

Attendu : `RUN_SUCCESS` et, dans la sortie, des chemins existants. Une
année FWI absente (consolidation EWDS en retard) échoue proprement : noter
l'année et continuer.

## 6. Feature store

### 6a. Composantes FWI complètes, 0,25°, Europe

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "bronze/cds_fwi_full_raw" --partition-range 2016...2026
```

Une requête EWDS par année, fichiers `data/raw/cds_fwi/fwi_full_<année>.zip`,
quelques centaines de Mo chacun. Lancer en tâche de fond, plusieurs heures.
L'année en cours peut manquer : noter et continuer.

### 6b. Maillage et champs statiques

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "silver/grid_cells,bronze/era5_static"
uv run python -c "import pandas as pd; c=pd.read_parquet('data/processed/grid_cells.parquet'); print(len(c), c['country_key'].value_counts().head(10).to_dict())"
```

Attendu : entre 15 000 et 25 000 mailles, `FR`, `ES`, `SE`, `NO`, `FI`,
`DE`, `PL`, `IT`, `TR` parmi les premiers effectifs. `FR` absent : rapporter
immédiatement (régression du piège Natural Earth).

### 6c. Météo journalière Europe, une requête CDS par mois

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "bronze/era5_daily_raw" --partition-range 2015-12...2026-09
```

Environ 130 requêtes de 3 à 6 minutes, une nuit. Reprenable à l'identique.
Attendu : `data/raw/era5_daily/era5_daily_europe_<année>_<mois>.nc`, 30 à
60 Mo chacun.

Vérification sur un mois :

```powershell
uv run python -c "import xarray as xr; ds=xr.open_dataset('data/raw/era5_daily/era5_daily_europe_2022_07.nc'); print(list(ds.data_vars), ds.sizes)"
```

Attendu : les sept variables, dimensions `time: 31`, `latitude: 153`,
`longitude: 281`.

### 6d. Table maille x jour, par année

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "gold/features_daily" --partition-range 2016...2025
```

Attendu : `data/lake/gold/features/daily_<année>.parquet`, 4 à 8 millions de
lignes par année, et dans les métadonnées de chaque run `positives` entre
1 000 et 15 000.

Vérification :

```powershell
uv run python -c "import pandas as pd; d=pd.read_parquet('data/lake/gold/features/daily_2022.parquet'); print(d.shape); print(d.isna().mean().round(3).to_string()); print(d.groupby(d['date'].dt.month)['y_a'].sum().to_dict())"
```

Attendu : nuls sous 2 % pour les colonnes FWI et météo, positifs
concentrés sur juin à septembre, `tp_7d` et `spei3` présents. Un taux de
nuls supérieur à 10 % sur une colonne : rapporter la colonne et s'arrêter.

## 7. Modèles Q2 et Q3 sur tables réelles

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "gold/tail_risk_country,gold/attribution_countryyear"
uv run python -c "import pandas as pd; t=pd.read_parquet('data/lake/gold/model_out/tail_risk_country.parquet'); print(t[['country_key','n_fires','n_exceedances','shape','rl_10y','return_period_10k_ha_years']].sort_values('n_fires', ascending=False).head(12).round(3).to_string())"
uv run python -c "import pandas as pd; c=pd.read_parquet('data/lake/gold/model_out/attribution_coefficients.parquet'); print(c.round(4).to_string()); a=pd.read_parquet('data/lake/gold/model_out/attribution_countryyear.parquet'); print(a.groupby('year')[['burnt_area_ha','adjusted_area_ha']].sum().round(0))"
```

Attendu : pour Q2, `shape` entre 0 et 1 pour `PT`, `ES`, `GR`, `IT`, avec
`rl_10y` supérieur à `rl_2y` ; pour Q3, `days_high_season` à coefficient
positif, un `run_id` non vide dans les deux tables. Tout `shape` négatif
sous -0,5 ou coefficient `days_high_season` négatif : rapporter.

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Ouvrir `http://localhost:5000` : deux expériences `q2_tail_risk` et
`q3_attribution`, un run chacune avec paramètres et métriques. Capture
d'écran `docs/img/mlflow-runs.png`. Ctrl+C.

## 7b. Q1 sur le feature store (après le verdict du spike)

```powershell
uv run dagster asset materialize -m src.orchestration.definitions --select "gold/ignition_eval"
```

Durée : plusieurs heures (une année tenue à la fois, cinq modèles, blocs
spatiaux). Attendu : `data/lake/gold/model_out/ignition/` avec
`metrics_a.csv`, `reliability_a.csv`, `bootstrap_a.csv`, `shap_a.csv`,
`tuned_a.csv`, `spatial_blocks_a.csv`, `predictions_a.parquet`,
`criterion_a.json`, plus `ignition_monthly_country.parquet` et
`ignition_metrics.parquet` dans `model_out/`.

Vérification :

```powershell
uv run python -c "import pandas as pd; m=pd.read_csv('data/lake/gold/model_out/ignition/metrics_a.csv'); print(m.pivot(index='model', columns='held_year', values='pr_auc').round(4)); b=pd.read_csv('data/lake/gold/model_out/ignition/spatial_blocks_a.csv'); print(b)"
type data\lake\gold\model_out\ignition\criterion_a.json
```

Attendu : cinq modèles par année tenue, aucune métrique `nan`, et pour
chaque année `pr_auc_blocked` du même ordre que la PR-AUC du modèle complet
(un écart supérieur à 30 % relatif signale une autocorrélation spatiale
exploitée : le rapporter, c'est un résultat, pas une erreur).

## 8. Bascule Power BI

Dans `powerbi/WildfireEU.SemanticModel/definition/expressions.tmdl`, faire
pointer `DataFolder` sur le chemin absolu de `data/lake/gold/star`. Ouvrir
`powerbi/WildfireEU.pbip` dans Power BI Desktop, Accueil > Actualiser.

Attendu : actualisation sans erreur, 12 tables chargées. Parcourir les dix
pages et comparer à une capture de référence prise avant la bascule : aucun
chiffre des cartons de la page Scale ne change (surface brûlée totale,
nombre de feux, émissions). Un chiffre qui change alors que la parité est
verte : rapporter immédiatement, revenir au chemin précédent.

Commiter `expressions.tmdl` uniquement si le chemin ne contient pas de
nom d'utilisateur ; sinon rapporter et ne pas commiter.

## 9. DVC, remote gratuit

1. Créer un bucket sur Cloudflare R2 ou Backblaze B2 (compte gratuit,
   quota vérifié le jour même sur la page de tarification, à noter dans le
   rapport). Nom du bucket : `wildfire-eu-study-lake`.
2. Renseigner l'accès localement, jamais dans le dépôt :

```powershell
uv run dvc remote modify --local storage endpointurl https://<endpoint-du-fournisseur>
uv run dvc remote modify --local storage access_key_id <clé>
uv run dvc remote modify --local storage secret_access_key <secret>
```

3. Suivre et pousser la gold et les sorties de modèles :

```powershell
uv run dvc add data/lake/gold/star data/lake/gold/model_out data/lake/gold/features
uv run dvc push
git add data/lake/gold/*.dvc .dvc/config
git commit -m "data: track gold star, features and model outputs with DVC"
git push origin ccr-9f718573-2htm0s
```

Attendu : `dvc push` termine sans erreur et `dvc status -c` répond
`Cache and remote 'storage' are in sync`. Le fichier `.dvc/config.local`
reste hors git (déjà ignoré par DVC).

## 10. Liste d'acceptation

Cocher chaque ligne dans le rapport final :

- [ ] `uv run pytest -q` sans `failed`, et le nombre de `skipped` a baissé par rapport à l'étape 1 (les tests de données tournent).
- [ ] `make dbt-build` : 13 modèles, tous les tests `PASS`.
- [ ] `tests/test_gold_parity.py` vert sur `data/lake/gold/star`.
- [ ] Graphe Dagster complet capturé, checks gold verts.
- [ ] `features/daily_2016.parquet` à `daily_2025.parquet` présents.
- [ ] `tail_risk_country.parquet` et `attribution_countryyear.parquet` présents avec `run_id`.
- [ ] MLflow : deux expériences, captures.
- [ ] Power BI actualisé sur `gold/star`, chiffres inchangés.
- [ ] `dvc status -c` en synchronisation.
- [ ] Commits poussés ; `data/`, `mlruns/`, `.dagster/`, `docs/internal/` hors git.
