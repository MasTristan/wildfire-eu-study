# Procédure : terminer le spike Q1

Exécutant : une session Claude Code locale, Windows natif, dans le worktree
de la branche `ccr-9f718573-2htm0s`. Chaque étape donne la commande, le
résultat attendu, la vérification et la conduite à tenir en cas d'écart.
Règle générale : une étape dont la vérification échoue se relance une fois
à l'identique ; si elle échoue encore, on s'arrête et on rapporte le message
d'erreur exact. On ne modifie ni le protocole, ni le critère, ni les scripts
de modélisation.

## 0. État de départ

```powershell
git pull --ff-only
git log --oneline -1
uv sync
uv run pytest -q
```

Attendu : la branche est au moins à `9169409`, `uv sync` ne signale aucune
erreur, pytest affiche `N passed, M skipped` sans `failed`. Si un test échoue,
rapporter sa sortie complète et s'arrêter.

## 1. Météo journalière (CDS horaire, amendement 2)

```powershell
uv run python notebooks/spike_q1/s02b_download_era5_hourly.py
```

Attendu : à la fin, 120 fichiers `data/raw/spike_q1/era5_daily_{iberia|greece}_{2019..2023}_{01..12}.nc`.
Durée : 3 à 6 minutes par requête, lancer en tâche de fond. Le script est
reprenable : relancer la même commande après toute interruption.

Vérification :

```powershell
uv run python -c "import glob; f=glob.glob('data/raw/spike_q1/era5_daily_*.nc'); print(len(f))"
uv run python -c "import xarray as xr; ds=xr.open_dataset('data/raw/spike_q1/era5_daily_iberia_2022_07.nc'); print(list(ds.data_vars), ds.sizes); print(float(ds['t2m_max'].max()), float(ds['rh'].min()), float(ds['tp_mm'].sum()))"
```

Attendu : `120` ; les variables `t2m_c, t2m_max, rh, rh_min, wind, wind_mean, tp_mm` ;
31 pas de temps en juillet ; t2m_max entre 25 et 48, rh min entre 0 et 100,
tp_mm somme positive. Une valeur hors de ces bornes : rapporter, ne pas corriger.

Écart fréquent : une requête CDS rejetée pour file pleine est réessayée toutes
les 120 secondes par le script lui-même. Un fichier horaire corrompu (erreur
NetCDF à l'ouverture) : supprimer `data/raw/spike_q1/era5h_<zone>_<année>_<mois>.nc`
et relancer la commande.

## 2. Jeu de données

```powershell
uv run python notebooks/spike_q1/s05_build_dataset.py
```

Attendu dans la sortie : `weather: CDS hourly reduction, 120 monthly files`,
puis `mesh: N cells ({'ES': ..., 'PT': ..., 'GR': ...})` avec N entre 1 100 et
1 400, puis `daily table: R rows, a-positives A, b-positives B`.

Vérification :

```powershell
uv run python -c "import pandas as pd; d=pd.read_parquet('data/interim/spike_q1/daily.parquet'); print(d.shape); print(d.isna().mean().round(3).to_string()); print(d.groupby('year')[['y_a','y_b']].sum())"
```

Attendu : R proche de N x 1826 ; aucune colonne météo avec plus de 5 % de
nuls hors `tp_7d` (nul les premiers jours) et `spei3` (nul la première décade) ;
des positifs chaque année pour les deux labels, concentrés sur 2022 pour (a)
en Espagne. Si `weather:` indique Open-Meteo au lieu de CDS, l'étape 1 est
incomplète : y revenir.

## 3. Entraînement et évaluation

```powershell
uv run python notebooks/spike_q1/s06_train_evaluate.py a b
```

Durée : minutes à quelques dizaines de minutes. Attendu : pour chaque label,
les lignes `lgbm_fwi`, `lgbm_fwi_space`, `lgbm_full` avec leurs réglages, puis
le tableau des métriques (cinq modèles x deux années) et le JSON du critère.

Vérification des sorties dans `notebooks/spike_q1/out/` :
`metrics_a.csv`, `metrics_b.csv`, `reliability_a.csv`, `reliability_b.csv`,
`bootstrap_a.csv`, `bootstrap_b.csv`, `importance_a.csv`, `importance_b.csv`,
`criterion_a.json`, `criterion_b.json`, `reliability_a.png`, `reliability_b.png`.

```powershell
uv run python -c "import pandas as pd; m=pd.read_csv('notebooks/spike_q1/out/metrics_a.csv'); print(m.pivot(index='model', columns='year', values=['pr_auc','brier']).round(5))"
type notebooks\spike_q1\out\criterion_a.json
```

Attendu : dix lignes de métriques par label, PR-AUC entre 0 et 1, Brier
positif, `criterion_a.json` avec `all_pass` vrai ou faux. Aucune métrique
`nan`. Un `nan` : rapporter le tableau complet et s'arrêter.

## 4. Concordance des labels

```powershell
uv run python notebooks/spike_q1/s07_concordance.py
```

Attendu : `concordance.csv` (une ligne par année plus `all`) et
`concordance_offsets.csv` dans `out/`, avec `a_seen_by_b_tolerant` supérieur
à `a_seen_by_b_same_day`.

## 5. VERDICT.md

Écrire `notebooks/spike_q1/VERDICT.md` avec exactement ces sections, dans cet ordre :

1. **Verdict** : une phrase, `GO` ou `GO réduit`, recopiée de `criterion_a.json` (`all_pass`).
2. **Critère** : le tableau `brier_rel_improvement` et `pr_auc_gain` par année de test, avec les seuils 0,10 et 0,05, et le gain relatif de PR-AUC (`(m4 - m3) / m3`) à côté du gain absolu.
3. **Escalier de skill** : le tableau complet des cinq modèles pour les deux années et les deux labels (copie de `metrics_a.csv` et `metrics_b.csv` en tableau Markdown).
4. **Incertitude** : `bootstrap_a.csv` recopié.
5. **Fiabilité** : les deux figures PNG référencées.
6. **Facteurs** : les dix premières lignes de `importance_a.csv`.
7. **Labels** : `concordance.csv` et `concordance_offsets.csv` recopiés, avec une phrase sur le décalage dominant.
8. **Données** : nombre de mailles par pays, lignes, positifs par année ; source météo : CDS horaire réduit en journalier (amendement 2), variables `t2m_c, t2m_max, rh, rh_min, wind, wind_mean, tp_mm` ; les substitutions déclarées (champs statiques ERA5 à la place de CORINE, GHSL omis, SPEI dernière décade close).
9. **Temps de calcul** : contenu de `data/interim/spike_q1/timings/*.json` agrégé par étape.
10. **Limites** : recopier la section « Known limits » de REVIEW.md et y ajouter tout écart constaté pendant l'exécution.

Aucune interprétation au-delà de ce que les tableaux montrent. Aucune
modification des seuils.

## 6. Clôture

```powershell
git add notebooks/spike_q1/out notebooks/spike_q1/VERDICT.md
git status --short
git commit -m "spike(q1): verdict, metrics, reliability, concordance and timings"
git push origin ccr-9f718573-2htm0s
git log --oneline -1
```

Attendu : seuls `out/` et `VERDICT.md` dans le commit ; rien sous `data/`
ni `docs/internal/`. Rapporter le hash du commit, le verdict et les deux
tableaux de métriques.

## 7. Après le verdict : table des hotspots et parité

```powershell
uv run python -c "from src.ingest import firms; df=firms.aggregate_raw(); print(df.shape, df['month_key'].min(), df['month_key'].max()); df.to_parquet('data/processed/fact_hotspot_countrymonth.parquet', index=False)"
make validate
```

Attendu : `fact_hotspot_countrymonth  ok` dans la sortie de `make validate`,
et une étendue de mois qui couvre toutes les années présentes dans
`data/raw/firms/`. Si `data/raw/firms/` ne contient que l'année courante,
l'étendue est celle-là : c'est normal, le rapporter.

```powershell
uv run python scripts/freeze_v1_reference.py
uv run pytest -q tests/test_gold_parity.py
```

Attendu : la commande de gel recopie les 12 tables et réécrit
`data/reference/v1/manifest.json` ; le test de parité passe intégralement.
Puis éditer `data/reference/v1/manifest.json` : ajouter sous
`tables.fact_hotspot_countrymonth` la clé
`"note": "rebuilt 2026-10 with the FIRMS window fix (src/ingest/firms.py window_url), differs from the v1.0 release asset"`.

```powershell
git add data/reference/v1/manifest.json
git commit -m "chore(parity): re-freeze fact_hotspot_countrymonth after the FIRMS window fix"
git push origin ccr-9f718573-2htm0s
gh release upload v1.0 dist/gold-v1.zip --clobber
```

## 8. Vérification GWIS

```powershell
uv run python src/ingest/gwis.py
```

Attendu : `Wrote data/interim/gwis_gfas_emissions.csv (... rows, 2003-01 to 20xx-xx, ...)`.
Si HTTP 404 : ouvrir `https://gwis.jrc.ec.europa.eu/apps/country.profile/` dans
un navigateur, relever le nom exact du fichier d'émissions GFAS proposé au
téléchargement, et le rapporter sans modifier le code.
