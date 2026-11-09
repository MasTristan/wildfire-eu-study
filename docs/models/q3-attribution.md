# Carte de modèle : Q3, part de la météo dans la surface brûlée

Deux estimations de la même relation, dans la même carte :

| Version | Expérience MLflow | `run_id` | Tables |
|---|---|---|---|
| Fréquentiste (MCO, effets fixes pays) | `q3_attribution`, run `ols_fe_20261109` | `d6dea9f655d043a6900f9d98de10dfd6` | `attribution_coefficients`, `attribution_countryyear`, `attribution_summary` |
| Bayésienne hiérarchique (PyMC) | `q3_attribution_bayes`, run `hier_20261109` | `7027f2aadd954abeb47823ec9d6e8a61` | `attribution_bayes_coefficients`, `attribution_bayes_countryyear` |

Les deux runs datent du 2026-11-09 ; chaque ligne des tables porte le `run_id` de
sa version.

## Question

Quand un pays brûle beaucoup plus qu'une année ordinaire, quelle part de l'écart
s'explique par la météo de la saison ? Et quelle surface aurait brûlé la même année
avec la météo moyenne du pays ?

La question porte sur les écarts d'un pays à lui-même d'une année sur l'autre.
Elle ne compare pas les pays entre eux : un pays grand et boisé brûle plus qu'un
petit, la méthode neutralise ce niveau propre à chaque pays.

## Données

- Un tableau pays x année, 2016 à 2025. La surface brûlée annuelle par pays vient de la table
  `fact_burntarea_countryyear`, source `EFFIS_RDA`. La variable à expliquer est
  son logarithme, `log(1 + hectares)`.
- Quatre variables de saison, calculées sur la table `fact_fireweather_countrymonth` :
  - `days_high_season` : nombre de jours de danger d'incendie élevé, juin à septembre ;
  - `spei_spring` : bilan hydrique SPEI moyen, mars à mai (négatif = sec) ;
  - `precip_anom_season` : anomalie de pluie moyenne, juin à septembre ;
  - `temp_season` : température moyenne, juin à septembre.
- Une année n'entre que si les quatre mois de juin à septembre sont présents ; rien
  n'est imputé.
- Chaque variable est standardisée sur l'ensemble du panel : un coefficient se lit
  « pour un écart-type du panel », qui n'est pas l'écart-type d'une année à l'autre
  dans un pays (voir Limites).
- Fréquentiste : 360 observations, 43 pays. Bayésien : 356 observations, 40 pays,
  car il faut au moins 4 années par pays (Géorgie, Moldavie et Russie sont écartées).

## Méthode

**Fréquentiste.** Une régression linéaire (moindres carrés) du log de la surface sur
les quatre variables et une constante par pays (effets fixes), erreurs standard
groupées par pays. Un coefficient b se lit : un écart-type de plus de la variable
multiplie la surface brûlée du même pays par exp(b). Pour chaque pays et chaque
année, la **composante météo** est la somme des coefficients multipliés par l'écart
de chaque variable à la moyenne du pays. La **surface ajustée** est la surface que
l'année aurait donnée avec cette composante à zéro, c'est-à-dire avec la météo
moyenne du pays : `expm1(log_surface - composante_meteo)`.

**Bayésienne hiérarchique.** Même tableau, même standardisation, mais chaque pays
a sa propre pente pour chaque variable, tirée d'une loi commune :

```
log_surface[pays, an] = alpha[pays] + somme_k( beta[pays, k] * z[pays, an, k] ) + bruit
beta[pays, k] = mu_beta[k] + tau[k] * b_raw[pays, k]
```

`mu_beta[k]` est la pente moyenne sur l'ensemble des pays, celle que la version
fréquentiste estime d'un bloc. `tau[k]` mesure combien les pays s'écartent de cette
pente moyenne. Conséquence pratique : un pays qui n'a que dix observations pour quatre
pentes emprunte de la force aux autres (on parle de mise en commun partielle) au lieu
de produire une pente instable. Le résultat n'est pas un nombre mais une distribution
des valeurs plausibles de chaque coefficient, compte tenu des données et des a priori.
On en publie la moyenne et un **intervalle HDI à 94 %**, le plus étroit des intervalles
qui contiennent 94 % de la distribution.

A priori, tous faiblement informatifs : `mu_beta ~ N(0, 1)`, `tau ~ HalfNormal(0,5)`,
`mu_alpha ~ N(8, 3)`, `sigma_alpha ~ HalfNormal(2)`, `sigma ~ HalfNormal(1,5)`. Les
raisons de chaque choix sont dans le docstring de `src/models/attribution_bayes.py`.
Les pentes sont non centrées (reparamétrisation qui évite les divergences du
sampler quand il y a peu de données par pays), les constantes sont centrées.

Échantillonnage : sampler NUTS de PyMC, 4 chaînes de 1000 tirages après 1000 de
chauffe, `target_accept = 0,99`, backend Numba sans compilateur C
([ADR 0010](../adr/0010-pymc-windows-native.md)).

## Validation

- **Tests** : la version fréquentiste retrouve le signe et l'ordre de grandeur d'un
  signal synthétique ; la version bayésienne retrouve des pentes connues sur un jeu
  simulé ; les années à saison incomplète sont écartées ; le schéma des tables est
  vérifié (`tests/test_models_attribution.py`, `tests/test_models_attribution_bayes.py`).
- **Diagnostics du sampler**, règle d'acceptation de l'ADR 0010 reprise par l'asset
  check `sampler_diagnostics` : R-hat maximal 1,0062 (limite 1,01), 0 divergence,
  ESS bulk minimal 566, ESS tail minimal 979. Le R-hat compare la variance entre
  chaînes à la variance dans chaque chaîne ; proche de 1, les chaînes racontent la
  même histoire. L'ESS est le nombre de tirages indépendants équivalents.
- **Comparaison des deux méthodes sur les mêmes lignes** : première figure. Les
  coefficients fréquentistes tracés sont ceux recalculés sur les 356 lignes du
  modèle bayésien (colonnes `freq_*` de `attribution_bayes_coefficients`).

![Coefficients fréquentistes et bayésiens avec leurs intervalles](../img/models/q3_coefficients.png)

- **Ajustement dans l'échantillon** : deuxième figure, surface observée contre
  surface ajustée de la météo, par année.

![Surface brûlée observée et ajustée de la météo, par année, 40 pays](../img/models/q3_observed_vs_adjusted.png)

- **Validation hors échantillon, version fréquentiste.** Le run MLflow `d6dea9f6...`
  enregistre un contrôle en laissant une année de côté (`leave_one_year_out_r2`) :
  on ajuste sur neuf années, on prédit la dixième, et on compare à la prédiction naïve
  « la moyenne du pays ». Erreur quadratique moyenne (RMSE, en unités de log de
  surface), pondérée par le nombre d'observations : 1,141 pour le modèle contre 1,330
  pour la moyenne du pays, soit 14 % de moins. Le modèle fait mieux dans 9 des 10 années
  laissées de côté, moins bien en 2019 (0,958 contre 0,932). Ce contrôle n'existe pas
  pour la version bayésienne.

## Limites

**Cette attribution est statistique, pas causale : elle décrit comment la surface
brûlée d'un pays varie avec la météo de sa saison sur dix ans, avec 356 observations
pour 40 pays, et rien dans le modèle ne sépare la météo de ce qui l'accompagne.**
Concrètement :

- **Variance non expliquée.** Les quatre variables expliquent environ 30 % de la
  variabilité d'un pays d'une année sur l'autre (R² intra-pays 0,305, mesuré dans
  l'échantillon). Les 70 % restants peuvent venir des départs de feu, de l'activité
  humaine, de la gestion ou d'autres facteurs ; le modèle ne les mesure pas. Le
  R² est calculé dans l'échantillon ; le contrôle hors échantillon est dans la section
  Validation.
- **Un écart-type du panel n'est pas une année ordinaire.** L'écart-type utilisé
  mêle les écarts entre pays et les écarts d'une année à l'autre dans un même pays. Or le
  modèle ne s'appuie que sur les seconds. Sur les 360 lignes, l'écart-type global des
  jours de danger vaut 44,4 jours contre 11,6 d'écart-type à l'intérieur d'un pays (rapport 0,26) ; pour la
  température, 4,51 contre 0,58 (rapport 0,13) ; pour le SPEI et la pluie, les deux
  sont proches (0,71 contre 0,67, 31,1 contre 29,4). Les multiplicateurs de la section
  Résultats sont donc **par écart-type du panel**. Pour un écart-type intra-pays, le multiplicateur
  bayésien serait environ x 1,37 pour les jours de danger, x 0,68 pour le SPEI, x 0,92 pour la
  pluie et x 1,06 pour la température.
- **Variables liées entre elles.** Le nombre de jours de danger élevé est lui-même une
  fonction de la température, de la pluie et d'autres champs. Le coefficient de la
  température est donc lu « à nombre de jours de danger donné », il n'est pas
  l'effet total de la température.
- **Températures et pluie non tranchées.** Les intervalles de ces deux variables
  contiennent zéro dans les deux méthodes. Le seul effet net est celui des jours de
  danger élevé et du bilan hydrique de printemps.
- **Pentes différentes selon les pays.** La pente moyenne cache des écarts : l'écart-type
  entre pays de la pente de la pluie est 0,30 (intervalle 0,05 à 0,54), seul des quatre à
  exclure nettement zéro. Un pays donné peut s'écarter de la pente moyenne.
- **Surface ajustée, mise en garde d'usage.** Elle se calcule sur l'échelle logarithmique
  puis revient en hectares. La somme de plusieurs pays d'une surface ajustée n'est
  pas une espérance de la surface totale ; c'est un repère visuel pour comparer
  une année à une météo moyenne.
- **Deux estimations fréquentistes coexistent.** La table autonome
  (`attribution_coefficients`, 43 pays, 360 lignes) et la comparaison qui accompagne
  le modèle bayésien (40 pays, 356 lignes) donnent des chiffres voisins mais
  différents.
- **Une seule source de surface.** Seul le registre EFFIS `EFFIS_RDA` alimente le
  panel.

## Résultats

Même échantillon de 356 observations et 40 pays pour les deux méthodes.
Coefficients par écart-type du panel de la variable, sur le log de la surface :

| Variable | Fréquentiste, 95 % | Bayésien, moyenne et HDI 94 % | Multiplicateur bayésien | P(coef > 0) |
|---|---|---|---|---|
| Jours de danger élevé (juin à septembre) | 1,156 [0,612 ; 1,699] | 1,195 [0,621 ; 1,670] | x 3,30 | 1,000 |
| Bilan hydrique SPEI (mars à mai) | -0,394 [-0,541 ; -0,247] | -0,406 [-0,528 ; -0,287] | x 0,67 | 0,000 |
| Anomalie de pluie (juin à septembre) | -0,108 [-0,277 ; 0,062] | -0,085 [-0,285 ; 0,106] | x 0,92 | 0,197 |
| Température moyenne (juin à septembre) | 0,871 [-0,215 ; 1,956] | 0,491 [-0,130 ; 1,184] | x 1,63 | 0,920 |

Lecture :

- **Accord pour trois variables sur quatre.** Mêmes signes, écart entre les deux
  estimations de 0,04 ou moins pour les jours de danger, le SPEI et la pluie. Un
  écart-type du panel de jours de danger en plus multiplie la surface d'un pays par 3,3
  environ (bayésien), 3,2 (fréquentiste, exp(1,156)) ; pour un écart-type intra-pays, 1,37
  environ (voir Limites).
- **Désaccord de taille sur la température.** 0,87 en fréquentiste, 0,49 en
  bayésien : le signe est le même, la taille diffère de 44 %, et l'intervalle
  bayésien est plus étroit (1,31 contre 2,17 de large). Les deux contiennent
  zéro. La probabilité a posteriori que la pente soit positive est 0,92.
  C'est le comportement attendu d'un modèle hiérarchique qui ramène une pente mal
  déterminée vers la pente commune, mais cette explication n'est pas testée ici.
- **Table autonome (43 pays, 360 lignes).** Jours de danger 1,120, SPEI -0,388, pluie
  -0,101 (p = 0,25), température 1,015 (p = 0,07).
- **Hétérogénéité entre pays** (écart-type des pentes, bayésien) : jours de danger
  0,26, SPEI 0,11, pluie 0,30, température 0,34.
  Seule la pluie a un intervalle qui exclut nettement zéro.
- **Part de variance.** R² intra-pays de la météo : 0,305 fréquentiste (360 lignes),
  0,315 bayésien global, 0,309 pour la version fréquentiste sur les 356 lignes. Par
  pays, le R² bayésien a pour médiane 0,26 et va de 0,06 à 0,53.

Observé et ajusté, somme des 40 pays, en milliers d'hectares (bayésien / fréquentiste) :

| Année | Observé | Ajusté bayésien | Ajusté fréquentiste |
|---|---|---|---|
| 2017 | 1 369 | 680 | 765 |
| 2018 | 203 | 456 | 461 |
| 2023 | 895 | 1 207 | 1 195 |
| 2024 | 1 828 | 987 | 925 |
| 2025 | 2 206 | 1 227 | 1 325 |

Les années 2017, 2024 et 2025 sont celles où la surface observée dépasse le plus la
surface ajustée de la météo : en 2025, 2 206 contre 1 227 (bayésien), soit 44 % de moins
avec une météo moyenne. La série ajustée ne devient pas plate : elle va de 456 à 1 227
en bayésien, la météo n'explique pas tout.

À prendre ainsi : les jours de danger et le déficit hydrique de printemps vont avec
des surfaces brûlées plus grandes, avec la même intensité dans les deux
méthodes. Les valeurs de la température et de la pluie ne sont pas tranchées.

## Reproduire

Depuis la racine du dépôt (tables locales dans `data/lake/gold/model_out/`).

Tableau des coefficients, comparaison fréquentiste et bayésienne (356 lignes), `P(>0)`,
multiplicateurs, R-hat :

```bash
uv run python -c "import pandas as pd; b=pd.read_parquet('data/lake/gold/model_out/attribution_bayes_coefficients.parquet'); print(b[b.level=='population'][['metric','freq_coef','freq_ci_low','freq_ci_high','mean','hdi_low','hdi_high','area_multiplier','prob_positive','r_hat','ess_bulk','run_id']].round(3).to_string()); print(b[b.level=='between_country_sd'][['metric','mean','hdi_low','hdi_high']].round(3).to_string()); print(b.r_hat.max())"
```

Version fréquentiste autonome (43 pays, 360 lignes) et son R² :

```bash
uv run python -c "import pandas as pd; M='data/lake/gold/model_out/'; print(pd.read_parquet(M+'attribution_coefficients.parquet').round(4).to_string()); print(pd.read_parquet(M+'attribution_summary.parquet').T)"
```

Contrôle hors échantillon (RMSE du modèle et de la moyenne du pays, par année laissée
de côté, puis pondérées) :

```bash
uv run python -c "import json, pandas as pd, numpy as np; j=json.load(open('mlruns/artifacts/d6dea9f655d043a6900f9d98de10dfd6/artifacts/leave_one_year_out.json')); d=pd.DataFrame(j['data'],columns=j['columns']); print(d.round(3).to_string()); w=d.n; print(np.sqrt((d.rmse_model**2*w).sum()/w.sum()).round(3), np.sqrt((d.rmse_country_mean**2*w).sum()/w.sum()).round(3), (d.rmse_model<d.rmse_country_mean).sum())"
```

Écart-type global et écart-type intra-pays de chaque variable (360 lignes), et multiplicateur
bayésien par écart-type intra-pays :

```bash
uv run python -c "import pandas as pd, numpy as np; from src import lake; from src.models import attribution as a; p=a.build_panel(pd.read_parquet(lake.GOLD_STAR/'fact_fireweather_countrymonth.parquet'), pd.read_parquet(lake.GOLD_STAR/'fact_burntarea_countryyear.parquet')).dropna(subset=a.METRICS); b=pd.read_parquet(lake.GOLD_MODEL_OUT/'attribution_bayes_coefficients.parquet'); b=b[b.level=='population'].set_index('metric')['mean']; print(len(p), p.country_key.nunique()); [print(m, round(p[m].std(ddof=0),3), round(np.sqrt(((p[m]-p.groupby('country_key')[m].transform('mean'))**2).mean()),3), round(float(np.exp(b[m]*np.sqrt(((p[m]-p.groupby('country_key')[m].transform('mean'))**2).mean())/p[m].std(ddof=0))),2)) for m in a.METRICS]"
```

Totaux par année de la dernière figure (observé, ajusté bayésien, ajusté fréquentiste, 40 pays) :

```bash
uv run python -c "import pandas as pd; M='data/lake/gold/model_out/'; b=pd.read_parquet(M+'attribution_bayes_countryyear.parquet'); f=pd.read_parquet(M+'attribution_countryyear.parquet'); f=f[f.country_key.isin(b.country_key.unique())]; print(pd.DataFrame({'obs':b.groupby('year').burnt_area_ha.sum()/1e3,'adj_bayes':b.groupby('year').adjusted_area_ha.sum()/1e3,'adj_freq':f.groupby('year').adjusted_area_ha.sum()/1e3}).round(0)); print(len(b), b.country_key.nunique(), len(f)); g=b.groupby('country_key')['r2_weather'].first(); print(g.median().round(3), g.min().round(3), g.max().round(3))"
```

Diagnostics du sampler, paramètres et durée du run bayésien :

```bash
uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlruns/mlflow.db'); r=mlflow.get_run('7027f2aadd954abeb47823ec9d6e8a61'); print(r.data.params); print(r.data.metrics)"
```

Reconstruire les tables (nouveaux `run_id`), puis les figures :

```bash
uv run dagster asset materialize -m src.orchestration.definitions --select "gold/attribution_countryyear,gold/attribution_bayes"
uv run python scripts/model_figures.py
```

Si les tables sont reconstruites, les `run_id` du haut de cette page changent.
