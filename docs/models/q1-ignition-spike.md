# Carte de modèle : Q1, occurrence des départs de feu (spike)

Run MLflow : **aucun**. Le spike Q1 (`notebooks/spike_q1/`) a été exécuté avant la mise
en place du suivi MLflow du projet, et les tables `q1_*` de `model_out` n'ont pas de
colonne `run_id`. La traçabilité repose sur les fichiers gelés et commités de
`notebooks/spike_q1/out/` (dernier commit `1d6122b`, 2026-11-08) et sur
`notebooks/spike_q1/VERDICT.md` (commit `c2dc3e6`). Les tables `q1_*` sont un
remaniement de ces fichiers, sans recalcul (asset `gold/q1_spike_results`).

Périmètre : Espagne, Portugal et Grèce, 2019 à 2023, feux cartographiés par EFFIS.
Pas d'extension à l'Europe ([ADR 0009](../adr/0009-q1-scope-after-reduced-go.md)).

## Question

Un modèle qui voit plus que les composantes de l'indice météo d'incendie (FWI)
prédit-il mieux les jours où un feu démarre que ces composantes seules ? L'unité est
une cellule de 0,25 degré (environ 25 km) un jour donné.

## Données

- 1 215 cellules (Espagne 847, Grèce 220, Portugal 148), 2019 à 2023, soit 2 218 590
  cellules-jours.
- **Étiquette (a), principale** : au moins un périmètre EFFIS dont la date de départ est ce
  jour et dont le centroïde est dans la cellule. 5 648 cellules-jours positives.
- **Étiquette (b), sensibilité** : au moins un nouveau groupe de détections VIIRS dans la
  cellule. 23 349 cellules-jours positives.
- Apprentissage 2019 à 2021, test 2022 et 2023, chaque année notée séparément. Test non
  échantillonné : 443 475 lignes par année, 1 709 positifs en 2022 et 1 393 en 2023
  (taux de base 0,00385 et 0,00314).
- Variables connues le jour J : sept composantes du FWI, météo ERA5 journalière, SPEI-3,
  champs statiques (couverture végétale, relief, fraction de terre), jour de l'année,
  nombre de départs dans la cellule les 7 et 30 jours précédents, climatologie des
  départs de la cellule (calculée sur les années d'apprentissage seulement).
- Substitutions déclarées : les champs statiques ERA5 remplacent CORINE (licence non
  acceptée), la population GHSL est omise, le déficit de pression de vapeur et l'évapotranspiration
  de référence ne sont pas dans le modèle (`et0`, `vpd_max`).

## Méthode

Cinq modèles, du plus simple au plus riche :

| N° | Modèle |
|---|---|
| 1 | Classe de danger officielle du FWI, chaque classe valant le taux de départs observé à l'apprentissage |
| 2 | Régression logistique sur log(1 + FWI) |
| 3 | LightGBM (arbres de décision boostés) sur les sept composantes FWI seules |
| 3b | Modèle 3 plus champs statiques, jour de l'année, nombre de départs récents, climatologie : le « lieu et la saison », sans météo ERA5 ni SPEI |
| 4 | LightGBM sur toutes les variables |

Négatifs échantillonnés 1 sur 20 à l'apprentissage et pondérés par 20, pour que les
probabilités restent calibrées sur le vrai taux de base. Hyperparamètres choisis sur
2021 (apprentissage sur 2019-2020), puis modèle réajusté sur 2019-2021.

**Critère de décision**, écrit avant le premier entraînement et jamais renégocié
(`PROTOCOL.md`), modèle 4 contre modèle 3, étiquette (a) : score de Brier plus bas d'au
moins 10 % en relatif **et** PR-AUC plus haute d'au moins 0,05 en absolu, en 2022 **et** en 2023.
Sinon : GO réduit.

Trois mesures, pour qui ne les pratique pas :

- **PR-AUC** (aire sous la courbe précision-rappel) : pour un événement rare, un
  modèle sans information vaut le taux de base (ici 0,003 à 0,004). Le **lift** est la PR-AUC
  divisée par le taux de base.
- **Score de Brier** : erreur quadratique moyenne des probabilités prédites. Plus bas est mieux.
- **Courbe de fiabilité** : on regroupe les lignes par probabilité prédite et on compare
  la probabilité moyenne prédite à la fréquence réellement observée. Sur la diagonale, le
  modèle est calibré.

Les intervalles du critère viennent d'un bootstrap par blocs d'une semaine (200 réplicats).

## Validation

- **Pré-enregistrement et revue** : protocole commité avant l'entraînement ; une revue
  indépendante avant le premier entraînement a trouvé et corrigé une fuite (climatologie
  construite avec l'année de validation) et ajouté le modèle 3b (`REVIEW.md`).
- **Tests** : `tests/test_spike_results.py` vérifie que les tables `q1_*` portent
  les chiffres du verdict.
- **Analyses exploratoires E1 à E4**, écrites avant leur exécution
  (`EXPLORATORY.md`), qui ne modifient pas le verdict. E1 est la validation par blocs
  spatiaux : les cellules sont groupées en blocs de 2 degrés, cinq plis, chaque pli
  est noté sur des blocs absents de l'apprentissage.
- **Fiabilité** : figure suivante.

![Courbes de fiabilité 2022 et 2023](../img/models/q1_reliability.png)

![Résultat des blocs spatiaux](../img/models/q1_spatial_blocks.png)

Règle de lecture fixée avant E1 : une PR-AUC par blocs à moins de 30 % (en relatif) de
la PR-AUC sans blocs veut dire que la compétence se transfère à des cellules jamais vues.

## Limites

**Ce résultat vaut pour l'Espagne, le Portugal et la Grèce en 2019 à 2023, sur les
feux cartographiés par EFFIS, et pas au-delà : le critère n'est pas atteint, et ce que
le modèle sait surtout, c'est que le feu revient là où il vient de brûler.**

- **Deux années de test.** 1 709 et 1 393 positifs. Le critère échoue sur une seule
  quantité, le gain de Brier de 2022.
- **Sensibilité à l'échantillon d'apprentissage.** Le bootstrap rééchantillonne les
  semaines de test, pas l'échantillon des négatifs. E4 refait l'apprentissage avec
  dix graines : le verdict reste celui de la graine 42, E4 donne la dispersion.
- **Persistance.** Les deux nombres de départs récents dans la cellule et la
  climatologie sont les trois variables les plus importantes. Sur les cellules sans
  départ depuis 30 jours (E3), la compétence recule fortement (voir Résultats).
- **E1 ne supprime pas la mémoire du lieu.** Les cellules écartées de l'apprentissage
  gardent leurs propres variables (départs récents, climatologie calculée sur leur
  historique 2019-2021). E1 montre que le modèle ajusté ne dépend pas des cellules
  vues à l'apprentissage, pas que l'historique local est dispensable. Les blocs sont
  contigus aux blocs d'apprentissage : l'autocorrélation spatiale aux frontières n'est
  pas supprimée.
- **Calibration et décalage de taux de base.** Les modèles 1 à 3 sous-estiment le nombre
  de départs d'un facteur 1,5 à 2,0 sur les deux années (rapport observé sur prédit, voir
  Résultats). Le taux de base de l'apprentissage est d'environ 0,0019 (2 546 positifs sur
  1 331 640 cellules-jours), la moitié de celui du test 2022 (0,0039) : le score de
  Brier du critère mesure donc aussi ce décalage de calibration, pas seulement la
  capacité à classer les jours.
- **Étiquette.** (a) manque les départs plus petits que la taille cartographiée par
  EFFIS ; (b) VIIRS est plus fréquente mais le résultat y est plus faible (gain de Brier
  de 5,6 % et 3,6 %).
- **Substitutions** de CORINE, de GHSL, absence de `et0` et `vpd_max`.
- **Pas de modèle d'ignition pour toute l'Europe** : abandon décidé en ADR 0009.

## Résultats

### Verdict : GO réduit

Étiquette (a), modèle 4 contre modèle 3. Intervalle à 95 % du bootstrap par blocs.

| Année | Gain de Brier (relatif) | Seuil | Gain de PR-AUC (absolu) | Seuil | Résultat |
|---|---|---|---|---|---|
| 2022 | 7,9 % [4,2 % ; 11,5 %] | 10 % : non atteint | +0,136 [0,096 ; 0,183] | 0,05 : atteint | ÉCHEC |
| 2023 | 13,0 % [6,8 % ; 17,3 %] | 10 % : atteint | +0,182 [0,114 ; 0,223] | 0,05 : atteint | SUCCÈS |

Le critère n'est pas atteint : **GO réduit**. L'échec tient à un seul nombre, le
gain de Brier de 2022, 2,1 points sous le seuil. Son intervalle [4,2 % ; 11,5 %]
contient 10 %, donc le spike ne peut pas dire si le vrai gain est sous ou au-dessus
du seuil. E4 apporte une autre dispersion : sur dix échantillons d'apprentissage, le gain 2022
va de 3,4 % à 7,9 % (médiane 5,7 %), aucune graine ne passe en 2022 (0 sur 10), sept sur dix passent en 2023,
aucune ne passe les deux années. L'échec du critère résiste donc au
tirage des négatifs ; en 2022, la graine du verdict donne un gain de Brier au niveau du
maximum des dix graines (7,9 %) et une PR-AUC du modèle 4 sous leur minimum
(0,1503 contre 0,1543).

### Scores par modèle, étiquette (a)

| Modèle | PR-AUC 2022 | Lift 2022 | PR-AUC 2023 | Lift 2023 |
|---|---|---|---|---|
| 1 Classe de danger | 0,0047 | 1,2 | 0,0031 | 1,0 |
| 2 Logistique FWI | 0,0051 | 1,3 | 0,0030 | 1,0 |
| 3 LightGBM FWI | 0,0143 | 3,7 | 0,0419 | 13,3 |
| 3b LightGBM FWI + lieu et saison | 0,1347 | 35,0 | 0,2126 | 67,7 |
| 4 LightGBM complet | 0,1503 | 39,0 | 0,2241 | 71,3 |

![PR-AUC par modèle et année](../img/models/q1_pr_auc.png)

Calibration : sur l'ensemble de l'année, le rapport entre départs observés et départs
prédits vaut 1,96, 1,89, 2,02 (modèles 1, 2, 3) en 2022 et 1,59, 1,52, 1,57 en 2023 ;
il vaut 1,10 (3b) et 1,13 (4) en 2022, 1,05 et 0,95 en 2023.

### Ce que disent E1 à E4

Récit de l'[ADR 0009](../adr/0009-q1-scope-after-reduced-go.md), repris phrase par
phrase et limité à ce que les analyses exploratoires confirment.

**1. « À l'échelle maille x jour, l'indice de danger ne localise presque pas les
départs. » Confirmé.** Lift de la classe de danger : 1,2 (2022) et 1,0 (2023), de la
logistique 1,3 et 1,0. Le modèle 3, plus souple, monte à 3,7 et 13,3 mais E1 montre qu'il ne transfère pas :
sur des blocs de cellules jamais vues sa PR-AUC tombe de 32 % (0,0097 contre 0,0143)
puis de 37 % (0,0263 contre 0,0419). Les modèles 3b et 4 restent à moins de 30 % de leur PR-AUC sans
blocs (modèle 4 : -4,3 % en 2022, +2,9 % en 2023 ; 3b : -8,1 % et -5,0 %,
un signe négatif voulant dire que la PR-AUC par blocs est plus haute).

**2. « La mémoire du lieu et de l'activité récente localise les départs. » Confirmé,
avec la persistance pour mécanisme principal.** Importance des variables du modèle complet
(part du gain) : départs dans la cellule les 7 jours précédents 17,0 %, les 30 jours précédents
16,5 %, climatologie de la cellule 15,3 % ; en groupes, lieu et saison 52,5 %, indice
de danger 18,7 %, météo et sécheresse 18,4 %, terrain 10,4 %. L'ablation E2, en
part de la PR-AUC du modèle 3b (2022, puis 2023) : sans les départs récents 72,1 % et
75,6 % ; avec le FWI et les départs récents seulement 90,5 % et 93,2 % ; sans aucune
variable FWI 70,7 % et 68,5 %. Aucun bloc ne porte seul le gain, les départs récents
avec le FWI s'en approchent le plus, et plus des deux tiers demeurent sans aucune
entrée FWI.
E3 précise le mécanisme : sur les cellules sans départ depuis 30 jours (38,2 % à 39,5 %
des positifs du test), le lift du modèle 4 est 11,3 et 22,4, contre 39,0 et 71,3 sur
tous les jours, et son avantage en PR-AUC sur le modèle 3 tombe à +0,014 et +0,021
(contre +0,136 et +0,182). Sur la sélection stricte, sans départ dans le voisinage 3 x 3
depuis 30 jours (16 % à 18 % des positifs), le lift du modèle 4 est 5,9 et 26,7.
Une grande part du gain est donc la persistance d'un même foyer, et le modèle reste
nettement au-dessus du modèle 3 en lift sur les départs nouveaux
(2,6 à 7,4 fois). Sur un départ nouveau, le lift du modèle 4 est environ 30 %
de son lift moyen (11,3 sur 39,0 en 2022, 22,4 sur 71,3 en 2023).

**3. « La météo ajoute peu. » Confirmé en moyenne, nuancé sur les départs nouveaux.**
Du modèle 3b au modèle 4, le gain de PR-AUC est de +0,016 (2022) et +0,012 (2023) et le
gain relatif de Brier de 1,6 % et 0,6 %. Sur les départs nouveaux, le lift passe de 9,8
à 11,3 (cellule, 2022), de 4,6 à 5,9 (voisinage, 2022), de 19,5 à 26,7 (voisinage, 2023),
mais reste à 22,6 contre 22,4 (cellule, 2023). La météo pèse un peu plus quand il n'y a
pas de départ récent, en valeur absolue ces gains sont petits : de -0,0003 à +0,005 de PR-AUC du modèle 3b
au modèle 4 sur ces sous-ensembles.

Le récit de la publication, limité à l'Ibérie et à la Grèce, 2019 à 2023, feux cartographiés
par EFFIS : l'indice de danger ne localise pas les départs à l'échelle maille x jour ;
la mémoire du lieu et de l'activité récente le fait, surtout par persistance d'un même
foyer ; la météo ajoute peu. Aucune phrase de cette carte ne va plus loin.

## Reproduire

Depuis la racine du dépôt. Les fichiers `notebooks/spike_q1/out/` sont commités ;
les tables `q1_*` de `data/lake/gold/model_out/` sont locales (asset
`gold/q1_spike_results`).

Critère, intervalles, scores et lift (tables `q1_criterion`, `q1_metrics`) :

```bash
uv run python -c "import pandas as pd; M='data/lake/gold/model_out/'; print(pd.read_parquet(M+'q1_criterion.parquet').round(4).to_string()); print(pd.read_parquet(M+'q1_metrics.parquet')[['year','model','base_rate','pr_auc','lift','brier']].round(5).to_string())"
```

Taux de base de l'apprentissage (positifs 2019 à 2021 de la concordance, sur les 1 331 640
cellules-jours d'apprentissage du verdict) :

```bash
uv run python -c "import pandas as pd; c=pd.read_csv('notebooks/spike_q1/out/concordance.csv'); n=c[c.year.astype(str).isin(['2019','2020','2021'])].a_positives.sum(); print(n, round(n/1331640,5))"
```

Gain de Brier du modèle 3b au modèle 4 et rapport observé sur prédit (calibration) :

```bash
uv run python -c "import pandas as pd; M='data/lake/gold/model_out/'; m=pd.read_parquet(M+'q1_metrics.parquet'); p=m.pivot(index='year',columns='model',values='brier'); print((1-p['lgbm_full']/p['lgbm_fwi_space']).round(4)); r=pd.read_parquet(M+'q1_reliability.parquet'); print(r.groupby(['year','model']).apply(lambda g: (g.obs_rate*g.n).sum()/(g.mean_pred*g.n).sum(), include_groups=False).round(2))"
```

Importance des variables :

```bash
uv run python -c "import pandas as pd; i=pd.read_parquet('data/lake/gold/model_out/q1_importance.parquet'); print(i.head(6)[['rank','feature','share']].round(3)); print(i.groupby('feature_group').share.sum().round(3))"
```

Blocs spatiaux (E1), ablation (E2), départs nouveaux (E3), graines (E4) :

```bash
uv run python -c "import pandas as pd; O='notebooks/spike_q1/out/'; print(pd.read_csv(O+'exploratory_spatial_blocks.csv')[['year','model','pr_auc','pr_auc_unblocked','relative_drop','transfers']].round(4).to_string()); print(pd.read_csv(O+'exploratory_ablation.csv')[['year','model','pr_auc','share_of_3b_pr_auc']].round(3).to_string()); n=pd.read_csv(O+'exploratory_new_ignitions.csv'); print(n[n.subset!='all'][['year','subset','model','pr_auc','lift','share_of_positives']].round(4).to_string()); print(pd.read_csv(O+'exploratory_seeds_summary.csv').round(4).to_string())"
```

Résultats de la graine du verdict contre le minimum des graines (PR-AUC du modèle 4 en 2022) :

```bash
uv run python -c "import pandas as pd; O='notebooks/spike_q1/out/'; m=pd.read_csv(O+'metrics_a.csv'); print(m[(m.year==2022)&(m.model=='lgbm_full')].pr_auc.round(4).tolist()); print(pd.read_csv(O+'exploratory_seeds_summary.csv')[['year','pr_auc_m4_min','brier_rel_max']].round(4))"
```

Rejouer le spike (nécessite les données téléchargées, voir
[la procédure du spike](../runbooks/spike-q1.md)) puis régénérer les tables et figures :

```bash
uv run python notebooks/spike_q1/s05_build_dataset.py
uv run python notebooks/spike_q1/s06_train_evaluate.py a b
uv run python notebooks/spike_q1/s07_concordance.py
uv run python notebooks/spike_q1/s08_exploratory.py
uv run dagster asset materialize -m src.orchestration.definitions --select "gold/q1_spike_results"
uv run python scripts/model_figures.py
```
