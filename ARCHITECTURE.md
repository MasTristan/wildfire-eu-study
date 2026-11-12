# Architecture

Comment les données traversent le projet, et pourquoi les frontières sont
placées là. Ce document donne les règles de chaque couche ;
[docs/platform.md](docs/platform.md) décrit les outils qui les appliquent
(Dagster, dbt, MLflow). Le détail du modèle sémantique est dans
[docs/data-model.md](docs/data-model.md), le catalogue des sources dans
[docs/sources.md](docs/sources.md).

## Vue d'ensemble

```
sources publiques
   |
   |  src/ingest/*.py          une API ou un fichier par module, aucune logique métier
   v
data/raw/                      brut tel que reçu, gitignoré, jamais réécrit
   |
   |  src/ingest/*.py (suite)   normalisation, agrégation spatiale
   v
data/interim/*.csv             tables intermédiaires, une par source
   |
   |  src/transform/*.py       dimensions conformes, faits au grain naturel
   v
data/processed/*.parquet       silver : une table par source, sous contrat pandera
   |
   |  dbt/models/gold/*.sql    passages et rollups en SQL, tests de grain
   v
data/lake/gold/star/           gold : le schéma en étoile, la seule chose que
   |                           les pages descriptives lisent
   |  src/models/*.py          trois modèles, un run MLflow par table
   v
data/lake/gold/model_out/      sorties de modèles, lues par trois pages séparées
   |
   |  DataFolder (paramètre M)
   v
powerbi/WildfireEU.SemanticModel     constellation de faits, TMDL
   |
   v
powerbi/WildfireEU.Report            rapport, 13 pages
```

Dagster porte ce graphe : chaque flèche est un asset ou un groupe d'assets,
et la gold porte les mêmes contrats que les tests.

## Les couches et leurs règles

### 1. Ingestion, `src/ingest/`

Un module par source. Chaque module fait une seule chose : aller chercher la
donnée et la ramener à un grain exploitable.

- **Aucun module n'écrit une colonne qu'il n'a pas mesurée.** Les trois
  ingestions climat (FWI, ERA5, SPEI) écrivent chacune sa table `interim` avec
  ses seules colonnes ; c'est `src/transform/fireweather.py` qui les assemble.
  Avant, un module écrivait des colonnes nulles « à remplir plus tard » et elles
  sont restées vides pendant des mois sans que rien ne le signale.
- **Reprenable.** Un fichier déjà téléchargé et non vide est sauté. Les
  requêtes lourdes sont découpées (une par année pour le FWI, une par décennie
  pour ERA5) : la file d'attente Copernicus expire sur une requête de onze ans.
- **`geo_ref.py` est le seul point d'entrée vers les polygones de pays.** Trois
  modules attribuent un pays par point-in-polygon ; centraliser garantit qu'ils
  attribuent les mêmes mailles aux mêmes pays, et que le correctif Natural Earth
  (`ISO_A2 = "-99"` pour la France, la Norvège et le Kosovo) s'applique partout.

### 2. Transformation, `src/transform/`

Construit le schéma en étoile. La règle structurante est celle du grain :

> Une table de faits par grain naturel. Jamais de fusion de sources de méthodes
> différentes dans un même enregistrement.

D'où la constellation plutôt qu'un star schema unique : les périmètres EFFIS
(grain feu), les agrégats EFFIS (grain pays-année), les hotspots FIRMS (grain
détection) et les grilles climat (grain maille-jour) ne sont pas réductibles à
un grain commun sans mentir.

Conséquence concrète : les émissions GWIS/GFAS vivent sur leurs **propres
lignes** de `fact_burntarea_countryyear`, avec la surface brûlée nulle. Poser
des tonnes GFAS sur une ligne EFFIS mélangerait deux mesures indépendantes du
même phénomène.

### 3. Gold, `dbt/`

La gold est écrite par dbt sur DuckDB, en Parquet. Neuf tables passent telles
quelles, trois rollups sont réécrits en SQL, une table est propre à la v2.

- **La migration ne peut pas dériver sans être vue.** `tests/test_gold_parity.py`
  compare chaque table gold à la référence gelée de la v1, valeur pour valeur.
  Les deux écarts volontaires (un correctif de fenêtre FIRMS, le drapeau UE de
  l'Allemagne) sont regelés table par table et notés dans le manifeste.
- **Un rollup ne mélange jamais deux sources** : c'est un test dbt
  (`no_source_mixing`), pas une convention.

### 4. Modèles, `src/models/`

La couche inférentielle est séparée de la couche descriptive, dans les deux
sens : aucun chiffre des dix pages descriptives ne sort d'un modèle, et les
sorties de modèles vivent dans leur propre dossier, lues par leurs propres
pages.

- **Chaque table publiée porte le `run_id` MLflow du run qui l'a écrite.**
- **Une règle de décision s'écrit avant le calcul** : critère du spike Q1,
  analyses exploratoires, règle `robust` de Q3.
- **Un résultat que le modèle ne soutient pas n'est pas publié** : pas de
  niveau de retour quand la forme de la queue atteint 1, pas d'effet annoncé
  quand l'intervalle contient zéro.

### 5. Modèle sémantique, `powerbi/WildfireEU.SemanticModel/`

TMDL versionné, diffable. 6 faits et 6 dimensions conformes pour la couche
descriptive, les tables de sorties de modèles à côté, une table de mesures,
un groupe de calcul temporel.

Toutes les relations sont **1 vers plusieurs, filtre simple**. Aucune
bidirectionnelle : les faits se croisent uniquement par les dimensions
partagées, ce qui rend le comportement de filtrage prévisible.

Le chemin des parquets est un paramètre unique, `DataFolder`, dans
`definition/expressions.tmdl` : après un clone, une seule ligne à modifier
plutôt qu'une partition par table. `tests/test_powerbi_model.py` vérifie que
chaque colonne déclarée a le type du Parquet qu'elle lit, et garde le modèle
contre les deux pièges qui font refuser l'ouverture à Desktop (identifiant de
lignage dupliqué, commentaire entre deux blocs).

## Ce que les tests vérifient

- `tests/test_data_layer.py` : schéma et intégrité référentielle des parquets,
  plus les invariants métier (les lignes GWIS ne portent jamais de surface
  brûlée, le SPEI n'est jamais imputé, la France est présente).
- Le rollup se compare à un agrégat EFFIS ingéré indépendamment, avec une
  tolérance de 2 %.
- `tests/test_contracts.py`, `tests/test_gold_parity.py`, `tests/test_dbt_gold.py` :
  contrats pandera, parité avec la v1, équivalence des rollups SQL et Python
  sur une silver synthétique.
- `tests/test_models_*.py` : chaque modèle retrouve des paramètres connus sur
  des données simulées.
- `tests/test_published_claims.py` : les chiffres cités en texte dans le
  rapport et les runs cités dans les cartes de modèle correspondent aux
  fichiers dont ils viennent.

Les tests qui ont besoin des données réelles se sautent sur un clone neuf ;
la CI exécute les autres, plus ruff, mypy et la construction stricte du site.

## Ce qui n'existe pas, et pourquoi

- **Pas de `Mart_CountryMonth`.** Le modèle cible en prévoyait un comme source
  unique pour Power BI et un futur Streamlit. Power BI consomme très bien la
  constellation, Streamlit n'existe pas : construire le mart aujourd'hui serait
  une table que personne ne lit. Décision consignée dans `docs/data-model.md`.
- **Pas de visuel IA** (arbre de décomposition, influenceurs clés). Ce sont des
  chercheurs de corrélation automatiques, exactement ce que ce projet refuse
  d'afficher sans horizon commun ni garde-fou sur le nombre d'observations.
- **Pas de `.pbix` binaire.** Format PBIP uniquement, pour que tout soit
  diffable.
