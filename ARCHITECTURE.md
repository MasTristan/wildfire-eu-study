# Architecture

Comment les données traversent le projet, et pourquoi les frontières sont
placées là. Le détail du modèle sémantique est dans
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
   |  src/transform/*.py       dimensions conformes, faits, rollups
   v
data/processed/*.parquet       le contrat, la seule chose que Power BI lit
   |
   |  DataFolder (paramètre M)
   v
powerbi/WildfireEU.SemanticModel     modèle en étoile, TMDL écrit à la main
   |
   v
powerbi/WildfireEU.Report            rapport, 10 pages
```

## Les trois couches et leurs règles

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

### 3. Modèle sémantique, `powerbi/WildfireEU.SemanticModel/`

TMDL écrit à la main, jamais généré par l'assistant de Desktop. 5 faits,
6 dimensions conformes, une table de mesures, un groupe de calcul temporel.

Toutes les relations sont **1 vers plusieurs, filtre simple**. Aucune
bidirectionnelle : les faits se croisent uniquement par les dimensions
partagées, ce qui rend le comportement de filtrage prévisible.

Le chemin des parquets est un paramètre unique, `DataFolder`, dans
`definition/expressions.tmdl` : après un clone, une seule ligne à modifier
plutôt que douze partitions à corriger.

## Ce que les tests vérifient

- `tests/test_data_layer.py` : schéma et intégrité référentielle des parquets,
  plus les invariants métier (les lignes GWIS ne portent jamais de surface
  brûlée, le SPEI n'est jamais imputé, la France est présente).
- Le rollup se compare à un agrégat EFFIS ingéré indépendamment, avec une
  tolérance de 2 %.

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
