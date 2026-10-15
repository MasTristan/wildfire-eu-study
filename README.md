# wildfire-eu-study

Étude BI sur les feux de végétation en Europe, 2016 à 2026 : où l'Europe brûle,
quand dans l'année, quels feux comptent réellement, et ce que faisait la
météo-feu au même moment. Deux deep dives, par pays et sur la France.

Le rapport Power BI compte 10 pages, une question par page. La couche données
est en Python, le modèle sémantique est écrit à la main en TMDL : 5 tables de
faits (un grain chacune), 6 dimensions conformes, une table de mesures dédiée,
un groupe de calcul temporel, drill-through, segments synchronisés, paramètre
de champ.

## Ce que le projet refuse de faire

C'est la contrainte de conception, plus que les visuels :

- **Rien n'est modélisé ni estimé.** Chaque chiffre est une agrégation d'une
  source publique citée sur la page qui l'affiche.
- **Une donnée absente reste nulle.** Jamais de zéro de remplissage, jamais
  d'interpolation. `precip_anomaly` est nulle en climat aride parce qu'un écart
  en pourcentage à une normale quasi nulle est un artefact de division.
- **Deux sources qui divergent sont montrées toutes les deux.** BDIFF compte les
  feux déclarés en France, EFFIS cartographie des périmètres au-dessus d'un seuil
  de détection : les deux ne sont jamais additionnés ni comparés.
- **Une corrélation ne s'affiche jamais sans son horizon commun**, et elle est
  masquée sous dix points d'observation plutôt que calculée sur une poignée.
- **Une source déclarée mais jamais ingérée est retirée du modèle**, sinon elle
  apparaît dans les segments et ne filtre rien.

## Architecture

Vue détaillée dans [ARCHITECTURE.md](ARCHITECTURE.md).

```
src/ingest/       une ingestion par source : EFFIS (agrégats et périmètres),
                  BDIFF, FIRMS, GWIS/GFAS, CDS FWI, ERA5, EDO SPEI
src/ingest/geo_ref.py   attribution pays partagée (voir le piège plus bas)
src/transform/    dimensions conformes, fait au grain feu, rollups
data/             raw et interim (gitignorés), processed en parquet
powerbi/          projet PBIP : modèle sémantique TMDL et rapport
tests/            tests de contrat de la couche données
docs/             modèle de données, catalogue de sources
```

## Démarrage

Les versions sont verrouillées dans `uv.lock` ; `requirements.txt` en est un
export pour qui préfère pip.

```bash
uv sync          # ou : pip install -r requirements.txt
make check       # ruff, mypy, pytest
```

Les tests passent sans réseau et sans données : ceux qui ont besoin des parquets
se sautent proprement. `make validate` applique les contrats pandera de
`src/contracts.py` aux parquets produits.

Pour reconstruire la couche données de bout en bout, dans cet ordre :

```bash
python src/ingest/effis.py && python src/ingest/effis_perimeters.py && python src/ingest/bdiff.py && python src/ingest/firms.py && python src/ingest/gwis.py && python src/ingest/cds_fwi.py && python src/ingest/era5.py && python src/ingest/edo_spei.py
```

```bash
python src/transform/dimensions.py && python src/transform/fact_fireevent.py && python src/transform/rollups.py && python src/transform/fireweather.py
```

Les trois ingestions climat sont reprenables : un fichier déjà téléchargé et non
vide est sauté, `--agg-only` réagrège sans retélécharger.

Une fois `data/processed/*.parquet` produit, ouvrir `powerbi/WildfireEU.pbip`
dans Power BI Desktop et actualiser.

## Après un clone : un seul chemin à régler

Power Query n'a pas de notion fiable de « le dossier de ce projet ». Le chemin
des parquets est donc un paramètre, déclaré **une seule fois** dans
`powerbi/WildfireEU.SemanticModel/definition/expressions.tmdl` :

```
expression DataFolder = "C:\...\wildfire-eu-study\data\processed"
```

Modifiez cette ligne, ou passez par Accueil > Transformer les données > Gérer
les paramètres dans Desktop.

## Comptes

Deux comptes gratuits, aucune licence payante :

- **NASA FIRMS** : MAP_KEY gratuite, dans `.env`.
- **ECMWF** : un seul Personal Access Token authentifie le CDS
  (`cds.climate.copernicus.eu`) **et** l'EWDS (`ewds.climate.copernicus.eu`).
  Seule l'URL de base change, `src/config.py` expose `cds_client("cds"|"ewds")`.
  Les licences des datasets s'acceptent une fois en ligne.

EFFIS, GWIS, EDO, BDIFF : ouverts, sans clé.

## Un piège qui a coûté un pays

Natural Earth admin-0 code la France, la Norvège et le Kosovo en
`ISO_A2 = "-99"`. Les trois modules qui attribuent un pays par point-in-polygon
filtraient sur ce champ : **la France était absente de toute la couche climat et
de la table de hotspots**, sur le pays qui a sa propre page de deep-dive. Les
totaux restaient plausibles, rien ne le signalait à l'écran.

Corrigé dans `src/ingest/geo_ref.py`, seul point d'entrée autorisé vers ces
polygones (repli sur `ISO_A2_EH`), et verrouillé par
`test_fireweather_covers_france`.

## Limites connues

- EFFIS cartographie au-dessus d'un seuil de détection : **tous les totaux sont
  des planchers**, les petits feux manquent.
- Le FWI et le SPEI sont des moyennes sur tout le territoire d'un pays, pas
  seulement sur la forêt, ce qui tasse l'indice des grands pays.
- Les émissions GFAS voient aussi les feux agricoles et sous le seuil EFFIS :
  elles ne sont **pas divisibles** par la surface brûlée EFFIS.
- La réanalyse FWI est consolidée avec plusieurs mois de retard, donc l'année en
  cours est toujours partielle. Les visuels annuels la neutralisent par la mesure
  `Complete Climate Year`.
- La Russie est exclue de la table d'émissions : 12 % seulement de son territoire
  est dans le domaine d'étude et son total national est sibérien.

## Attribution

Copernicus Emergency Management Service (EFFIS), Copernicus Climate Change
Service (CDS et EWDS, ERA5), Copernicus European Drought Observatory,
Global Wildfire Information System, NASA FIRMS, Commission européenne (JRC),
bdiff.agriculture.gouv.fr, geo.api.gouv.fr, Natural Earth, World Bank Open Data.
Réutilisation soumise aux licences respectives.

Outil analytique exploratoire : les corrélations montrées sont descriptives et
n'établissent pas de causalité.

## Auteur et contact

Tristan Mas. Questions, remarques ou corrections : ouvrez une issue sur ce
dépôt.

Licence du code : MIT, voir [LICENSE](LICENSE). Les jeux de données ingérés ne
sont pas couverts et restent sous leurs licences respectives.
