# 0006. ERA5 journalier via l'archive Open-Meteo, pas le CDS

Statut : accepté le 2026-10-16.

## Contexte

Le spike Q1 a besoin de la météo journalière ERA5 par maille pour 2019 à
2023. Deux voies CDS ont été tentées : le dataset de statistiques
journalières dérivées (240 requêtes calculées à la demande, file qui ne
draine pas) puis le retrait horaire (une requête par année et par zone,
même file). Aucune n'a produit de données en une journée.

## Décision

L'archive historique Open-Meteo, modèle `era5` épinglé : même réanalyse,
même grille 0,25°, agrégats journaliers calculés côté serveur en UTC, une
requête par lot de centres de mailles, sans file ni clé. Licence CC BY 4.0,
usage non commercial, attribution dans docs/sources.md. Téléchargement
reprenable par lot, quotas respectés, sonde des variables servies.

## Alternatives écartées

- Persister sur le CDS : aucun contrôle sur la file.
- ARCO-ERA5 sur Google Cloud (Zarr) : découpage par pas de temps sur le
  globe entier, lecture d'une région trop coûteuse en bande passante.
- Se passer de la météo brute : le FWI en encode une partie, mais la
  question du spike porte sur ce que chaque bloc ajoute.

## Conséquences

- Le jeu de variables change (agrégats journaliers au lieu de champs
  horaires) ; amendement daté dans le protocole, avant tout entraînement.
- Open-Meteo devient candidat pour la prévision quotidienne de Q4, ce qui
  lèverait aussi la dépendance au CDS en opérationnel. À trancher en
  phase 3, dans un nouvel ADR.
