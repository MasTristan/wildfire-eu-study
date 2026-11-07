# 0003. Grille commune 0,25°, celle du FWI

Statut : proposé le 2026-10-15.

## Contexte

Le feature store maille x jour joint des sources à des résolutions
différentes : FWI et ERA5 à 0,25°, SPEI à 5 km en EPSG:3035, CORINE à
100 m, hotspots en points, périmètres en polygones.

## Décision

La maille de référence est la grille native du FWI (0,25°, centres sur le
réseau EWDS). Les sources plus fines sont agrégées vers elle (moyenne
pondérée par surface ou fraction de classe), les points et polygones lui
sont affectés par leur position ou leur centroïde. Aucune interpolation du
FWI vers une autre grille.

## Alternatives écartées

- H3 résolution 5 : élégant, mais impose de réinterpoler la variable
  centrale (FWI) et complique la lecture des cartes par un lecteur
  météo.
- 0,1° (ERA5-Land) : quadruple le volume sans source feu plus fine pour le
  justifier.

## Conséquences

- Identifiant de maille entier déterministe (`common.cell_id`), partagé par
  toutes les tables.
- Environ 20 000 mailles terrestres en Europe, 45 millions de lignes
  maille x jour sur 2016 à 2026 : tient dans DuckDB sur 32 Go.
