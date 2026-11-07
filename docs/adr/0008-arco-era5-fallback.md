# 0008. ARCO-ERA5 comme repli du CDS pour l'horaire ERA5

Statut : accepté le 2026-11-07, pour le spike Q1. Complète 0007 sans le
remplacer : le CDS mensuel reste la source par défaut.

## Contexte

Le 2026-11-07 la file CDS a cessé de livrer à 64 fichiers mensuels sur 120
(« Data Stores Service protection measures »), avec plus de cinquante
requêtes encore nécessaires pour 2021 à 2023, années de test du critère.

## Décision

Les mois que le CDS ne livre pas viennent du store public ARCO-ERA5 de
Google (`full_37-1h-0p25deg-chunk-1.zarr-v3`), lu en HTTPS anonyme par
`src/ingest/era5_arco.py` : même réanalyse horaire, même grille 0,25°,
même fonction de réduction journalière que la voie CDS. Le repli n'est
utilisé qu'après une vérification bloquante : un mois déjà obtenu par le
CDS, recalculé depuis ARCO, doit lui être identique à des tolérances fixées
avant la mesure (0,01 °C, 0,1 point d'humidité, 0,01 m/s, 0,01 mm).

## Faits mesurés sur le store

- Index temporel en heures depuis 1900-01-01, latitudes 90 à -90,
  longitudes 0 à 359,75 ; vérifiés à chaque exécution par `check_store`.
- Données définitives jusqu'au 2026-06-30 au 2026-11-06.
- Un bloc par heure sur le globe entier : 13,2 Mo par heure pour les cinq
  variables, environ 9,8 Go lus par mois pour n'en garder qu'une région.

## Conséquences

- Acceptable pour le spike (environ 25 mois, 250 Go de transfert, sans
  stockage brut). Inadapté au feature store Europe 2016 à 2026 (1,3 To par
  passage) : la production reste sur le CDS mensuel, en tâche de nuit.
- Le protocole porte un amendement daté, écrit avant tout entraînement.
