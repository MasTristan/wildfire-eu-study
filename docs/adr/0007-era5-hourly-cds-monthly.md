# 0007. Météo journalière ERA5 depuis l'horaire CDS, une requête par mois

Statut : accepté le 2026-10-16. Remplace 0006 pour tout usage à l'échelle
du maillage ; 0006 reste valable pour des sondages à petite échelle.

## Contexte

Le quota gratuit d'Open-Meteo est atteint après environ cent années-maille
par jour : douze jours pour les 1 220 mailles du spike, impossible pour les
20 000 mailles du feature store. Le dataset CDS de statistiques journalières
dérivées met des heures par requête. Le dataset horaire brut
(`reanalysis-era5-single-levels`) répond à une requête mensuelle sur une
zone en quelques minutes, et une telle requête reste loin du plafond de
champs par requête.

## Décision

Une requête CDS par mois et par zone, cinq variables horaires (t2m, d2m,
u10, v10, tp), réduction locale en valeurs journalières UTC par
`src/ingest/era5_daily.py` : moyenne et max de température, moyenne et min
d'humidité relative (Magnus-Tetens), max et moyenne du vent, cumul de
précipitations. Le fichier horaire est supprimé après réduction. Le spike
et le feature store utilisent la même fonction de réduction.

## Conséquences

- Europe 2016 à 2026 : environ 130 requêtes, une nuit de file CDS, moins
  de 10 Go de journalier conservé.
- Open-Meteo garde deux usages : vérification ponctuelle d'une maille, et
  prévision à quelques jours sur un petit nombre de points.
- Le déficit de pression de vapeur et l'évapotranspiration de référence,
  servis par Open-Meteo seulement, ne sont plus dans le feature store.
