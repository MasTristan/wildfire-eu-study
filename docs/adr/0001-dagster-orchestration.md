# 0001. Dagster comme orchestrateur, un asset par table de lake

Statut : proposé le 2026-10-15, à confirmer au verdict du spike Q1.

## Contexte

La v1 s'exécute par douze commandes manuelles dans un ordre documenté dans
le README. La v2 ajoute un feature store, des modèles, une prévision
quotidienne : le graphe de dépendances dépasse ce qu'un README peut tenir,
et la fraîcheur de chaque table doit être visible.

## Décision

Dagster, en modèle « assets » : une définition par table du lake, clés
préfixées par couche (`bronze/`, `silver/`, `gold/`), partitions annuelles
statiques sur les sources qui s'y prêtent, asset checks portant les
contrats pandera et les tests dbt. Exécuteur in-process en développement.

## Alternatives écartées

- Airflow : orienté tâches, lignage de données absent sans outillage
  supplémentaire, plus lourd à faire tourner sur un poste unique.
- Prefect : flux et tâches, pas de notion native d'asset ni de partition
  de données ; le lignage visuel, qui est ici un livrable, manque.
- Makefile seul : suffisant pour la v1, aveugle sur la fraîcheur et les
  partitions.

## Conséquences

- Les modules d'ingestion v1 restent le code qui fait le travail ; les
  assets les enveloppent. La réécriture en assets partition-safe se fait
  source par source en phase 1 sans changer les clés.
- Le graphe d'assets devient un visuel de la publication.
- Coût : un outil de plus à expliquer dans la documentation.
