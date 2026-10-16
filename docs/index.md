# Wildfire EU Study

Étude des feux de végétation en Europe, 2006 à 2026. La v1 est un rapport
Power BI de dix pages sur une couche de données Python et un modèle
sémantique écrit à la main. La v2, en construction, en fait une plateforme :
lake bronze, silver, gold orchestré par Dagster, gold en dbt sur DuckDB,
feature store maille x jour, modèles d'occurrence, de taille de feu et
d'attribution, chaque table publiée traçable à un run MLflow.

- [Plateforme](platform.md) : couches, assets, dbt, feature store, modèles.
- [Décisions d'architecture](adr/README.md) : une décision par fichier, avec
  les alternatives écartées.
- [Procédures](runbooks/materialise-v2.md) : reconstruire la v2 de bout en
  bout, avec les vérifications attendues à chaque étape.

Dépôt : [github.com/mastristan/wildfire-eu-study](https://github.com/mastristan/wildfire-eu-study).
