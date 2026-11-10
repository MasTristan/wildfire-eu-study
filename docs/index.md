# Wildfire EU Study

Les feux de végétation en Europe, de la donnée publique brute au rapport. Trois
choses dans ce dépôt :

- **une plateforme de données** : huit sources publiques ingérées, un lake
  bronze, silver, gold orchestré par Dagster, la gold écrite par dbt sur
  DuckDB, des contrats de données et un test de parité ;
- **un rapport Power BI de 13 pages** : dix pages descriptives, où chaque
  chiffre est une agrégation d'une source citée, et trois pages de modèles ;
- **trois modèles statistiques** : occurrence des départs de feu (un essai
  pré-enregistré dont le critère n'est pas atteint), taille des feux rares,
  part de la météo dans la surface brûlée. Chaque table publiée est traçable
  à un run MLflow.

Périodes couvertes : périmètres EFFIS de 2016 à juillet 2026, feux déclarés
en France (BDIFF) depuis 2006.

## Par où commencer

- [Cartes de modèle](models/index.md) : ce que chaque modèle permet de dire,
  avec les limites avant les résultats.
- [Plateforme](platform.md) : couches, assets Dagster, dbt, modèles, et ce
  qui n'est pas matérialisé.
- [Modèle de données](data-model.md) et [Sources](sources.md).
- [Décisions d'architecture](adr/README.md) : une décision par fichier, avec
  les alternatives écartées.
- [Procédures](runbooks/materialise-v2.md) : reconstruire le tout de bout en
  bout, avec la vérification attendue à chaque étape.

Dépôt : [github.com/mastristan/wildfire-eu-study](https://github.com/mastristan/wildfire-eu-study).
