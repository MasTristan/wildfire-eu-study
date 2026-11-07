# 0004. MLflow en store fichier, DVC pour les données, aucun serveur

Statut : proposé le 2026-10-15.

## Contexte

Un seul poste, une seule personne, zéro budget cloud (règle du projet),
mais chaque figure de la publication doit remonter à un run et à une
version de données identifiables.

## Décision

- MLflow en store local sans serveur (`mlruns/`, gitignoré) : métadonnées
  dans un fichier SQLite, artefacts dans un dossier. Le store « fichiers
  plats » de MLflow est en maintenance depuis la version 3, SQLite est le
  store local recommandé ; `mlflow ui` le lit directement.
- Cartes de modèle en Markdown versionnées dans le dépôt, qui citent
  l'identifiant de run.
- DVC pour versionner gold, feature store et sorties de modèles, avec un
  remote S3 gratuit (Cloudflare R2 ou Backblaze B2, quota vérifié le jour
  de la décision). La bronze n'est pas versionnée : re-téléchargeable.
- Publication des données et des modèles sur Hugging Face, release finale
  figée avec DOI sur Zenodo.

## Alternatives écartées

- Serveur MLflow ou Docker Compose : aucun second lecteur, donc aucun
  bénéfice ; introduit en fin de parcours seulement s'il reste du temps.
- Weights and Biases : service tiers, et le projet ne dépend d'aucune
  licence ni compte payant.
