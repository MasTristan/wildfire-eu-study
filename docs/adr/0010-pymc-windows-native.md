# 0010. PyMC en Windows natif, backend Numba, sans WSL2

Statut : accepté le 2026-11-09. Remplace la partie WSL2 de
[0005](0005-windows-native-wsl-pymc.md) ; le reste de 0005 (pipeline sous
Windows natif, Dagster in-process) reste en vigueur.

## Contexte

L'ADR 0005 réservait WSL2 à PyMC, au motif qu'une chaîne C++ y est fournie par
conda-forge. WSL2 n'est pas installé sur la machine et l'installer est un
changement système réservé au propriétaire. La décision a donc été de tenter
PyMC en Windows natif, avec un critère de bascule fixé avant de toucher au vrai
modèle : le modèle Q3 réel doit échantillonner 4 chaînes x 1000 tirages (après
1000 de chauffe) en moins de 15 minutes, avec R-hat inférieur à 1,01 et zéro
divergence (ou moins de 1 % après reparamétrisation non centrée). Sinon, on
s'arrête et on garde WSL2.

## Mesures

Machine : Windows 11, Python 3.11.9, `uv`, pas de g++ ni de MSVC dans le PATH.
Versions épinglées : `pymc==5.28.5`, `arviz==0.23.4` (PyTensor 2.38.3,
Numba 0.65.1, NumPy 2.4.6). L'ajout a fait descendre `cachetools` de 7.2.2 à
6.2.6 dans le lock (contrainte de PyMC), sans autre changement de version d'un
paquet déjà présent.

| Mesure | Résultat |
|---|---|
| Installation (`uv add`) | 43 s, aucune compilation |
| Compilateur C trouvé par PyTensor | non (`cxx` vide) |
| Mode Python de repli, modèle jouet, 1 chaîne, 200 + 200 tirages | 22 s : environ 50 ms par itération sur trois paramètres, inutilisable pour le vrai modèle |
| Backend Numba, modèle jouet, 4 chaînes x 1000 + 1000 | 37 s d'échantillonnage, 48 s de bout en bout, R-hat 1,00 |
| Modèle Q3 réel (356 lignes, 40 pays, 4 métriques), 4 x 1000 + 1000, `target_accept` 0,99 | 64 à 73 s |
| Idem, trois graines (20260101, 7, 99) | R-hat max 1,0062, 1,0083, 1,0095 ; 0 divergence ; ESS bulk min 566, 517, 720 |
| Idem, asset Dagster `gold/attribution_bayes` (in-process), première version du panel | 68 s mesurés à l'époque, 82 s de bout en bout, R-hat max 1,0062, 0 divergence, ESS bulk min 566 ; le run MLflow de cette version consignait `duration_s` = 55,5 s (voir ci-dessous) |

Le critère est tenu avec un facteur 12 de marge sur la durée. Deux pièges
rencontrés en chemin, tous deux réglés dans `configure_backend`
(`src/models/attribution_bayes.py`) :

- Sur Python 3.11, `pip-system-certs` importe `pip` au démarrage, ce qui
  désactive le shim distutils de setuptools ; PyTensor importe ensuite
  setuptools en cherchant un compilateur et plante sur une assertion de
  `_distutils_hack`. `SETUPTOOLS_USE_DISTUTILS=stdlib` contourne le problème.
- Sans g++, PyTensor doit être prévenu de ne pas chercher de compilateur
  (`PYTENSOR_FLAGS=cxx=`) et le modèle est compilé avec `mode="NUMBA"`.

Pendant la mise au point du modèle réel, avec des intercepts non centrés, les
mêmes réglages donnaient R-hat 1,02 à 1,03 et 0 à 7 divergences selon la
graine : `mu_alpha` était mal identifié contre la moyenne des écarts pays. Les
intercepts sont donc centrés (chaque pays a une dizaine d'observations sur son
propre niveau) et les pentes non centrées.

## Décision

PyMC tourne en Windows natif avec le backend Numba. WSL2 n'est pas installé et
n'est plus nécessaire au projet. La version bayésienne de Q3 est livrée comme
asset Dagster `gold/attribution_bayes`, avec un asset check
`sampler_diagnostics` qui reprend la règle ci-dessus (R-hat sous 1,01,
divergences sous 1 % des tirages) et une expérience MLflow
`q3_attribution_bayes`.

## Conséquences

- Plus de dépendance à `.wslconfig` ni au disque virtuel WSL2 : les points
  de 0005 sur le plafond mémoire et le compactage ne s'appliquent plus.
- `pymc` et `arviz` rejoignent les dépendances runtime, donc `requirements.txt`
  (synchro vérifiée par la CI). Une vingtaine de paquets en plus, dont Numba et
  llvmlite.
- L'échantillonnage utilise le multiprocessing par spawn : tout script qui
  appelle `fit_attribution_bayes` hors Dagster doit avoir une garde
  `if __name__ == "__main__"`.
- Sur une machine avec g++ (le runner Linux de la CI), `configure_backend`
  garde le backend C par défaut de PyTensor. Seul le backend Numba a été mesuré
  ici ; le chemin C n'est pas mesuré.
- Le critère de bascule reste la règle d'acceptation du sampler : si une
  évolution du panel ou du modèle la fait échouer, l'asset check devient rouge
  avant que la table ne soit lue par le rapport.
- Les deux modèles reçoivent exactement les mêmes lignes, par une seule fonction
  (`prepare_panel`) : lignes complètes, puis pays à au moins 4 années complètes.
  Sur les données actuelles, 9 pays sont écartés : EG, IL, JO, LY, PS, SY (anomalie
  de pluie vide en zone aride) et GE, MD, RU (moins de 4 années). Les tables
  fréquentiste et bayésienne portent donc les mêmes 356 observations et 40 pays.
  Dans la première version, la table fréquentiste autonome gardait 360 lignes et
  43 pays pendant que le comparateur du modèle bayésien en avait 356 et 40 ; la
  formulation « écartés des deux côtés » ne valait que pour ce comparateur.
- Deux durées, deux mesures. `duration_s` (MLflow) est le temps de l'appel à
  `pm.sample`, de l'entrée au retour : démarrage des quatre processus et
  compilation Numba compris, résumé a posteriori exclu. La ligne que PyMC affiche
  (« took N seconds ») ne compte que la boucle d'échantillonnage. Sur la version
  courante (panel corrigé, asset Dagster en processus) : 50 s pour PyMC, 52,4 s
  pour `duration_s`, 54,1 s pour le pas Dagster entier (échantillonnage, résumé et
  écriture des tables) ; R-hat max 1,0052, 0 divergence, ESS bulk min 974.
  Le 68 s du tableau ci-dessus n'est pas `duration_s` (55,5 s pour le run qui
  l'accompagnait) : le dispositif qui l'a produit n'est pas tracé, donc il n'est
  pas reproductible ni comparable et il ne sert plus de référence. Les sept
  variantes de robustesse font de 45 à 59 s chacune (`duration_s`), 6 min 45 s
  pour l'asset `gold/attribution_robustness` entier ; R-hat max 1,0086.
