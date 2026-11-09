# 0009. Périmètre de Q1 et de la phase 1 après le GO réduit du spike

Statut : accepté le 2026-11-08.

## Contexte

Le spike Q1 (notebooks/spike_q1/VERDICT.md) n'atteint pas le critère
pré-enregistré : en 2022, le gain relatif de Brier du modèle complet sur le
modèle aux composantes FWI vaut 7,9 % pour un seuil de 10 %, avec un
intervalle bootstrap [4,2 % ; 11,5 %] qui contient le seuil. Tout le reste
passe : 13,0 % en 2023, et +0,136 puis +0,182 de PR-AUC. La règle écrite
avant l'entraînement s'applique telle quelle : GO réduit.

Le résultat lui-même est net et en partie inattendu. À l'échelle maille x
jour, la classe de danger officielle et le FWI ne localisent presque pas les
ignitions : leur PR-AUC est à peine au-dessus du taux de base. Le gain vient
du lieu et du temps, c'est-à-dire des départs récents, de la climatologie
et des champs statiques. Le modèle 3b atteint 90 à 95 % du modèle complet.
La météo ERA5 et le SPEI ajoutent peu au-delà.

## Décision

1. **Q1 reste, à l'échelle du spike.** Ibérie et Grèce, 2019 à 2023, avec
   les sorties du verdict telles quelles. Le critère est présenté comme non
   atteint, avec son intervalle. Pas d'extension à l'Europe.
2. **Trois analyses exploratoires précèdent toute publication de Q1**
   (EXPLORATORY.md, écrit avant leur exécution) : contrôle par blocs
   spatiaux, ablation du bloc lieu et temps, et score sur les seules
   ignitions nouvelles. Elles servent à mesurer la part de persistance du
   même foyer dans le gain, le risque que le verdict lui-même désigne. Elles
   ne modifient pas le verdict.
3. **Abandonnés** : la prévision opérationnelle quotidienne (Q4), qui était
   conditionnée à un GO plein ; la matérialisation du feature store Europe
   2016 à 2026, qui ne sert plus que Q1 Europe ; Q5. Le code du feature
   store reste dans le dépôt, testé et documenté, sans être matérialisé.
4. **Inchangés et prioritaires** : la plateforme (Dagster, dbt, parité,
   bascule Power BI, DVC), Q2 (queue des tailles de feux) et Q3
   (attribution météo, fréquentiste puis bayésienne), qui ne dépendent pas
   de Q1.

## Conséquences

- La phase 1 perd sa dépendance la plus coûteuse, la file CDS : environ 130
  requêtes ERA5 mensuelles et 11 requêtes FWI Europe en moins. Le spike a
  mesuré que cette file est le goulot, pas le calcul.
- Le récit de la publication change. Il ne dit plus « un modèle bat
  l'indice européen ». Il dit : « à l'échelle maille x jour, l'indice de
  danger ne localise pas les départs ; la mémoire du lieu et de l'activité
  récente le fait, et la météo ajoute peu ». Ce récit ne vaut que si E1 à E3
  le confirment, et il se limite à l'Ibérie et à la Grèce pour 2019 à 2023,
  feux cartographiés par EFFIS.
- Un résultat négatif sur le critère, publié avec ses intervalles, sert la
  crédibilité du projet. C'est la raison d'être de la pré-inscription.

## Note du 2026-11-09 : analyses postérieures à la décision

La décision ci-dessus n'est pas réécrite. Deux précisions et un renvoi.

- Le point 2 parle de trois analyses exploratoires ; `EXPLORATORY.md` en compte
  maintenant huit. E4 (sensibilité au tirage des négatifs, dix graines) a été
  ajoutée le 2026-11-08 avant son exécution. E5 à E8 ont été déclarées le
  2026-11-09, avant leur exécution : recalibration d'intercept (E5), étiquette
  restreinte aux périmètres de 30 ha ou plus (E6), sans janvier à mars (E7),
  blocs spatiaux avec climatologie sans les blocs écartés (E8).
- Lecture de E4 à E8 (détail dans la section 12 de `VERDICT.md`) : le critère
  et le GO réduit ne changent pas. L'échec de 2022 résiste au tirage des
  négatifs (0 graine sur 10) et à la recalibration ; il ne tient pas au plancher
  de taille des périmètres. En revanche, sans janvier à mars, le gain de Brier
  de 2023 tombe de 13,0 % à 2,7 % : le succès de 2023 repose en partie sur les
  départs d'hiver. Le transfert à des cellules jamais vues (E1) résiste à E8.
- Le récit de la conséquence 2 doit se lire avec « persistance et lieu »
  plutôt que « départs » : l'étiquette mêle départs nouveaux et persistance du
  même foyer, et les périmètres EFFIS sont connus avec un délai, donc le spike
  ne dit pas ce qu'aurait su un prévisionniste le matin du jour J.
