# Spécification de l'outil exploratoire

Statut : proposition, à valider. Nature : analytique exploratoire, factuel, neutre.
Pas de thèse imposée. L'utilisateur mène sa propre enquête.

## Principe directeur

Fil conducteur d'interaction : **vue d'ensemble d'abord, puis zoom et filtre,
détails à la demande** (Shneiderman). Chaque question d'enquête est un axe que
l'utilisateur pivote, pas un chapitre qu'on lui impose. Deux états temporels
cohabitent : "que se passe-t-il" (saison en cours, live) et "que s'est-il passé"
(historique).

## Les axes d'enquête (mappés au modèle de données)

| Question | Ce qu'on montre | Source dans le modèle |
|---|---|---|
| Quoi / combien | surface brûlée, nombre de feux, intensité (FRP), émissions, surface par végétation | mesures des faits |
| Quand | année -> mois -> jour, longueur de saison | `Dim_Date` |
| Où | Europe -> pays -> commune (FR) -> point du feu, cartes | `Dim_Country` / `Dim_Commune` + géométries |
| Comment | distribution de taille des feux, classe/sévérité, végétation touchée | attributs événement + `Dim_LandCover` |
| Pourquoi | causes (France), météo-feu (FWI), sécheresse | `Dim_FireCause` + `Fact_FireWeather` |

Le "pourquoi" présente les corrélations comme des faits mis côte à côte, jamais
comme une causalité assénée. L'utilisateur juge.

## Architecture en niveaux (drill-down)

- **N0 Vue d'ensemble Europe** : carte + cartons de synthèse + courbe longue. Point d'entrée, tout filtrable.
- **N1 Pays** : séries temporelles, saisonnalité, comparaison inter-pays, mix végétation.
- **N2 Local** : province / commune (France), petits multiples, liste des feux.
- **N3 Feu** : fiche d'un événement (périmètre sur carte, date, surface, land cover, cause, % Natura 2000).

Détails à la demande : infobulles riches, drill-through de niveau en niveau.

## Facettes transverses (cross-filter)

Période, pays, taille de feu, type de végétation, cause, source. Tout filtre agit
sur toutes les vues. Le slicer de **source** (`Dim_Source`) empêche visuellement
de mélanger hotspots (comptage) et surface brûlée (hectares).

## Répartition des médias

- **Power BI** : cœur exploratoire faceté. Vue d'ensemble + drill-through
  pays -> commune -> feu, slicers, cross-filter, infobulles. Son point fort.
- **Streamlit** : exploration cartographique lourde (densité hotspots FIRMS,
  périmètres réels, curseur temporel) et explorateur de corrélation climat
  (l'utilisateur choisit la variable à croiser avec la surface brûlée).

## Ce que "award-winning" veut dire ici (sans thèse)

- **Interaction** : drill fluide, cross-filter instantané, faible latence.
- **Système visuel** neutre et impeccable : grille, typo, palette accessible, light et dark.
- **Rigueur** : faits honnêtes, provenance visible sur chaque vue, ruptures
  méthodologiques signalées (comptages EFFIS homogènes 2012+, périmètres 2016+).
- **Complétude** : les cinq axes réellement explorables, pas deux remplis et trois vides.

## Conséquence sur le build

Un outil exploratoire complet exige toutes les facettes : causes (BDIFF), FWI
(EWDS), land cover, géométries, hotspots. On construit donc le modèle complet,
rien à retirer du data model.
