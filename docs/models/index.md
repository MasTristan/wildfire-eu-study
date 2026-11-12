# Cartes de modèle

Trois questions, trois modèles. Chaque carte suit le même plan : question, données,
méthode, validation, **limites**, résultats, reproduire. Les limites viennent avant
les résultats, volontairement : on lit ce que le modèle ne permet pas de dire avant de voir
ce qu'il dit.

Règle du projet : chaque modèle a sa carte, un run MLflow référencé, une figure de
validation et une phrase de limite écrite avant la figure de résultat. Chaque chiffre
cité dans une carte se retrouve avec une commande de sa section Reproduire.

| Question | Carte | Modèle | Run MLflow | Statut |
|---|---|---|---|---|
| Q1 : un modèle qui voit plus que l'indice météo prédit-il mieux les départs de feu ? | [Q1, spike d'occurrence](q1-ignition-spike.md) | LightGBM, cinq modèles en escalier | `q1_spike`, `73450271a109409e95ea4851cea5ad77` | critère non atteint, GO réduit, Ibérie et Grèce 2019 à 2023 |
| Q2 : quelle taille ont les feux rares ? | [Q2, queue des tailles de feux](q2-tail-risk.md) | Pareto généralisée au-dessus d'un seuil, par pays | `b47581300d4b4f2a828fffa5b76530e3` | 8 pays de l'UE publiés, niveaux de retour pour 6 (forme sous 1), loi rejetée par le test d'adéquation pour 3 |
| Q3 : quelle part de la surface brûlée la météo explique-t-elle ? | [Q3, attribution à la météo](q3-attribution.md) | régression à effets fixes et modèle bayésien hiérarchique | `31eb92ec7227401cb036848ecf438a60` (fréquentiste), `e1b00cc4742b4bd58a6ff0825c8f7e7b` (bayésien), `9a6e2948b6e147c3b1f2fd46d8f6bc23` (robustesse) | jours de danger et SPEI robustes ; pluie et température non distinguables de zéro |

Les cartes lisent les tables de `data/lake/gold/model_out/` (locales, non versionnées)
et les sorties gelées du spike (`notebooks/spike_q1/out/`). Les `run_id` cités sont ceux
des tables au moment de la rédaction ; reconstruire une table produit un nouveau run.

## Figures

Toutes les figures sont produites par un seul script qui ne recalcule rien : il lit les
tables et écrit dans `docs/img/models/`.

```bash
uv run python scripts/model_figures.py
```

## Gabarit

Le gabarit d'une carte est `MODEL_CARD_TEMPLATE` dans `src/models/tracking.py`, avec
la section Limits avant Results ; `write_model_card` l'écrit pour un modèle donné.
