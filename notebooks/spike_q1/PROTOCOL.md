# Spike Q1: protocol (written before the first model is trained)

Question: does a model that sees more than the fire-weather indices predict
ignition days better than the fire-weather components alone?

## Frozen scope

- Zone: Spain, Portugal, Greece (0.25 degree cells whose centre is inside the
  country polygon and where the FWI is defined). Years 2019 to 2023.
- Unit: one 0.25 degree cell on one UTC day.
- Train 2019 to 2021. Test 2022 and 2023, each scored separately.
- Hyper-parameters are chosen on 2021 (train on 2019-2020), then the model is
  refitted on 2019-2021. 2022 and 2023 are never used for any choice.

## Labels

- (a) EFFIS: at least one EFFIS perimeter whose `firedate` is the day and
  whose centroid falls in the cell. Primary label.
- (b) VIIRS: at least one new hotspot cluster (S-NPP, science quality, type 0,
  confidence not low) in the cell. Clusters are DBSCAN, eps 2 km, on the day;
  a cluster is new if no detection of day J-1 lies within 2 km of it.
- Concordance of (a) and (b) is reported. (b) is a sensitivity analysis.

## Features, all known at day J

FWI, FFMC, DMC, DC, ISI, BUI, DSR (cems-fire-historical-v1, 0.25 degree);
ERA5 t2m, d2m, derived relative humidity, wind speed, u10, v10, tp, tp summed
over the 7 previous days; SPEI-3 of the last completed dekad; static ERA5
vegetation cover and type, orography and land fraction (CORINE substitute, see
VERDICT.md); day of year; ignition count in the cell over the 7 and 30 previous
days (past only); cell climatology of ignitions per calendar day (+/- 15 days),
computed on training years only and, for training rows, leaving out the row's own year.

## Sampling and weights

Training negatives are sampled 1 in 20 (seed 42), positives all kept, negatives
weighted 20 so that probabilities are calibrated to the true base rate. Test
sets are never sampled.

## Models

1. EFFIS danger class from the FWI (official thresholds 5.2, 11.2, 21.3, 38, 50),
   each class mapped to its empirical training ignition rate.
2. Logistic regression on log(1+FWI).
3. LightGBM on the seven FWI-system components only.
4. LightGBM on all features.

Models 3 and 4 share the same hyper-parameter search space and budget.

## Decision criterion (not renegotiable after the first training)

Computed on label (a), against model 3:
Brier score of model 4 lower by at least 10 percent (relative) AND PR-AUC higher
by at least 0.05 (absolute), on 2022 AND on 2023.

- Both hold: GO, full plan.
- Otherwise: reduced GO, as written in PLAN.md section 5, phase 0.

Reported but not decisive: the same criterion on label (b), block-bootstrap
(weekly blocks) intervals for the differences, reliability diagrams, timings.
