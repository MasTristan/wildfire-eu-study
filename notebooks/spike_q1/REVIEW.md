# Spike Q1: independent review before the first training (2026-10-15)

Reviewed: PROTOCOL.md, common.py, s05_build_dataset.py, s06_train_evaluate.py,
s07_concordance.py, against the protocol and for information leakage. The
review happened after the protocol was committed and before any model was
trained; the fixes below were applied in the same commit as this note, and
s06 and s07 were run end to end on a synthetic daily table to check they
execute.

## Fixed

1. **Leak into hyper-parameter tuning through the climatology feature.**
   `add_climatology` built the feature from 2019-2021 for every row, leaving
   out only the row's own year. During tuning the model is fitted on
   2019-2020 and validated on 2021, so a 2019 row carried a feature computed
   with 2021 labels: the validation year entered a training feature. The
   test years were never affected (their climatology uses training years
   only), but the choice of `num_leaves`, `min_data_in_leaf` and the number
   of iterations was. Fix: the climatology now takes the list of years it may
   use; tuning uses 2019-2020, the final fit and the test 2019-2021.
2. **Missing step in the skill staircase.** With only "FWI components" and
   "everything", a Brier gain could come from learning where fires happen
   (static fields, calendar, past ignitions) as much as from more weather.
   Added model 3b `lgbm_fwi_space`: FWI components plus static fields, day of
   year, lag counts and climatology, without ERA5 weather or SPEI. It is
   reported, not decisive: the criterion stays 4 versus 3 as pre-registered.
3. **Day offset between the two labels** added to s07: for each EFFIS start,
   the signed offset of the nearest VIIRS cluster start in the 3x3
   neighbourhood within five days. A mass at -1 or -2 would mean EFFIS
   `firedate` lags ignition, which misaligns label (a) with day-J features
   for every model alike and bounds the skill any model can show.

## Checked and accepted

- Lag counts: rolling sums shifted by one day, day J excluded, EFFIS history
  from 2016. Rain over 7 days: same construction. SPEI: last dekad whose end
  is strictly before J. No feature uses information after J.
- Negatives sampled 1 in 20 with weight 20 in training only; validation and
  test years are complete and unweighted, so early stopping and all metrics
  are computed on the true base rate.
- Cell mesh: FWI lattice restricted by the shared `geo_ref.grid_mask`, the
  same point-in-polygon the v1 climate layer uses.
- Model 1 maps the official EFFIS danger classes to their empirical training
  ignition rate; model 2 is a logistic regression on log(1 + FWI). Both are
  honest floors.
- Bootstrap: weekly temporal blocks, 200 replicates, intervals on the two
  criterion quantities.

## Known limits, to be stated in VERDICT.md

- The PR-AUC gain of 0.05 was fixed before the base rate was known. At a base
  rate well under 1 percent, PR-AUC values are small and 0.05 absolute is a
  demanding bar. The criterion is pre-registered and stays; VERDICT.md must
  report the relative gain next to it so the reader can judge.
- Label (a) depends on EFFIS mapping above its detection floor: small
  ignitions are not positives. Label (b) is the sensitivity analysis for that.
- Static ERA5 vegetation fields stand in for CORINE (licence not accepted on
  the account); GHSL population is omitted. Both are declared substitutions,
  not silent ones.
- The yearly hold-out controls temporal leakage, not spatial autocorrelation.
  Spatial block validation belongs to Q1 proper, not to the spike.

## Side finding, outside the spike

`src/ingest/firms.py` passed the window end where the FIRMS area API expects
the window start. Fixed in `window_url`, with a regression test. The v1
`fact_hotspot_countrymonth` carries the defect and must be rebuilt and
re-frozen as the parity reference for that one table, with a note in the
manifest; the spike does not use that table (it downloads VIIRS itself with
the correct call).
