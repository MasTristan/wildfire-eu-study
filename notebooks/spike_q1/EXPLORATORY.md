# Spike Q1: exploratory analyses after the verdict

Written on 2026-11-08, after VERDICT.md (reduced GO) and before any of the
analyses below was run. None of them can change the verdict: the decision
criterion was applied as pre-registered and stays applied. They exist
because the verdict's reading names a risk that has to be measured before
anything in it is published: the strongest features are the past-ignition
counts of the cell (`laga_7`, `laga_30`) and the climatology, so part of the
skill may come from persistence of the same fire activity rather than from
locating new ignitions.

All three reuse the verdict pipeline unchanged (`s06_train_evaluate.py`
functions, the hyper-parameters selected in `out/criterion_a.json`, the same
training years, sampling and test years), on label (a). Script:
`s08_exploratory.py`. Outputs: `out/exploratory_*.csv`.

## E1. Spatial block control

The yearly hold-out controls time, not space. Cells are grouped in 2 degree
blocks; five folds over blocks; for each fold, models 3, 3b and 4 are trained
on 2019 to 2021 without the fold's blocks and scored on the fold's blocks in
the test year. Predictions of the five folds are pooled and scored.

Reading rule, fixed now: a pooled PR-AUC within 30 percent (relative) of the
unblocked PR-AUC of the same model means the skill transfers to cells the
model has not seen. A larger drop means the model memorises cells, and the
report must say so.

## E2. Ablation of the place-and-time block

Three variants of model 3b, trained exactly like 3b:

- 3c: FWI components, static fields, day of year, climatology; no lags.
- 3d: FWI components and the two lag counts only.
- 3e: static fields, day of year, climatology and lags; no FWI.

Reading: the share of 3b's PR-AUC each variant keeps tells whether the gain
is "where" (3c), "what burnt recently" (3d), or independent of fire weather
altogether (3e).

## E3. New ignitions only

Test cell-days with no ignition in the cell over the previous 30 days
(`laga_30 == 0`), and the stricter set with no ignition in the 3 x 3
neighbourhood over the previous 30 days. Models 1, 3, 3b and 4, already
trained for the verdict (refitted identically here), are scored on those
subsets. Also reported: the share of test positives that fall in each
subset.

Reading: if model 4 keeps most of its PR-AUC advantage over model 3 on new
ignitions, the skill is not persistence of the same event.

## Addendum, 2026-11-08, written before the second run of s08

The first run of `s08_exploratory.py` drew a new negative sample for each
refitted model, while `s06_train_evaluate.py` draws one sample and trains
every model on it. Only the first model refitted got the same draw, which is
why model 3 matched section 3 of VERDICT.md exactly and models 3b and 4 did
not (2022, model 4: 0.169 against 0.150). The script now draws one shared
sample exactly as s06 does. On a synthetic table it reproduces s06's PR-AUC
for all four models to the last digit. E1 is unaffected: it trains one model
per fold by construction.

That gap shows that the choice of training sample alone moves model 4's
PR-AUC by about 12 percent relative. No interval in the verdict covers that
variance: the bootstrap resamples test weeks, not the training sample.
Hence a fourth analysis.

### E4. Sensitivity to the training sample

Ten seeds (0 to 9). For each, a fresh negative sample and a LightGBM seed,
the hyper-parameters of `out/criterion_a.json` unchanged, models 3, 3b and
4 refitted and scored on 2022 and 2023. Reported: the minimum, median and
maximum of the two criterion quantities and of model 4's PR-AUC, the share of
seeds passing the criterion in each year, and the share passing both years.

Reading rule, fixed now: the verdict stays the one obtained with seed 42 as
pre-registered, whatever E4 shows. E4 gives the spread to quote next to it:
if a sizeable share of seeds passes, the report says the result sits on the
bar; if none does, it says the miss is robust to the training sample.

### Reading of E3, complement

PR-AUC scales with the base rate, which falls by a factor of two to five on
the new-ignition subsets. The script now writes, for every subset, the base
rate and the lift (PR-AUC divided by base rate), the form comparable across
subsets. The pre-declared reading of E3, on the absolute PR-AUC advantage,
stays reported as written. The lift is reported next to it.

## Addendum, 2026-11-09, written before any of E5 to E8 was run

The adversarial review of the models (2026-11-09, findings Q1-1 to Q1-6, restated
in the questions below) names four things that E1 to E4 do not measure. They
become E5 to E8, declared here and committed before any execution. Script:
`s09_exploratory.py`, outputs `out/exploratory_*.csv`. As for E1 to E4, the
verdict pipeline is reused unchanged: same functions of `s06_train_evaluate.py`,
hyper-parameters read from `out/criterion_a.json` (no retuning), training
2019 to 2021, one negative sample shared by all models and drawn as `s06`
draws it (seed 42), test 2022 and 2023, label (a) unless stated. The criterion
quantities (relative Brier gain and absolute PR-AUC gain of model 4 against
model 3) are recomputed where a model pair is refitted and are reported as
information, never as a new decision.

### E5. Brier after intercept recalibration

Question. The training base rate (0.0019) is half the 2022 test rate (0.0039).
How much of the Brier gain of model 4 over model 3 is a difference in
calibration level rather than in ranking?

Calculation. Models 3 and 4 of the verdict, refitted identically. For each, a
single logit shift `b` is learned on 2021 (unsampled rows, predictions of the
final model): the root of `sum(sigmoid(logit(p) + b)) = sum(y)`, which is the
maximum-likelihood intercept with the logit as offset. Predictions on 2022 and
2023 become `sigmoid(logit(p) + b)`. Limit declared now: 2021 belongs to the
training years, so these predictions are in-sample and `b` is probably a lower
bound of the shift an out-of-sample year would need. Reported per model and
year: `b`, observed over predicted count on 2021 and on the test year, Brier
before and after, relative Brier gain of model 4 over model 3 before and
after, and whether the 10 percent bar is met. PR-AUC does not change under a
monotone map and is not recomputed.

Reading rule, fixed now. If the relative Brier gain of 2022 reaches 10 percent
or more after recalibration, the 2022 miss is largely a calibration-level
effect. If it stays under 10 percent, the miss is not explained by the base
rate shift. If the gain shrinks markedly for both years, the verdict's gain
partly measured the shift and the report says so.

This analysis does not modify the verdict.

### E6. Label restricted to perimeters of 30 ha or more

Question. The smallest EFFIS perimeters drop from about 20 ha (2016 to 2019) to
1 ha or less (2020 onward), so the label changes definition inside the
training years. Does the result hold on a label whose size floor is the same
over the whole period?

Calculation. Label (a) rebuilt from `fact_fireevent` (EFFIS, same cell mapping
as `s05_build_dataset.py`) keeping only events with `area_ha >= 30`, from 2016
on so that the lag counts use the same history. Lags (7 and 30 days),
climatology (from 2019 to 2021 of the restricted label) and labels all come
from the restricted events; everything else is unchanged. Models 3, 3b and 4
are refitted with the unchanged hyper-parameters. Reported: positives and base
rate per year, PR-AUC, lift, Brier, and the criterion quantities per test year.

Reading rule, fixed now. If model 4 keeps a lift of at least 5 against the base
rate in both years and keeps its PR-AUC advantage over model 3, the skill does
not come from the size-floor change. If the relative Brier gain reaches 10
percent in both years on this label, the report says the verdict's miss
depends on the label's definition; if not, that it does not.

This analysis does not modify the verdict.

### E7. Without January to March

Question. 35 percent of the label (a) positives fall in January to March, a
season that mixes agricultural and pastoral burning with wildfires, and the
EFFIS date of those events is the least certain. Does the result hold outside
that season?

Calculation. All rows of January, February and March removed from the training
table and from the test table, after the features are built (lags and
climatology unchanged, so a day in April keeps its history). Models 3, 3b and 4
refitted with the unchanged hyper-parameters. Reported: share of positives
removed in training and in each test year, base rate, PR-AUC, lift, Brier, and
the criterion quantities per test year.

Reading rule, fixed now. If the lift of model 4 and its PR-AUC advantage over
model 3 stay within 30 percent (relative) of their values on all months, the
skill does not come from the winter positives. A larger drop means the verdict
rests partly on them.

This analysis does not modify the verdict.

### E8. E1 with a climatology that never saw the held-out blocks

Question. In E1 the climatology of a held-out cell was computed from that
cell's own 2019 to 2021 labels, so the held-out blocks' labels were not fully
held out. Does the transfer to unseen cells survive when they are?

Calculation. Same folds, blocks and models as E1 (3, 3b, 4). In each fold, the
held-out cells receive no cell-specific climatology: their `clim` is the mean
`clim` of the training cells for the same day of year. Training cells keep
theirs (it never used held-out labels, each cell's climatology being its own).
The lag counts of held-out cells are kept, as in E1: they are past
observations available on day J. Pooled PR-AUC per model and year, compared
with the unblocked PR-AUC and with E1.

Reading rule, fixed now (E1's rule): a pooled PR-AUC within 30 percent
(relative) of the unblocked one means the skill transfers. If 3b and 4 cross
the 30 percent line here and not in E1, the E1 reading is withdrawn.

This analysis does not modify the verdict.

## Reading of E5 to E8 (run 2026-11-09, `s09_exploratory.py`)

Applied to the rules written above. None changes the verdict.

- **E5.** The logit shift learned on 2021 is +0.33 for model 3 and -0.19 for
  model 4. After recalibration the relative Brier gain of model 4 over model 3
  is 8.2 percent in 2022 (7.9 before) and 12.9 percent in 2023 (13.0 before).
  The 2022 miss is therefore not a calibration-level effect: the pair of
  models is shifted in nearly the same way. What recalibration does show is how
  large the shift is. The observed-over-predicted ratio of model 3 is 2.02 in
  2022 and stays at 1.46 after the shift (model 4: 1.13 before, 1.32 after),
  because 2021 has a base rate of 0.0026 against 0.0039 in 2022. The declared
  limit holds: the shift is learned in-sample and on a year that is itself
  lower than 2022.
- **E6.** On perimeters of 30 ha or more, the base rate is 0.0010 (training),
  0.0015 (2022) and 0.0010 (2023). Model 4 reaches PR-AUC 0.0653 (lift 44.3)
  in 2022 and 0.1525 (lift 154.4) in 2023 against 0.0048 and 0.0313 for model
  3, so it keeps a lift above 5 and its advantage over model 3. The relative
  Brier gain is 2.3 percent (2022) and 5.5 percent (2023), under 10 percent in
  both years. By the rule, the miss of the criterion does not depend on the
  label's size floor; the skill in lift does not either. On this label model
  3b is above model 4 in 2023 (PR-AUC 0.1555 against 0.1525).
- **E7.** Without January to March, 35.6 percent of the training positives and
  35.6 percent (2022) and 54.5 percent (2023) of the test positives are
  removed. Model 4 keeps a lift of 28.7 (2022) and 48.9 (2023) against 39.0 and
  71.3, a drop of 26 and 31 percent, but its PR-AUC advantage over model 3 falls
  to +0.087 and +0.087 against +0.136 and +0.182 (drops of 36 and 52 percent),
  and the relative Brier gain falls to 4.2 and 2.7 percent. By the rule (30
  percent), the verdict rests partly on the winter positives, in particular the
  2023 pass of the Brier bar, which does not survive their removal. The lift
  stays high, so the ranking skill outside the winter is real.
- **E8.** With held-out cells given the regional climatology instead of their
  own, the pooled PR-AUC of model 4 is 0.1311 (2022) and 0.1734 (2023), a drop
  of 12.8 and 22.6 percent against the unblocked values (E1: -4.3 and +2.9
  percent); model 3b drops 9.5 and 5.5 percent (E1: -8.1 and -5.0). Model 3
  is unchanged (it does not use the climatology) and still does not transfer.
  Models 3b and 4 stay within 30 percent, so the reading of E1 stands, with a
  smaller margin for model 4 in 2023. E1 overstated the transfer of model 4 by
  0.026 and 0.044 of PR-AUC.
