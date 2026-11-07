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
