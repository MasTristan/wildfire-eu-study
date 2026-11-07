# Spike Q1: verdict

Question: does a model that sees more than the fire-weather indices predict
ignition days better than the fire-weather components alone?

## 1. Verdict

**The pre-registered criterion is not met: reduced GO** (PROTOCOL.md, rule
"Otherwise", PLAN.md section 5 phase 0).

Criterion, label (a) EFFIS, model 4 (`lgbm_full`) against model 3 (`lgbm_fwi`):
Brier lower by at least 10 percent relative AND PR-AUC higher by at least 0.05
absolute, on 2022 AND on 2023.

| Test year | Brier gain (relative) | Threshold 10 % | PR-AUC gain (absolute) | Threshold 0.05 | Result |
|---|---|---|---|---|---|
| 2022 | 7.9 % | not met | +0.136 | met | FAIL |
| 2023 | 13.0 % | met | +0.182 | met | PASS |

The failure is one number, the 2022 Brier gain, 2.1 points under the bar. The
block-bootstrap interval of that gain, [4.2 %, 11.5 %], contains 10 %, so the
spike cannot say whether the true gain is under or over the bar. The criterion
was frozen before training and is applied as written.

Reading beyond the criterion (reported, not decisive):

- The skill is large in relative terms. PR-AUC of the FWI-only model is
  barely above the base rate (2022: 0.014 for a base rate of 0.0039). The full
  model reaches 0.150 (10.5 times) in 2022 and 0.224 (5.3 times) in 2023. The
  0.05 absolute bar was fixed before the base rate was known (REVIEW.md), and
  here it is the Brier bar that bites, not the PR-AUC bar.
- Most of the gain comes from knowing where and when fires happen, not from
  more weather. Model 3b (`lgbm_fwi_space`: FWI components plus static fields,
  day of year, past-ignition lags and climatology, no ERA5 weather, no SPEI)
  reaches PR-AUC 0.135 (2022) and 0.213 (2023), that is 90 % and 95 % of the
  full model. Adding ERA5 weather and SPEI on top of 3b gives +0.016 and +0.012
  PR-AUC and 1.7 % and 0.7 % relative Brier. Feature gain ranking confirms it:
  the two lag counts and the climatology are the top three features, then
  DC, FFMC and day of year.
- The same models on label (b) VIIRS give a weaker result (section 4).

## 2. Setup actually run

- Zone: Spain, Portugal, Greece, 1215 cells of 0.25 degree (ES 847, GR 220,
  PT 148), 2019 to 2023, 2 218 590 cell-days.
- Positives: label (a) 5 648 cell-days, label (b) 23 349.
- Train 2019 to 2021 (1 331 640 rows after sampling 1 negative in 20, weight
  20; 2 546 positives for (a), 15 086 for (b)). Test 2022 and 2023, unsampled,
  443 475 rows each (positives (a): 1 709 and 1 393).
- Hyper-parameters chosen on 2021 (training on 2019-2020, climatology from
  2019-2020 only), same grid for models 3, 3b and 4; refit on 2019-2021.
- Selected: model 3 leaves 15, min_data 50, 41 iterations; model 3b leaves 63,
  min_data 300, 66 iterations; model 4 leaves 63, min_data 300, 84 iterations
  (label (a)).

## 3. Metrics, label (a) EFFIS (primary)

| Year | Model | PR-AUC | Brier | Log-loss |
|---|---|---|---|---|
| 2022 | 1 EFFIS danger class | 0.0047 | 0.00384 | 0.02588 |
| 2022 | 2 logistic on log(1+FWI) | 0.0051 | 0.00384 | 0.02578 |
| 2022 | 3 LightGBM FWI components | 0.0143 | 0.00383 | 0.02429 |
| 2022 | 3b LightGBM FWI + place and time | 0.1347 | 0.00358 | 0.01833 |
| 2022 | 4 LightGBM all features | 0.1503 | 0.00352 | 0.01791 |
| 2023 | 1 EFFIS danger class | 0.0031 | 0.00313 | 0.02172 |
| 2023 | 2 logistic on log(1+FWI) | 0.0030 | 0.00313 | 0.02161 |
| 2023 | 3 LightGBM FWI components | 0.0419 | 0.00309 | 0.01853 |
| 2023 | 3b LightGBM FWI + place and time | 0.2126 | 0.00271 | 0.01366 |
| 2023 | 4 LightGBM all features | 0.2241 | 0.00269 | 0.01343 |

Base rate: 0.00385 (2022), 0.00314 (2023). Weekly block bootstrap (200
replicates), model 4 against model 3:

| Year | Quantity | 2.5 % | Median | 97.5 % |
|---|---|---|---|---|
| 2022 | Brier gain (relative) | 4.2 % | 7.7 % | 11.5 % |
| 2022 | PR-AUC gain (absolute) | 0.096 | 0.134 | 0.183 |
| 2023 | Brier gain (relative) | 6.8 % | 12.7 % | 17.3 % |
| 2023 | PR-AUC gain (absolute) | 0.114 | 0.180 | 0.223 |

Reliability diagrams: `out/reliability_a.png` (data in `out/reliability_a.csv`).

## 4. Metrics, label (b) VIIRS (sensitivity)

| Year | Model | PR-AUC | Brier | Log-loss |
|---|---|---|---|---|
| 2022 | 1 EFFIS danger class | 0.0123 | 0.00964 | 0.05440 |
| 2022 | 2 logistic on log(1+FWI) | 0.0123 | 0.00964 | 0.05443 |
| 2022 | 3 LightGBM FWI components | 0.0227 | 0.00960 | 0.05226 |
| 2022 | 3b LightGBM FWI + place and time | 0.1171 | 0.00911 | 0.04554 |
| 2022 | 4 LightGBM all features | 0.1284 | 0.00906 | 0.04502 |
| 2023 | 1 EFFIS danger class | 0.0108 | 0.00882 | 0.05108 |
| 2023 | 2 logistic on log(1+FWI) | 0.0096 | 0.00883 | 0.05117 |
| 2023 | 3 LightGBM FWI components | 0.0254 | 0.00876 | 0.04879 |
| 2023 | 3b LightGBM FWI + place and time | 0.0964 | 0.00846 | 0.04297 |
| 2023 | 4 LightGBM all features | 0.0998 | 0.00845 | 0.04266 |

Criterion applied to label (b): Brier gain 5.6 % (2022) and 3.6 % (2023),
PR-AUC gain +0.106 and +0.074. The Brier bar is missed in both years; same
direction as (a), weaker. Bootstrap intervals: Brier [3.4 %, 8.0 %] and
[2.3 %, 4.6 %]; PR-AUC [0.069, 0.141] and [0.049, 0.093] (`out/bootstrap_b.csv`).

## 5. Concordance of labels (a) and (b)

Same-day agreement is moderate: Jaccard 0.139, kappa 0.241 over 2019-2023
(2022: 0.187 and 0.311; 2023: 0.185 and 0.309). A VIIRS cluster is seen on the
same day for 62.6 % of EFFIS ignitions, and within the tolerant window for
85.7 %. Label (b) is far more frequent (23 349 against 5 648), which is the
detection of small fires that (a) does not map.

Day offset of the nearest VIIRS cluster start (3x3 cells, plus or minus five
days) relative to the EFFIS `firedate`: 75.2 % at 0, 6.1 % at +1, 4.4 % at -1,
6.5 % none. There is no mass at -1 or -2, so the worry in REVIEW.md (EFFIS
`firedate` lagging ignition and misaligning label (a) with day-J features) is
not supported by the data. Detail: `out/concordance.csv`,
`out/concordance_offsets.csv`.

## 6. Declared substitutions and limits

- Static ERA5 fields (high and low vegetation cover and type, land fraction,
  orography) stand in for CORINE, whose licence was not accepted on the
  account. GHSL population is omitted.
- ERA5 daily weather: vapour pressure deficit and reference
  evapotranspiration are not in the model (only Open-Meteo served them,
  amendment 2). The run reports "weather columns not available, excluded from
  the full model: et0, vpd_max".
- The yearly hold-out controls temporal leakage, not spatial autocorrelation.
- Label (a) misses ignitions below EFFIS mapping size. Label (b) is the
  sensitivity analysis for that.
- The tuning leak found in the review (climatology built with the validation
  year) was fixed before the first training.
- Side finding, outside the spike: `src/ingest/firms.py` passed the window end
  where the FIRMS area API expects the start. Fixed with a regression test.
  The v1 `fact_hotspot_countrymonth` carries the defect and must be rebuilt and
  re-frozen. The spike downloads VIIRS itself with the correct call.

## 7. Amendments to the protocol

1. 2026-10-16, Open-Meteo for daily ERA5 weather, abandoned within the day.
2. 2026-10-16, hourly CDS reanalysis reduced locally to daily values.
3. 2026-11-07, ARCO-ERA5 for the months the CDS did not deliver (section 8).

None changes the perimeter, the models or the decision criterion.

## 8. Data provenance

- FWI: EWDS `cems-fire-historical-v1`, 0.25 degree, 10 files (2 areas, 5 years).
- Labels: EFFIS perimeters from `fact_fireevent` (v1 gold), VIIRS S-NPP science
  quality (`VIIRS_SNPP_SP`), 732 five-day windows.
- SPEI-3: EDO, nearest 10 km cell, last completed dekad before day J.
- Daily ERA5 weather: 120 month-zone files, per amendment 3:
  - 67 from the CDS hourly reanalysis (`reanalysis-era5-single-levels`).
  - 53 from ARCO-ERA5 (`gs://gcp-public-data-arco-era5`,
    `full_37-1h-0p25deg-chunk-1.zarr-v3`), reduced by the same function
    (`src/ingest/era5_daily.hourly_to_daily`).
  - Identity checked on 2019-07 in both areas against the CDS files, maximum
    absolute differences for (t2m_c, t2m_max, rh, rh_min, wind, wind_mean,
    tp_mm): Iberia (0.000551, 0.001160, 0.003044, 0.010323, 0.000532,
    0.000186, 0.004195), Greece (0.000559, 0.001160, 0.003220, 0.009300,
    0.000540, 0.000185, 0.002805). All within tolerance (0.01 degree C, 0.1
    humidity point, 0.01 m/s, 0.01 mm).
  - ARCO months, Iberia (26): 2021-07 to 2022-12 and 2023-05 to 2023-12.
  - ARCO months, Greece (27): 2021-07 to 2022-12 and 2023-04 to 2023-12.
  - Every other month-zone file comes from the CDS: 2019-01 to 2021-06 in both
    areas, plus 2023-01 to 2023-04 for Iberia (34 files) and 2023-01 to
    2023-03 for Greece (33 files).
- Static fields: ERA5 single levels, one file per area.

## 9. Compute time

Machine: Windows 11 desktop, no GPU, LightGBM on CPU.

| Step | Time |
|---|---|
| Dataset build (FWI 3.0 s, weather 3.4 s, SPEI 2.8 s, labels 13.4 s, assemble 5.0 s) | about 28 s |
| Load, per label | 0.2 s |
| Features, per label | 0.5 s |
| Hyper-parameter search and final fits, (a) / (b) | 16.7 s / 19.0 s |
| Prediction, (a) / (b) | 3.2 s / 3.1 s |
| Block bootstrap, 200 replicates, (a) / (b) | 52.4 s / 57.9 s |
| Concordance | a few seconds |

Modelling is cheap: a full run from the built table takes about three minutes.
Acquisition dominated by far:

| Download | Time |
|---|---|
| CDS hourly ERA5, 70 recorded requests (retries included), median 263 s | 9.8 h cumulative, queue-bound, stalled at 67 of 120 files |
| Open-Meteo weather (amendment 1, 5 of 61 batches, abandoned) | 1.1 h |
| ARCO-ERA5 fill, `--workers 8`, 27 months | 2.66 h, 286 to 462 s per month |
| ARCO identity check, 2019-07, Iberia / Greece | 474 s / 389 s |
| VIIRS, 732 windows | 403 s |

Compute is not the constraint on the full plan; the CDS queue is. ARCO-ERA5
delivered a month in about five to eight minutes at no queue cost.

## 10. What to do with it

Reduced GO. The signal is real (PR-AUC five to ten times the FWI-only model),
but it is mostly spatial and seasonal memory (lags, climatology, static
fields). The full plan should not assume that the ERA5 weather and SPEI layer
carries the gain; its marginal contribution over model 3b is small.
Spatial block validation, which the spike did not do, is the first thing to
check before trusting model 3b.

Reproduction: `notebooks/spike_q1/s05_build_dataset.py`, then
`s06_train_evaluate.py a b`, then `s07_concordance.py`. Outputs in
`notebooks/spike_q1/out/`.

## 11. Exploratory analyses (post-verdict, not decisive)

Pre-declared in `EXPLORATORY.md` (2026-11-08, addendum before the second
run), run by `s08_exploratory.py` on label (a) with the hyper-parameters of
`out/criterion_a.json`. None of them changes the verdict of section 1. The
first run drew one negative sample per refitted model while `s06` draws one
shared sample; `s08` now draws one shared sample as `s06` does, and its refit
reproduces section 3 to the last digit (maximum absolute PR-AUC difference
0.0 over the four models and two years).

### E1. Spatial block control (2 degree blocks, 5 folds, pooled)

| Year | Model | PR-AUC blocked | PR-AUC unblocked | Relative drop | Transfers |
|---|---|---|---|---|---|
| 2022 | lgbm_full | 0.1568 | 0.1503 | -4.3 % | True |
| 2022 | lgbm_fwi | 0.0097 | 0.0143 | 32.4 % | False |
| 2022 | lgbm_fwi_space | 0.1456 | 0.1347 | -8.1 % | True |
| 2023 | lgbm_full | 0.2176 | 0.2241 | 2.9 % | True |
| 2023 | lgbm_fwi | 0.0263 | 0.0419 | 37.3 % | False |
| 2023 | lgbm_fwi_space | 0.2233 | 0.2126 | -5.0 % | True |

Reading: models 3b and 4 stay within 30 percent of their unblocked PR-AUC (model 4: -4.3 % and 2.9 %, 3b: -8.1 % and -5.0 %) so their skill transfers to unseen cells, while model 3 drops by 32.4 % and 37.3 % and does not.

### E2. Ablation of the place-and-time block

| Year | Model | Base rate | PR-AUC | Lift | Brier | Share of 3b PR-AUC |
|---|---|---|---|---|---|---|
| 2022 | lgbm_fwi | 0.00385 | 0.0143 | 3.7 | 0.00383 | 0.106 |
| 2022 | lgbm_fwi_space | 0.00385 | 0.1347 | 35.0 | 0.00358 | 1.0 |
| 2022 | 3c_no_lags | 0.00385 | 0.0971 | 25.2 | 0.00367 | 0.721 |
| 2022 | 3d_fwi_lags | 0.00385 | 0.1219 | 31.6 | 0.0036 | 0.905 |
| 2022 | 3e_no_fwi | 0.00385 | 0.0952 | 24.7 | 0.00363 | 0.707 |
| 2022 | lgbm_full | 0.00385 | 0.1503 | 39.0 | 0.00352 | 1.116 |
| 2023 | lgbm_fwi | 0.00314 | 0.0419 | 13.3 | 0.00309 | 0.197 |
| 2023 | lgbm_fwi_space | 0.00314 | 0.2126 | 67.7 | 0.00271 | 1.0 |
| 2023 | 3c_no_lags | 0.00314 | 0.1608 | 51.2 | 0.00286 | 0.756 |
| 2023 | 3d_fwi_lags | 0.00314 | 0.198 | 63.0 | 0.00274 | 0.932 |
| 2023 | 3e_no_fwi | 0.00314 | 0.1457 | 46.4 | 0.00285 | 0.685 |
| 2023 | lgbm_full | 0.00314 | 0.2241 | 71.3 | 0.00269 | 1.054 |

Reading: variants keep 72 % and 76 % of 3b without lags (3c), 91 % and 93 % with FWI and lags only (3d), and 71 % and 69 % without FWI (3e), so no single block carries the gain, the lags with FWI come closest, and more than two thirds remains with no fire-weather input.

### E3. New ignitions only

| Year | Subset | Model | Rows | Positives | Base rate | PR-AUC | Lift | Brier | Share of positives |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | all | danger_class | 443475 | 1709 | 0.00385 | 0.0047 | 1.2 | 0.00384 | 1.0 |
| 2022 | all | lgbm_fwi | 443475 | 1709 | 0.00385 | 0.0143 | 3.7 | 0.00383 | 1.0 |
| 2022 | all | lgbm_fwi_space | 443475 | 1709 | 0.00385 | 0.1347 | 35.0 | 0.00358 | 1.0 |
| 2022 | all | lgbm_full | 443475 | 1709 | 0.00385 | 0.1503 | 39.0 | 0.00352 | 1.0 |
| 2022 | cell_new_30d | danger_class | 415036 | 675 | 0.00163 | 0.003 | 1.8 | 0.00162 | 0.395 |
| 2022 | cell_new_30d | lgbm_fwi | 415036 | 675 | 0.00163 | 0.0041 | 2.5 | 0.00163 | 0.395 |
| 2022 | cell_new_30d | lgbm_fwi_space | 415036 | 675 | 0.00163 | 0.0159 | 9.8 | 0.00161 | 0.395 |
| 2022 | cell_new_30d | lgbm_full | 415036 | 675 | 0.00163 | 0.0184 | 11.3 | 0.00161 | 0.395 |
| 2022 | neighbourhood_new_30d | danger_class | 329676 | 272 | 0.00083 | 0.002 | 2.4 | 0.00083 | 0.159 |
| 2022 | neighbourhood_new_30d | lgbm_fwi | 329676 | 272 | 0.00083 | 0.0019 | 2.3 | 0.00083 | 0.159 |
| 2022 | neighbourhood_new_30d | lgbm_fwi_space | 329676 | 272 | 0.00083 | 0.0038 | 4.6 | 0.00082 | 0.159 |
| 2022 | neighbourhood_new_30d | lgbm_full | 329676 | 272 | 0.00083 | 0.0049 | 5.9 | 0.00082 | 0.159 |
| 2023 | all | danger_class | 443475 | 1393 | 0.00314 | 0.0031 | 1.0 | 0.00313 | 1.0 |
| 2023 | all | lgbm_fwi | 443475 | 1393 | 0.00314 | 0.0419 | 13.3 | 0.00309 | 1.0 |
| 2023 | all | lgbm_fwi_space | 443475 | 1393 | 0.00314 | 0.2126 | 67.7 | 0.00271 | 1.0 |
| 2023 | all | lgbm_full | 443475 | 1393 | 0.00314 | 0.2241 | 71.3 | 0.00269 | 1.0 |
| 2023 | cell_new_30d | danger_class | 423092 | 532 | 0.00126 | 0.0015 | 1.2 | 0.00126 | 0.382 |
| 2023 | cell_new_30d | lgbm_fwi | 423092 | 532 | 0.00126 | 0.0076 | 6.0 | 0.00125 | 0.382 |
| 2023 | cell_new_30d | lgbm_fwi_space | 423092 | 532 | 0.00126 | 0.0284 | 22.6 | 0.00124 | 0.382 |
| 2023 | cell_new_30d | lgbm_full | 423092 | 532 | 0.00126 | 0.0281 | 22.4 | 0.00124 | 0.382 |
| 2023 | neighbourhood_new_30d | danger_class | 358540 | 252 | 0.0007 | 0.001 | 1.4 | 0.0007 | 0.181 |
| 2023 | neighbourhood_new_30d | lgbm_fwi | 358540 | 252 | 0.0007 | 0.0026 | 3.6 | 0.00071 | 0.181 |
| 2023 | neighbourhood_new_30d | lgbm_fwi_space | 358540 | 252 | 0.0007 | 0.0137 | 19.5 | 0.0007 | 0.181 |
| 2023 | neighbourhood_new_30d | lgbm_full | 358540 | 252 | 0.0007 | 0.0188 | 26.7 | 0.0007 | 0.181 |

Reading: on cells with no ignition in the previous 30 days (38 % to 39 % of test positives) model 4 keeps +0.014 (2022) and +0.021 (2023) of PR-AUC over model 3, against +0.136 and +0.182 on all days, and on the stricter neighbourhood subset (16 % to 18 % of positives) +0.003 and +0.016, so it does not keep most of its absolute advantage; in lift, model 4 stays 2.6 to 7.4 times above model 3 on the new-ignition subsets (all days: 10.5 and 5.3 times).

### E4. Sensitivity to the training sample (10 seeds)

| Year | Brier gain min | Brier gain median | Brier gain max | Model 4 PR-AUC min | median | max | Seeds passing the year | Seeds passing both years |
|---|---|---|---|---|---|---|---|---|
| 2022 | 3.4 % | 5.7 % | 7.9 % | 0.1543 | 0.1628 | 0.1688 | 0 / 10 | 0 / 10 |
| 2023 | 7.3 % | 11.3 % | 13.7 % | 0.2258 | 0.2323 | 0.2491 | 7 / 10 | 0 / 10 |

Reading: no seed passes both years (2022: 0 of 10, 2023: 7 of 10), so the miss of the criterion is robust to the training sample; the 2022 Brier gain stays between 3.4 % and 7.9 % against the 10 % bar.
