# T1 camera selection and reconstruction

The risk-weighted policy lowered mean absolute RGB error (MAE) by **0.00662** relative to uniform weighting, averaged equally over the four objects with paired results. Risk had lower MAE on set45 and set32, higher MAE on set103, and the same MAE on set61. The two remaining objects had no M4 result. [Full paired results](paired.csv)

## Independent target reconstruction

Each available entry combines six independent 256 × 256 center crops, with 393,216 comparable pixels per object and condition. The error counts below use mean absolute FP32 display-sRGB error across three channels strictly greater than 0.05. The full crop includes object and background. A verified foreground-only mask was unavailable. [Per-view counts and error sums](reconstruction-views.csv)

| Object | M2 MAE | Fixed M4 MAE | Uniform M4 MAE | Risk M4 MAE | Risk − uniform MAE | Error pixels, risk / uniform |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| set118 | 0.05051 | N/A | N/A | N/A | N/A | N/A |
| set76 | 0.15567 | N/A | N/A | N/A | N/A | N/A |
| set45 | 0.12102 | 0.06839 | 0.08477 | 0.07242 | −0.01234 | 118,326 / 125,697 |
| set103 | 0.28864 | 0.08220 | 0.07850 | 0.08643 | +0.00794 | 170,531 / 156,267 |
| set61 | 0.12463 | 0.07634 | 0.07263 | 0.07263 | 0.00000 | 116,464 / 116,464 |
| set32 | 0.14461 | 0.07670 | 0.10712 | 0.08506 | −0.02206 | 136,458 / 147,092 |

The three random camera-pair assignments are separate trajectories for each object. Their summary uses every registered seed. On set103, seed 137103 produced MAE 0.24383 after the registered initializer fallback. [Per-condition results](reconstruction-conditions.csv)

| Object | Seed 137101 MAE | Seed 137102 MAE | Seed 137103 MAE | Three-seed mean | Three-seed range |
| --- | ---: | ---: | ---: | ---: | --- |
| set118 | N/A | N/A | N/A | N/A | N/A |
| set76 | N/A | N/A | N/A | N/A | N/A |
| set45 | 0.09241 | 0.05989 | 0.07765 | 0.07665 | 0.05989–0.09241 |
| set103 | 0.08391 | 0.08877 | 0.24383 | 0.13884 | 0.08391–0.24383 |
| set61 | 0.08685 | 0.08391 | 0.09868 | 0.08981 | 0.08391–0.09868 |
| set32 | 0.09945 | 0.09988 | 0.07012 | 0.08982 | 0.07012–0.09988 |

The independent reader matched all 252 scheduled view records, 42 condition totals, the random means and ranges, and the paired object-equal endpoint. Of 16,515,072 scheduled pixels across logical trajectories, 11,796,480 were comparable in 180 views. Of these, 4,734,928 exceeded 0.05. The shared set61 fit contributes the same six view records to each of its two logical trajectories. The 72 missing views retain zero comparable pixels and an undefined MAE. [Independent recount](independent-nbv.json) · [Object-equal means and defined denominators](reconstruction-object-equal.csv)

## Camera choices before target access

The registered two-photo starting order was 24, 22. Each completed selection used the original source self-renders, four virtual query renders, eight candidate-view predictions and the frozen MS-L1 query model. All five strategies committed both camera IDs before their object received any candidate RGB. The fixed four-camera condition used 24, 23, 17, 22. Selected conditions inserted the lower selected ID first. [All 30 original strategy receipts](strategy-receipts.json) · [Actual observation by strategy](trajectory-observations.csv)

| Object | Query points after deduplication | Uniform | Risk | Random 137101 | Random 137102 | Random 137103 |
| --- | ---: | --- | --- | --- | --- | --- |
| set118 | N/A | N/A | N/A | N/A | N/A | N/A |
| set76 | N/A | N/A | N/A | N/A | N/A | N/A |
| set45 | 3,567 | 23, 31 | 10, 17 | 17, 20 | 17, 40 | 7, 17 |
| set103 | 3,984 | 10, 17 | 13, 17 | 10, 31 | 10, 23 | 17, 31 |
| set61 | 3,639 | 17, 31 | 17, 31 | 10, 13 | 7, 20 | 20, 31 |
| set32 | 3,490 | 23, 31 | 17, 31 | 7, 10 | 7, 13 | 17, 40 |

The frozen checkpoint probe required exactly 20,000 final Gaussians. The successful two-camera fits for set118 and set76 ended with 16,251 and 17,199. Their original probe runs failed, so neither object received candidate-photo access. All twelve dependent M4 conditions retain a failure record. Each object still contributed its M2 reconstruction at all six evaluation cameras. [Original probe and checkpoint identities](probe-lineage.csv) · [Per-object access](object-access.csv)

For each of the four completed selectors, uniform and risk each had one maximizing pair among 28 pairs at the registered tolerance. The frozen common fallback triggered zero times. On set61, both strategies chose 17, 31. Its separate observation records resolve to identical ordered inputs, initializer metadata and arrays, seed, recipe, and full numerical job. Both trajectories reference the same fit. [Pair utilities](pair-utilities.csv) · [Condition identities](conditions.csv)

All 29 physical scene fits completed 5,000 optimizer updates. Source-only SIFT triangulation qualified for 27 fits. The two exceptions were set103 M2 and set103 random 137103 M4, which each used the registered 2,048-point inverse-depth initialization. [Initializer and fit by condition](conditions.csv)
