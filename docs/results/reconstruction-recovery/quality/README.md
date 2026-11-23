# Reconstruction quality

The development rule selected **r2-scale-decay-02 at 5,000 updates** on the registered 5,000-update schedule. Selection used the mean of the six target RGB MAEs in each of set1 M2 and M4, giving both conditions equal weight. set6 was previously exposed in D1 and serves as a two-condition regression. The synthetic control uses its original floating-point reference RGB.

| Object | Sources | Target MAE, old → recovery | Errors/E, old → recovery | Zero-fit mean colour / closest source MAE | Views improved vs D1 | Source MAE, old → recovery |
|---|---|---:|---:|---:|---:|---:|
| scene_057_offset_bell_cage | M4 | 0.0175 → 0.0112 | 8.65% → 3.83% | 0.1283 / 0.0211 | 6/6 | 0.0092 → 0.0015 |
| set1 | M2 | 0.2603 → 0.1453 | 89.66% → 54.46% | 0.2500 / 0.1715 | 6/6 | 0.0746 → 0.0063 |
| set1 | M4 | 0.2037 → 0.0855 | 87.53% → 34.01% | 0.2570 / 0.1711 | 6/6 | 0.1020 → 0.0150 |
| set6 | M2 | 0.2521 → 0.1381 | 94.77% → 63.21% | 0.2265 / 0.1977 | 6/6 | 0.0917 → 0.0093 |
| set6 | M4 | 0.2136 → 0.0880 | 91.00% → 44.33% | 0.2271 / 0.1961 | 6/6 | 0.1181 → 0.0182 |

## Candidate recipes at matched update counts

Each column of errors uses the same six set1 target cameras. The last column gives M2 and M4 equal weight. Each recipe was fitted once per source count. These observations support a stopping decision, but one run per setting cannot establish why the recipes differ.

| Updates | Recipe | M2 target MAE | M4 target MAE | Paired mean |
|---:|---|---:|---:|---:|
| 500 | r2-geometry-01 | 0.2012 | 0.1462 | 0.1737 |
| 500 | r2-scale-decay-02 | 0.2021 | 0.1461 | 0.1741 |
| 1,500 | r2-geometry-01 | 0.2835 | 0.2696 | 0.2765 |
| 1,500 | r2-scale-decay-02 | 0.2929 | 0.2781 | 0.2855 |
| 3,000 | r2-geometry-01 | 0.1463 | 0.0951 | 0.1207 |
| 3,000 | r2-scale-decay-02 | 0.1435 | 0.0883 | 0.1159 |
| 5,000 | r2-geometry-01 | 0.1497 | 0.0963 | 0.1230 |
| 5,000 | r2-scale-decay-02 | 0.1453 | 0.0855 | 0.1154 |

The second recipe varies the learning rate for Gaussian log-scales. This parameter controls updates to the spatial widths of the reconstruction Gaussians.

Errors are pixels whose mean absolute display-RGB difference exceeds 0.05. The full 256 × 256 crop includes background and low-opacity pixels. Each rate uses observed RGB-comparable pixels E. The tables retain reference-valid V, missing observations and object-mask status. DTU supplies no independent 2D object mask for these views, so the image-wide figures cannot separate object from background.

The mean-colour comparator broadcasts the per-channel mean of the allowed source photographs. The closest-source comparator copies the full crop whose optical axis has the smallest angle to the target, breaking ties by source ID. Both use source images only.

[Visual readback](visual-readback.md) · [Development curves](quality-curves.svg) · [Source views](source-views.csv) · [Target views](target-views.csv) · [Paired pixels](paired-targets.csv) · [Candidate decisions](stopping-candidates.csv) · [Provenance](evidence.json)
