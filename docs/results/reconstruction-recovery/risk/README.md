# Frozen risk on the same camera views

Four released CPU methods scored both generations of reconstruction. B-L1-transfer always reads s00. The three multisource methods use all photographs permitted by M2 or M4. Weights, preprocessing and thresholds follow v0.3.0-rc.1. Each entry compares the original D1 measurement with the selected recovery reconstruction.

| Object | Sources | Method | Common AP, old → new | Common prevalence, old → new | Low coverage L/V, old → new | Error in low LY/L, old → new | Unknown U/E, old → new |
|---|---|---|---:|---:|---:|---:|---:|
| set1 | M2 | B-L1-transfer | 0.8643 → 0.6613 | 86.81% → 54.57% | 53.85% → 19.68% | 91.96% → 47.49% | 0.01% → 1.05% |
| set1 | M2 | MS-L1 | 0.9140 → 0.8006 | 86.81% → 54.57% | 3.24% → 36.71% | 73.45% → 28.94% | 33.28% → 4.11% |
| set1 | M2 | MS-frozen | 0.9249 → 0.8141 | 86.81% → 54.57% | 7.12% → 30.50% | 67.14% → 25.49% | 33.28% → 4.11% |
| set1 | M2 | MS-ft | 0.9328 → 0.8186 | 86.81% → 54.57% | 10.10% → 38.73% | 67.03% → 27.30% | 33.28% → 4.11% |
| set1 | M4 | B-L1-transfer | 0.8547 → 0.4393 | 86.07% → 32.51% | 53.33% → 23.34% | 88.45% → 26.19% | 0.00% → 2.15% |
| set1 | M4 | MS-L1 | 0.9313 → 0.6128 | 86.07% → 32.51% | 3.54% → 51.34% | 52.61% → 15.63% | 17.10% → 6.30% |
| set1 | M4 | MS-frozen | 0.9285 → 0.6301 | 86.07% → 32.51% | 9.24% → 34.84% | 64.63% → 11.74% | 17.10% → 6.30% |
| set1 | M4 | MS-ft | 0.9309 → 0.6378 | 86.07% → 32.51% | 11.95% → 46.20% | 65.73% → 13.58% | 17.10% → 6.30% |
| set6 | M2 | B-L1-transfer | 0.9454 → 0.7171 | 94.67% → 63.27% | 51.81% → 19.21% | 95.02% → 45.51% | 0.49% → 0.38% |
| set6 | M2 | MS-L1 | 0.9548 → 0.7810 | 94.67% → 63.27% | 2.38% → 19.60% | 89.16% → 34.73% | 28.74% → 3.98% |
| set6 | M2 | MS-frozen | 0.9558 → 0.7787 | 94.67% → 63.27% | 1.79% → 12.07% | 86.17% → 26.25% | 28.74% → 3.98% |
| set6 | M2 | MS-ft | 0.9575 → 0.7812 | 94.67% → 63.27% | 1.74% → 17.22% | 84.97% → 31.17% | 28.74% → 3.98% |
| set6 | M4 | B-L1-transfer | 0.9014 → 0.5166 | 89.89% → 44.56% | 48.24% → 19.54% | 91.53% → 31.71% | 0.00% → 0.40% |
| set6 | M4 | MS-L1 | 0.9109 → 0.5890 | 89.89% → 44.56% | 0.60% → 30.15% | 82.34% → 27.77% | 20.02% → 5.54% |
| set6 | M4 | MS-frozen | 0.9101 → 0.5837 | 89.89% → 44.56% | 0.49% → 10.92% | 75.52% → 20.05% | 20.02% → 5.54% |
| set6 | M4 | MS-ft | 0.9124 → 0.5912 | 89.89% → 44.56% | 0.45% → 20.30% | 72.99% → 25.27% | 20.02% → 5.54% |

## Errors at matched coverage

The two percentages below are points from the registered 1–80% grid. Every method and both reconstructions use the same number of accepted pixels per attainable view on the eight-score intersection. Rates pool errors across the accepted pixels.

| Object | Sources | Requested coverage | Views attained | Accepted pixels | B-L1-transfer, old → new | MS-L1, old → new | MS-frozen, old → new | MS-ft, old → new |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| set1 | M2 | 5% | 6/6 | 19,662 | 97.27% → 59.51% | 77.14% → 45.68% | 75.31% → 29.11% | 73.15% → 20.59% |
| set1 | M2 | 20% | 6/6 | 78,642 | 91.55% → 47.26% | 78.67% → 27.89% | 77.61% → 26.92% | 76.54% → 26.12% |
| set1 | M4 | 5% | 6/6 | 19,662 | 93.82% → 33.96% | 65.79% → 22.21% | 74.45% → 12.90% | 74.14% → 4.98% |
| set1 | M4 | 20% | 6/6 | 78,642 | 91.94% → 30.25% | 72.90% → 10.53% | 72.73% → 10.60% | 71.49% → 7.93% |
| set6 | M2 | 5% | 6/6 | 19,662 | 96.98% → 60.99% | 91.40% → 46.69% | 91.19% → 50.76% | 90.70% → 49.94% |
| set6 | M2 | 20% | 6/6 | 78,642 | 94.92% → 60.38% | 93.09% → 52.91% | 93.40% → 54.77% | 92.96% → 54.12% |
| set6 | M4 | 5% | 6/6 | 19,662 | 92.19% → 47.56% | 86.94% → 32.88% | 85.96% → 37.16% | 85.57% → 34.89% |
| set6 | M4 | 20% | 6/6 | 78,642 | 90.28% → 44.97% | 87.41% → 34.24% | 87.53% → 37.68% | 87.29% → 35.83% |

AP ranks larger scores as greater RGB error and groups identical scores. For each reconstruction the common domain is E intersected with finite scores from all four methods. The table gives each defined view equal weight for AP and for its companion positive prevalence. The method-conditions table also gives own-domain AP and prevalence, defined-view counts and pooled numerators. A single-class or empty view has undefined AP.

E contains pixels having a usable reference and a comparable target prediction. Among E, low means finite score strictly below the frozen threshold, high includes equality, and unknown means no finite score. Counts satisfy E = L + H + U and Y = LY + HY + UY. The coverage denominator V includes every reference-valid pixel. Source support and alpha describe the reconstruction only.

The matched-coverage figure compares all four methods on the same pixels across old and new renders. Each view uses a fixed target fraction of V and a shared label-independent order for ties. Points pool the accepted and positive counts of views reaching the requested coverage. The per-view table gives missing points, support, cutoff tie bounds, and counts at the original MS-L1 operating coverage. These are descriptive comparisons of a development object and a previously exposed regression object.

[Matched-coverage figure](matched-coverage.svg) · [Frozen rates](method-conditions.csv) · [All views](method-views.csv) · [Matched points and tie bounds](matched-coverage.csv) · [Source support](source-support.csv) · [Unknown reasons](unknown-reasons.csv) · [Provenance](evidence.json)
