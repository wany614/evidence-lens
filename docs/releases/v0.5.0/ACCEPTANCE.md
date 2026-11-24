# Acceptance of the finite image grids

The two linear methods gave identical Windows and Mac arrays on all 42 old and new view cases per method. MS-frozen differed by at most 0.00001574 on their common finite domain. MS-ft differed by at most 0.00001437. One pixel in portrait v01 and one in square v03 crossed the MS-ft threshold. Validity and unknown-reason masks agreed for every method and view. [Full cross-host comparisons](../../verification/v0.5.0/README.md)

The distributed wheel was built from commit `c259c987b77296f368d78a9b09ef504226ac6cc3`. Its SHA-256 is `40d9b6ffb9d881c63d4710fee71112920dd665da831ef744fb062a20f86de734`. The four risk models keep their v0.3.0-rc.1 weights. Fifty-nine tests ran from the audited source archive in a fresh offline environment, including the rectangle encoder, asset importer, old file and resident entry points, display and evaluation denominators. [Build and data identities](BASELINE_MANIFEST.json) · [Frozen acceptance protocol](../../../protocol/v05-grid-acceptance.json)

| Width × height in analysis | Visual patch grid, width × height | Authored example |
|---|---|---|
| Historical 256 × 256 | 18 × 18 after the historical 252-square input transform | Six original views and a separate landscape letterbox control |
| 256 × 256 | 19 × 19 | square |
| 448 × 252 | 32 × 18 | landscape |
| 252 × 448 | 18 × 32 | portrait |
| 378 × 252 | 27 × 18 | photo |
| 375 × 251 | 27 × 18 | odd |

Five analytic scenes checked the source pixel centres, recovered 3D coordinates, disocclusion, expected depth, opacity and unknown reasons. The padded rows and columns of the odd size carry zero normalized encoder input. Each per-target score preserves the analysis dimensions. Display resizing keeps the categorical assignments discrete. Two genuine 175-tensor encoders were compared to the pinned official DINOv2 implementation. The first Mac check exposed an attention-backend difference in each historic-square encoder. Matched kernels reproduced the original tokens exactly; the default-backend measurement is retained separately. [Numerical encoder evidence](NEURAL_GRID.md)

The authored bell contains four measured source images per route. Windows rendered the same 4,096-Gaussian asset at the native analysis cameras. Mac installed both distributions in the same locked environments. The full installation produced 144 method/view scores across six routes and four methods. On the original M4 fixture all four NPY members for all 24 method/views were byte-identical across versions. The offline Linux base installation matched Mac across 42 original and new six-view method cases, comprising 3,670,686 pixels. Each tested installation, CPU dependency group, cross-host score difference and threshold crossing has a separate entry in the platform tables. [Platform and per-view results](../../verification/v0.5.0/README.md)

An exposed set118 recovery checkpoint contained 16,251 Gaussians under its registered cap of 20,000. Removing an opacity row was rejected. On the original CUDA renderer, all 196,608 RGB values, 65,536 alpha values and 65,536 expected camera-Z values for s00 equalled the stored reference render bit for bit. This is a new engineering regression. The old T1 ledger retains 48 missing method/layout results out of 288. [Complete asset and camera evidence](upstream/README.md)

The independent bell evaluator opened six original target photographs after the score matrix was sealed. On the aligned 256 × 144 frustum the native route had 35,416 errors among 221,184 labelled pixels, and the letterbox route had 34,989. The common finite intersection contained 217,547 pixels. At the 20% quota, the six per-view selections totalled 43,507 pixels. The relative error rate at that quota depended on the scoring method. The tables also give each route's native content area, every view, timing, RSS, unknown reasons and measured differences. [Comparison and authored example board](bell-comparison/README.md) · [Order of freezing, scoring and independent access](access-order.json)

Recorder failures include the first Blender worker's dependency import, the initial Mac neural comparison, a missing registered parent, an early Windows environment lookup and two workers whose snapshots omitted a required profile asset. The original records, package hashes, retried tasks and reclaimed measurements are linked in the run ledger. No risk parameters or threshold calibrators were fitted during this release. [Operations and resources](operations.json) · [Boundaries of earlier performance and scientific results](SCOPE_AND_LIMITATIONS.md)
