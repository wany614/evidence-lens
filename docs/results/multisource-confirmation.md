# Multi-source confirmation on four held-out families

The frozen experiment returned 48 of 48 target views in 8 paired reconstruction conditions. Each family supplied two or four calibrated source images of one authored world. Each condition jointly optimized 4,096 Gaussians for 256 source-only updates, then rendered the same six target cameras at 256 × 256. The four families are the independent units.

## What additional source images changed

| Sources used to fit the scene | Mean RGB absolute error | RGB error above 0.05 / evaluable | Expected camera-Z error (m) | Common scoring coverage / evaluable |
| --- | ---: | ---: | ---: | ---: |
| M2 | 0.023 | 10.66% | 1.005 | 98.43% |
| M4 | 0.015 | 6.90% | 0.958 | 98.78% |

RGB is display sRGB in [0,1]. A positive label requires mean absolute channel error strictly above float32 0.05. Each reconstruction contributed 1,572,864 RGB-evaluable pixels. M2 had 167,672 confirmed errors and M4 had 108,545. All four families improved, by 2.83 to 4.19 percentage points. RGB validity and score availability are counted separately.

The depth column uses each reconstruction’s own valid pixels. On the pixels valid for depth in both reconstructions, the equal-family optical-axis mean absolute error was 1.011 m for M2 and 0.970 m for M4. The Gaussian prediction is expected depth and the Blender reference is the first surface intersection. The numerical archive provides the per-family pixel counts for both depth comparisons.

## How well each model located RGB errors

Scores below use all available sources. The single-source transfer always sees the first source. Average precision (AP) is evaluated on pixels scoreable by all four methods. Within each family, every view with both label classes has equal weight. The table then gives each family equal weight. Low coverage uses reference-valid pixels as its denominator. Low-group error uses RGB-evaluable pixels scored low. Unknown uses the RGB-evaluable domain.

| Model | Scene | Common AP | Defined families | Low coverage | Error among low | Unknown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B-L1-transfer | M2 | 0.500 | 4/4 | 43.41% | 3.68% | 0.00% |
| B-L1-transfer | M4 | 0.417 | 4/4 | 44.94% | 2.04% | 0.00% |
| MS-L1 | M2 | 0.747 | 4/4 | 75.78% | 0.38% | 1.57% |
| MS-L1 | M4 | 0.704 | 4/4 | 82.91% | 0.39% | 1.22% |
| MS-frozen | M2 | 0.727 | 4/4 | 73.76% | 0.46% | 1.57% |
| MS-frozen | M4 | 0.651 | 4/4 | 80.24% | 0.49% | 1.22% |
| MS-ft | M2 | 0.734 | 4/4 | 74.64% | 0.47% | 1.57% |
| MS-ft | M4 | 0.665 | 4/4 | 81.75% | 0.52% | 1.22% |

## Selection and scope

MS-L1 remains the default under the registered development rule. At seeds 1729 and 2718, fine-tuning gained 0.0075 and 0.0057 AP over freezing and fell 0.0268 and 0.0224 below MS-L1. The frozen operating thresholds are 0.43483029 for B-L1-transfer, 0.06428983 for MS-L1, 0.11090798 for MS-frozen and 0.11582898 for MS-ft. The four confirmation families became available together after these choices were recorded.

The numerical ZIP contains the full study in `study.md`, family and stratum counts, matched-pixel comparisons, risk–coverage curves, camera and reconstruction costs, gradient readbacks and the engineering failure ledger. These measurements cover opaque synthetic objects, the registered camera rig and a fixed optimization budget. [Download the numerical study](https://github.com/wany614/evidence-lens/releases/download/v0.3.0-rc.1/evidence-lens-numerical-v0.3.0-rc.1.zip) · [Download the CPU models](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1)

Evaluation run `20261123T171931255547Z-eb84d1b8e8a04712` has manifest `e56d65588be02b2b8a51c3e9d11f542d30e23e2db612263036c22ce415ee26d4`. Joint report `20261123T172148046549Z-ab231062830c43ae` has manifest `8979e720706fd8cf214a2623fe92c2ced91b99b7232397724b758c283ac4857f`. Its original `study-summary.json` SHA-256 is `35cc80a1c4237d406aa21e896d25c3bfc4dbe1cb3602d86b22f7e2188ac704bd`. The method SHA-256 is `1a2cbdb0ea3a52dfbde2e518c833b18d677209444c6aa345314ba3757aa3c9db` and the opening SHA-256 is `50b5cf30513ab739d60f44e404e914cb08681e2950bd91c058d6f328ddbd3026`.
