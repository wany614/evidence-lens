# Evidence Lens v0.1.0

This version packages two fitted pixel-error classifiers, three comparison methods and a CPU example. Researchers can grade the original textured-plane example, score another compatible prediction bundle and export continuous risk and low/high/unknown decision maps. The CPU wheel depends on NumPy and Pillow. Package paths below start at the root of the downloaded ZIP. [Run the example](README.md) · [Model card](MODEL_CARD.md)

The scientific pipeline used Apple SHARP checkpoint SHA-256 `94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d` and gsplat 1.5.3 on Windows with CUDA. The renderer supplied display-sRGB colour, opacity and expected optical-axis depth (ED) in metres. B0 used the baseline source, R1 halved its resolution, and Q1 rounded only the three Gaussian colour coefficients through FP16.

## What is in the release

| Item | Contents |
| --- | --- |
| Model | Full and geometry-only logistic coefficients, preprocessing statistics, feature schema and frozen operating thresholds. |
| Example | Original 128 × 128 source, source and target camera calibration, numerical predictions from the textured-plane renderer and the expected v01 scores. |
| Results | Per-family and family-equal numbers, paired stress comparisons, timing and acquisition-completion tables. |
| Verification | Source identifiers, model and file hashes, training and comparison receipts, the wheel, and the clean CPU grading command. |

Two isolated macOS arm64 CPU preflights graded the original plane example. Python 3.12.16, NumPy 2.5.5 and Pillow 12.3.0 reproduced the recorded v01 scores exactly. Python 3.11.18, NumPy 1.26.4 and Pillow 11.0.0 reproduced validity and decisions on six targets across all five methods, with a maximum finite score difference of 1.1921 × 10⁻⁷ against the 10⁻⁶ tolerance. The packaged v01 has 15,714 finite L1-full scores and 670 unknown pixels. `demo.json` · `example/v01-prediction.png`

## Held-out evidence

The four held-out object layouts contributed 72/72 completed targets across B0, R1 and Q1. On B0 the family-mean fair AP was 0.804 for L1-full, 0.771 for L1-geometry and 0.323 for the constant baseline. The ranking intersection covered 99.75% of the reference-valid domain. L1-full marked 62.71% of the domain low risk, with 11.03% actual error among the pixels it accepted. The corresponding per-family medians, ranges and missed-error counts are in the model card. [Model card](MODEL_CARD.md) · `results/family-equal-summary.csv`

R1 changed confirmed RGB-error incidence by +1.46 percentage points versus B0. Q1 changed the predicted colour by a family-mean 1.06327e-05 display-sRGB mean absolute difference. Its geometry checks were bitwise equal for 24/24 paired target views. The held-out report run was `20261123T050620457525Z-a311c1632b7d4b1b`. `results/selection.json` · `results/Q1-numerical-response.csv`

## Version identity

| Record | Identifier |
| --- | --- |
| Training run | `20261123T042239692597Z-0616eff18c3149a7` |
| Model and threshold-selection run | `20261123T042819219876Z-27eaa01eba29489d` |
| Frozen model index SHA-256 | `b3c65130bf803611e3904ac69a270f6944b48a8d78ce049bdc1424c15ba6cb54` |
| Frozen method code commit | `cf72c00c555929ab55ea372dca4e98b6504a8721` |
| Comparison protocol SHA-256 | `30fd194160cb74da1adc244c643d2f1cb13f8d1b6a89744b3904480f4055c3be` |

The package manifest records the commit used to build this release, the wheel and asset digests, and the licenses for source, parameters and example data. `release.json` · `protocol/method-freeze.json`
