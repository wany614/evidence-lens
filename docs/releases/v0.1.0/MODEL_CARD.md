# Evidence Lens v0.1.0 model card

Evidence Lens estimates where a target image synthesized from one source view may have a large colour error. The score applies to one pixel in one predicted target view. **L1-full** is the primary model. Its frozen identifier is `20261123T042819219876Z-27eaa01eba29489d`. The five JSON files in its model directory total 12,387 bytes. File paths below are relative to the downloaded release ZIP. Its index is `model/model.json`.

The study used Apple SHARP checkpoint SHA-256 `94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d` and gsplat 1.5.3. The renderer produced display-sRGB colour, accumulated opacity and expected optical-axis depth (ED) in metres. The three held-out conditions were the baseline B0, a half-resolution source R1 and colour-coefficient rounding Q1.

## Inputs and decisions

CPU inference takes the original RGB input, perspective camera intrinsics and poses, a source self-render, and the predicted target RGB, accumulated opacity and expected optical-axis depth in metres. Inputs use integer-centred pixels and display sRGB in [0, 1]. A pixel needs valid rendered RGB and opacity and finite, positive, verified target depth for learned scoring. Unsupported depth gives an unknown learned score. Source features can carry missing-value flags where a correspondence is unavailable. `model/feature_schema.json`

Higher scores rank pixels as more likely to have a large error. A finite score below its frozen threshold gives a low decision. A finite score at or above it gives a high decision. Unknown has code 255 and appears grey in the preview. The score has no established probability calibration. A low-risk decision can still contain an error. `model/thresholds.json`

**L1-full** uses geometry, appearance and missingness. Its 26 input columns include source and target opacity, reprojected depth conflicts, camera motion, distance to the input border, local depth and RGB changes, source-image and self-render residuals, and frame-relation flags. **L1-geometry** was fitted separately with 20 columns after removing the three RGB features and their missingness flags. The three baselines are the training-family mean error incidence, one minus target opacity, and the L0 support/depth rule. `model/L1-full.json` · `model/L1-geometry.json`

## Training and operating point

The experiment used 24 distinct, procedurally authored, static opaque object layouts. Each family contributed one source and six nearby target cameras at 512 × 512 under baseline condition B0. Families 001–016 supplied training pixels. Cycle stand, balance scale, corner booth and switchback ramp supplied the operating thresholds. Four other layouts were reserved for final evaluation. The labels came from independent Blender target renders. A pixel was positive when the mean absolute difference across its three display-sRGB channels was **strictly greater than 0.05**. Only locations with valid reference and prediction RGB received a colour-error label. `protocol/families-v1.json` · `protocol/study-v1.json`

Training sampled 4,096 eligible pixels per target, giving 393,216 rows with 120,458 positives. The fit weighted families equally and views equally within each family. Both L2 logistic regressions used four blocked family folds to choose among inverse regularization strengths 0.1, 1 and 10. Both selected 0.1. Numeric features used training-fold median imputation and weighted standardization. The exported NumPy calculation agreed with the training implementation to within 3 × 10⁻⁸ across the sampled rows. `training/training_dataset_manifest.json` · `training/cv_results.json` · `proofs/model-fit.json`

The four development layouts selected the complete score-tie boundary nearest 80% family-mean low-risk coverage. L1-full reached 80.00% coverage with 6.47% error among accepted pixels. L1-geometry reached 80.00% with 7.95% error. Their thresholds are approximately 0.34015505 and 0.33971211. Inference uses their full stored precision for every condition. The constant baseline had one score group and accepted all development pixels at its selected boundary. L0 used its fixed binary rule point. `model/thresholds.json`

The formal SHARP generation and gsplat rendering ran on a Windows CUDA GPU. The recorded macOS CPU fit took 7.48 seconds with sampled peak process-tree memory of 734.5 MiB. Threshold selection took 7.04 seconds with a sampled peak of 347.2 MiB. These are individual stage measurements for this dataset and environment. `results/stage-times.csv`

## Held-out results

The four held-out layouts provided six planned targets per condition. 72 of 72 target predictions completed. V counts every reference-valid RGB pixel. E also requires valid predicted RGB. Fair average precision (AP) uses the five-method score intersection S within E, averages only targets containing positive and negative errors, and then gives each eligible family equal weight. S/V gives the share of the evaluation domain represented by that comparison. System coverage and error rates use their own stated denominators. `results/family-results.csv` · `results/data-completion.csv`

### Baseline B0

| Method | Fair AP | Low-risk coverage L/V | Error among low LY/L | Known errors flagged HY/Y | Unknown among E |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1-full | 0.804 | 62.71% | 11.03% | 78.42% | 0.25% |
| L1-geometry | 0.771 | 60.57% | 11.20% | 78.24% | 0.25% |
| L0 | 0.402 | 81.70% | 27.03% | 31.88% | 0.00% |
| 1-alpha | 0.355 | 77.01% | 31.84% | 24.21% | 0.00% |
| constant | 0.323 | 100.00% | 32.48% | 0.00% | 0.00% |

L1-full accepted 62.71% of B0 test pixels at its development threshold, with 11.03% error among those accepted. On development the same figures were 80.00% and 6.47%. The common mask covered 99.75% of V on average and supplied 24/24 two-class targets. L1-full’s median family AP was 0.800, with a range of 0.768 to 0.846. Its median family low-risk error rate was 11.08%, ranging from 6.89% to 15.07%. `results/family-equal-summary.csv`

| B0 layout | L1-full fair AP | Low-risk coverage | Error among low | True-error pixels marked low |
| --- | ---: | ---: | ---: | ---: |
| Drum kit | 0.846 | 62.54% | 9.88% | 97,196 |
| Picket gate | 0.773 | 52.25% | 15.07% | 123,829 |
| Potter’s wheel | 0.828 | 60.02% | 12.27% | 115,863 |
| Orbital mobile | 0.768 | 76.04% | 6.89% | 82,436 |

Across the 24 planned B0 targets there were 2,043,247 known high-error pixels. L1-full marked 419,324 of them low and left 15,732 unknown. These are pixel counts pooled across families. On geometrically disoccluded surfaces, the family-mean incidence of strict RGB error among evaluable pixels was 36.23%. On surfaces visible to the source camera it was 33.14%.

### Input resolution and colour precision

R1 halves the source to 256 × 256 before a fresh generation and retains 512 × 512 targets. Q1 rounds only the three B0 Gaussian colour coefficients through FP16 before rerendering. The operating thresholds are fixed in both conditions.

| Condition | Targets completed | Confirmed error Y/V | Invalid output (V−E)/V | Common coverage S/V | L1-full fair AP | L1-full low coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 24/24 | 32.48% | 0.00% | 99.75% | 0.804 | 62.71% |
| R1 | 24/24 | 33.93% | 0.00% | 99.32% | 0.812 | 61.22% |
| Q1 | 24/24 | 32.48% | 0.00% | 99.75% | 0.804 | 62.71% |

R1 changed confirmed error incidence by +1.46 percentage points versus B0. Conditional error among E was 33.93% for R1 and 32.48% for B0. The mean absolute change between Q1 and B0 predicted RGB was 1.06327e-05 in display-sRGB units, giving each family equal weight. The largest paired per-pixel channel-mean change was 0.0001127621. The target-to-reference error label uses 0.05 per pixel, so even a smaller change can move a pixel already close to the boundary across it. This Q1 measurement covers one round trip of the degree-zero colour coefficients in these four layouts. Geometry, opacity, support, geometry-only scores and rule diagnostics were bitwise identical in 24/24 paired targets and 4/4 paired source self-renders. The target comparisons counted 0 differing geometry-related values and 6,237,618 differing full-model scores. `results/paired-family-equal.csv` · `results/Q1-numerical-response.csv`

## Intended use and limits

Use the maps to inspect synthetic novel views for likely reconstruction errors near changed visibility, edges and weak support. The measured domain is opaque, static object layouts under a single Apple SHARP checkpoint, a fixed rendering pipeline and nearby perspective cameras. Real photographs, transparent or moving objects, wide camera changes and other synthesis methods lie outside the measured domain. Four held-out families give a descriptive case series. Related pixels and nearby views share geometry, so the family is the unit of comparison.

## Running the CPU example

The packaged input is an original Blender tabletop image. An independent renderer maps it onto an assumed plane at two metres and predicts six 128 × 128 views. The isolated macOS arm64 CPU preflight on Python 3.12.16, NumPy 2.5.5 and Pillow 12.3.0 reproduced all five methods exactly on the fixed v01. A second Mac installation on Python 3.11.18, NumPy 1.26.4 and Pillow 11.0.0 matched validity and decisions across all six targets and five methods. Its largest finite score difference was 1.1921 × 10⁻⁷ against the recorded reference, within the 10⁻⁶ tolerance. L1-full produced 12,181 low, 3,533 high and 670 unknown pixels on v01. The six views together produced 70,874 low, 22,761 high and 4,669 unknown. `example/provenance.json` · [Run the example](README.md)

The downloaded example includes its original v01 prediction preview at `example/v01-prediction.png`.

## Attribution and terms

Formal evaluation used SHARP by Mescheder and colleagues, *Sharp Monocular View Synthesis in Less Than a Second*, and gsplat for rendering. Evidence Lens fitted the two logistic regressions from prediction features and independent Blender error labels. The release supplies those fitted parameters and an original textured-plane example. The respective license documents govern source, parameters and example assets. [SHARP](https://github.com/apple-aiml-research/ml-sharp/tree/aed6527499ef91cba3b54c18d49a870f25947190) · [gsplat](https://github.com/nerfstudio-project/gsplat/tree/937e29912570c372bed6747a5c9bf85fed877bae) · `LICENSE` · `licenses/MODEL-LICENSE.txt` · `licenses/EXAMPLE-LICENSE.txt`
