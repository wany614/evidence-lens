# Stage-one evidence

A raw-pixel replay matched the original RGB eligibility and decision counts on all four held-out families. Its 120 camera-by-method average-precision values differed by at most 2.22 × 10⁻¹⁶. The same raw files confirm a large auxiliary depth discrepancy. The camera, PLY and expected-depth controls passed on the recorded SHARP/gsplat pipeline. These results support a research preview for RGB-error risk on the measured synthetic scenes. The rc.2 candidate remains private pending the owner's repository and terms. [Issue and correction log](issues.json) · [Dependency and reuse record](impact.json)

## Frozen population and raw-pixel replay

We selected the lowest-numbered training family (tabletop) and development family (cycle stand) by ID. All six targets and the source were retained. The four original test families contributed their existing B0 and R1 views. Four new Blender source renders used the actual 256 × 256 R1 camera grids and original scenes. The acquired 256 × 256 neural input was the original box-filtered RGB image. The new source rays supply evaluation truth only. [Selection and metric definitions](../../protocol/phase1-depth-plan-v1.json) · [Source comparisons](B0-R1-native-source.csv)

| B0 outcome | Result |
| --- | ---: |
| Target views, original test families | 24, 4 |
| Reference-valid and RGB-evaluable pixels V = E | 6,291,456 |
| Common scoreable pixels S | 6,275,724 |
| Confirmed colour errors Y | 2,043,247 |
| L1-full / L1-geometry / L0 common-mask AP | 0.803633 / 0.770993 / 0.402398 |
| Opacity baseline / constant-prior AP | 0.354892 / 0.323163 |
| L1-full low-risk coverage over V / error rate among low | 62.71% / 11.03% |
| Confirmed errors flagged high / low / unknown | 1,608,191 / 419,324 / 15,732 |
| Confirmed errors outside the high-risk decision | 435,056 |
| Depth within evaluation domain Z / reference surfaces | 6,252,055 / 6,291,456 |
| Depth above tolerance ZY | 6,214,493 |
| Mean absolute depth error / above-tolerance fraction | 1.145853 m / 99.4002% |

AP, coverage, low-region error and depth headline values weight each family equally. Counts pool pixels across all targets. RGB uses mean absolute display-sRGB error greater than 0.05. Depth uses the registered surface, positive finite Z and alpha of at least 0.5. The per-pixel tolerance is the greater of 0.02 m or 2% of reference Z. The held-out numbers here use only 24 B0 targets. The separate camera-chain audit counts 43,128 zero-alpha pixels across 168 source and target grids from all 24 B0 families. Q1 is a separately recorded colour-only condition. Of the 39,401 excluded surface pixels, 15,732 had invalid predicted depth and a further 23,669 had low alpha. The depth intersection with RGB eligibility is tabulated explicitly. Of the comparable pixels, 6,152,685 had predicted Z below the reference. The 770 region rows contain empty-region indicators and fixed definitions. [RGB replay](rgb-replay.csv) · [Family depth counts](heldout-B0-reproduction.csv) · [70 views](per-view-all.csv) · [770 region measurements](per-view-regions.csv)

The source-view median ratio of predicted to reference depth was 0.610 for the training tabletop and 0.949 for the development cycle stand. At half input resolution the four existing test families' median source ratios changed by 0.0063 to 0.0155 in absolute terms. Positive, finite depth with enough opacity is a measurement domain. Opacity-weighted splat depth and Blender's first surface can differ, especially at boundaries. Synthetic-scene appearance and single-image scale are additional plausible contributors. The experiment does not identify each contributor's share. The estimator has no truth-fitted depth multiplier.

## Camera, colour and background

The original source focal length was 597.333 px at width 512 and 298.667 px at width 256. Both give f/W = 7/6. The archived camera uses integer-centred pixels. The production adapter supplies the half-pixel adjustment required by the CUDA rasterizer. Known splats traversed pinned SHARP PLY save/load and the unchanged renderer at the seven cameras. All 335 checks passed on Windows and on Mac regrading of its raw arrays. The largest centre error for the analytic splat fixture was 0.000029 px. A separate set of 1,008 Blender projection checks had maximum single-axis error 0.000072 px. Actual Position-pass samples in the original scenes can be subpixel locations within their image pixels. Their largest recorded sample reprojection offset was 0.6423 px. In a layered control, alpha was 0.625, accumulated depth was 1.6875 m and normalized expected depth was 2.70000005 m. [Pre-execution numerical expectations](../../protocol/phase1-geometry-probe-v1.json)

The original Blender World supplies illumination. The splat renderer uses black for empty rays. An analytic enclosure check on 24 actual scenes covered the padded sensor footprints of all 168 B0 grids and four R1 source grids. Every one of the 44,040,192 B0 reference samples hit a surface. Switching off the World for camera rays gave zero linear RGB difference in the closed tabletop control. Rendering the empty World exposed a nonzero background. Open-background scenes would require a matched camera-background contract before extending this result. [Backdrop protocol](../../protocol/phase1-backdrop-v1.json)

The study used a pre-trained SHARP checkpoint and gsplat 1.5.3. The formal cases use their outputs. The distributable example uses the project's two-metre plane projection. A closed-scene depth discrepancy persists with the tested input, coordinate and renderer conventions. With no specific unresolved focal-length or serialization hypothesis left by the checks, the optional additional neural generation would not separate the remaining explanations. [Pinned SHARP prediction path](https://github.com/apple-aiml-research/ml-sharp/blob/aed6527499ef91cba3b54c18d49a870f25947190/src/sharp/cli/predict.py) · [Pinned gsplat implementation](https://github.com/nerfstudio-project/gsplat/tree/937e29912570c372bed6747a5c9bf85fed877bae)

## Scope of verification

The independent local archive audit rechecked all 296 historical run manifests, 37,902 members and one shared checkpoint against the existing catalog, including its one recorded deliberate failure. The archive includes 144 cold-shelf runs and one previously restored replay. Full study imagery, geometric arrays, logs, active databases and SHARP weights stay in the private research archive. The internal rc.1 delivered identical decisions and decoded risk/state maps across macOS, Windows and Debian. A subsequent packaging review led to rc.2. Its isolated builder and grader bind the release to the exact commit, all 30 finite comparisons and the 12 output maps. The detached acceptance supplies the tested ZIP digest and platform runs. The owner has four fixed formal case boards with source, truth, prediction, observed RGB error, L1 risk, operating state, depth and visibility. The study's original model, parameters, predictions, labels and thresholds retain their identities.

Stage one supplies no new independent model-quality cohort. The technical prerequisites for a new, separately frozen RGB-risk pilot are listed in the handoff. No second-stage acquisitions or training are part of this release. [Stage-two entry](phase2-entry.md) · [Original study results](../results/README.md)
