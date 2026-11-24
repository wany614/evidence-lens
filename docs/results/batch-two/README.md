# Frozen scores, empirical cutoffs, and camera selection on DTU

Six calibration objects supplied no usable MS-L1 operating point under the registered target. On two previously exposed objects, risk weighting the camera selection slightly reduced target error on one object and selected the same photographs as uniform coverage on the other. The project also verified a standard-format import path and the CPU scorer on original, redistributable evidence. These experiments used release `v0.3.0-rc.1` and the recovery protocol recorded at `bc8da49815751fa7a6e191d0dc17c9f987f9c825`. [Six-object T0 confirmation](../t0/README.md) · [Released model card](../../releases/v0.3.0-rc.1/MODEL_CARD.md)

## Calibration on six physical subjects

Set48 (mugs), set75 (green pepper), set110 (bronze statue), set42 (cabbage), set77 (teapot) and set52 (anatomical model) each provided six reference cameras. Each condition used its own two or four authorized source images. All 12 reconstructions, 72 target predictions, 288 four-method score maps and the score-only candidates were sealed before the collective reference opening. Each reference-valid domain contained the full 256² crop. The error event was a mean absolute display-sRGB difference strictly above 0.05. [Cohort and source identities](cohort/README.md) · [Reference and transfer receipts](cohort/complete-intake/cohort-summary.json)

| Original MS-L1 counts across C6 | M2 fixed | M4 fixed |
|---|---:|---:|
| Reference-valid V and comparable E, each | 2,359,296 | 2,359,296 |
| Confirmed errors Y | 1,723,315 | 697,119 |
| Finite scores S | 2,213,504 | 2,044,171 |
| Low L / errors LY | 689,596 / 410,540 | 1,300,140 / 215,589 |
| High H / errors HY | 1,523,908 / 1,231,355 | 744,031 / 431,124 |
| Unknown U / errors UY | 145,792 / 81,420 | 315,125 / 50,406 |

The original cutoff gave object-equal mean coverage / low-group error of 29.23% / 62.86% for M2 and 55.11% / 17.29% for M4. The table pools pixels for auditing. For inference each subject first contributes its six-view L/V and LY/L, then each subject receives equal weight. Among the 201 precommitted anchors per full fit with all subjects accepting pixels and mean coverage at least 20%, minimum observed mean LY/L was 60.69% for M2 and 15.15% for M4. These are descriptions of the calibration curves. Both formal profiles carry a null cutoff. All twelve five-object training folds also returned a null cutoff, so their omitted subjects have no adapted error rate. [Full curves, worst objects and leave-one-out analysis](calibration/README.md) · [Counts for every view and method](calibration/original-method-views.csv)

## Development camera policy

The full-crop six-target MAE difference, risk-weighted minus uniform coverage, was −0.000257 for set1 and exactly zero for set6. The latter used one immutable reconstruction for both trajectories. With fixed source cameras the three optimizer seeds spanned 0.005505 and 0.001643 MAE on set1 and set6. These two development comparisons leave the incremental benefit of risk-weighted selection unresolved. [All strategies, paired denominators, costs and three random repeats](active-views/README.md) · [Twelve calibration cells and fourteen development trajectories](reconstruction/README.md)

## Software and model identity

The wheel is `0.4.0b1`. The NumPy/Pillow default remains the previously selected MS-L1. The optional released MS-ft retains its actual fine-tuned DINOv2 backbone and projection/head. The seed-1729 fit made 288 optimizer updates and changed 1,775,998 eligible backbone scalars; the model card links the original gradient and weight-change receipts. Batch two introduced no risk-model fitting on project data. The two C6 threshold searches and twelve sensitivity folds have separate records. [Download the original weights](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1) · [Model and training lineage](../../releases/v0.3.0-rc.1/MODEL_CARD.md)

Thirty original-scene comparisons covered COLMAP 3.10 text and binary, three declared Gaussian representations and five cameras. Native and imported data were rendered with gsplat 1.5.3 on Windows CUDA. The largest absolute differences were 2.98×10⁻⁷ in display RGB and opacity and 4.24×10⁻⁵ m in expected camera-Z, with identical depth support. Clean CPU scoring and offline HTML use prepared evidence. A matched C6 layout displays its infeasible result and preserves the original maps. The helical-bead subject displays its separate applicability status. [Reader contracts, rejects and measured costs](../../tooling/assets/README.md) · [Original interactive report](../../../examples/helical-beads/report.html) · [Outside-scope example](../../../examples/helical-beads/outside-scope.html) · [Clean Linux replay](linux-cpu.json)

## The still-closed confirmation cohort

Set118, set76, set45, set103, set61 and set32 were assigned before C6 opening, using the same registered eligibility pool and allocation seed. No T1 photograph was accessed. The final freeze binds the release, seed, numerical implementation, candidate pool, query construction, 42-cell upper bound, error denominators and stopping rule. Both adapted cutoff endpoints have no fitted value. Independent T1 evaluation of original risks and paired camera choices belongs to the next batch. [Prospective endpoints](holdout/prospective.json) · [Final method and data freeze](holdout/t1-freeze.json) · [Acquisition audit](holdout/acquisition-audit.json)

| Decision or independent check | Recorder run ID |
|---|---|
| Physical allocation | `20261124T014705638610Z-007327d6cbbc4242` |
| C6 algorithm registration | `20261124T020129961316Z-2b715d6ae3314c12` |
| Score-only candidates, full and leave-one-out | `20261124T020900657339Z-71eb2236f1ad42c0` |
| C6 collective opening | `20261124T021200190993Z-38fecc1ba0c146e0` |
| Calibration and independent native recount | `20261124T021235608840Z-f0d12afab15c4b3e` · `20261124T021259461920Z-0f16697bc8ca4700` |
| Final development MAE | `20261124T022032641624Z-d7da793c7f6b4dff` |
| Windows CUDA format parity | `20261124T021118010901Z-e7bca336730e47df` |
| Original portable package | `20261124T021735523259Z-8ee25029bf9d4422` |
| Prospective T1 endpoints and final freeze | `20261124T020659530443Z-846609b6d59c485a` · `20261124T023243052988Z-5e3007d1b3ed48dc` |
| Windows native CPU replay and independent numerical readback | `20261124T023920980304Z-851ccb4c1c7c4bff` · `20261124T024840489724Z-802b4dde406a4262` |
| Whole-batch resource census | `20261124T024917877156Z-aff148a6b7024c46` |
| Windows archive and native-stage verification, then cumulative census | `20261124T025913936867Z-db39ad69bea64f5c` · `20261124T025958729214Z-9bf1bb2c875f4aee` |
| Independent readback of all nine ordered Windows stages | `20261124T030422988018Z-770cf828f0f945dd` |

The 25 actual fits accumulated 956.795 seconds of Windows recorder time. The first census measured 3,272,347,811 file bytes. The cumulative census after the Windows archive check measured 3,297,594,802. Mac and Windows reproduced the three-target bead scores. The offline Linux container reproduced the six-target published bell-cage result. [Platform comparisons, failure history and costs](resources/README.md)

The public figures and counts identify their inputs. DTU photographs and their visual comparisons remain in the private research archive. Raw examples in the CPU distribution come from an original analytic scene. [Study software and data notices](../../../THIRD_PARTY_NOTICES.md) · [Official DTU data and citation](https://roboimagedata.compute.dtu.dk/?page_id=36)
