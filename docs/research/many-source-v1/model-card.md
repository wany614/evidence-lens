# Evidence Lens multi-source CPU model

## Intended use

For a fixed static Gaussian reconstruction, the model orders target pixels by their risk of an RGB error. Its training event is FP32 display-sRGB mean absolute difference above 0.05 on the frozen 378 × 252 grid. A finite score is an uncalibrated ranking. A request supplies a fixed target prediction, the frame, source photographs and matching source self-renders. A separate evaluator can add independent reference labels.

## Training, model identity and operating point

The final model is `b-cpu-lr-l2-0.1-final`, SHA-256 `b9e6b761f2ff15d2c2aada851aa0e239f550163368107692b0c536843f11c721`. It has 44 coefficients and one intercept. The final fit used 144,000 source-query rows, including 6,833 positive rows from six checkpoints and 36 target units across the three original training families. Repeated querying of a physical pixel redistributes that pixel's training weight. The fitted weighted finite means, imputation values and scales are included in model.json.

The development endpoint selected L2 = 0.1 from three candidates, followed by one deterministic refit. The selected simple comparison was `residual`. The choice used the authored b04 family and ETH3D delivery_area with equal scene weights. All coefficient fitting used b01-staggered-arcade, b02-terraced-court and b03-woven-truss. The operating threshold is null. Calibration fits and neural updates both total zero. At Q4 the study retained the released MS-L1 and the genuine fine-tuned MS-ft as separate controls.

## Frozen external evaluation

| Scene | Labelled errors / pixels | Common K16 pixels | Accepted | New errors | Simple errors | Target-equal difference |
|---|---:|---:|---:|---:|---:|---:|
| living_room | 332,811 / 952,560 | 952,560 | 190,510 | 51,334 | 45,524 | 3.050% |
| bridge | 341,069 / 952,560 | 952,560 | 190,510 | 44,943 | 46,014 | -0.562% |

The primary comparison status is `completed_negative`. Each target accepts floor(20% of its shared finite domain). Each of the two ETH3D confirmation scenes supplies ten repeated target views. P1 measures missed comparable evidence using an explicitly costed full audit. Independent mesh visibility is available for original families. External scene geometry uses supplied camera calibration in scene units. Outputs describe these two external cases and the registered families. Their use for new environments, dynamic video, physical safety or historical authenticity remains unvalidated.

## Provenance and rights

Scientific source commit `6e55dd6969b58faa08dc4a7033d7c6687a919dad`. Stage A `sha256:6e8e837afa0818cdf8c100b494672b46c10bdbd22e522f388983543f0a5b0f23`. Final fit `20261124T115857282953Z-3f6e989f5d4844b1`, confirmation permission `20261124T122014191481Z-ddac5a608d5a4d90`, results `20261124T122203772897Z-ba03676b98d64b31`. The upstream ledger includes 15 registered Gaussian fits. The CPU study uses gsplat v1.5.3 reconstructions, the registered analysis profile, and known calibration. See training.json, development-selection.json and freeze.json for the input manifests and feature identities.

Copyright 2026 wany614. The original logistic parameters and preprocessing are Apache-2.0. Original explanatory text and grantable measurements are CC BY 4.0. ETH3D imagery is subject to its own terms. The CPU parameters do not require the optional DINO control package. [ETH3D](https://www.eth3d.net/) · [Stage A](https://github.com/wany614/evidence-lens/releases/tag/v0.5.0) · [Legacy risk models](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1)
