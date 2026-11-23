# R0 historical checkpoint diagnosis

Only 453 of the 4,096 initial Gaussians contributed to either DTU source image in M2. For M4 the union over four sources was 497. By the end of the original 256 updates, those same rows had changed in every parameter group. Source error decreased in all four reconstructions, while the six target views retained substantially higher errors. R0 used the original set1 and set6 checkpoints and performed zero optimizer updates. [Parameter changes](parameter-rows.csv) · [Source contributions](native-source-union.csv) · [Original optimization logs](historical-log.csv)

| Original condition | Source MAE, initial → final | Target MAE, final | Contributing source rows, initial → final | Median motion of changed rows |
|---|---:|---:|---:|---:|
| set1 M2 | 0.403224 → 0.074569 | 0.260280 | 453 → 162 | 0.2065 m |
| set1 M4 | 0.385142 → 0.102017 | 0.203712 | 497 → 191 | 0.1842 m |
| set6 M2 | 0.313180 → 0.091704 | 0.252138 | 453 → 152 | 0.2037 m |
| set6 M4 | 0.297835 → 0.118072 | 0.213622 | 497 → 163 | 0.1802 m |

MAE averages absolute display-sRGB channel differences across every pixel in each 256 × 256 crop, then weights the views equally. The stored source losses agree with a fresh source-only backward pass at each endpoint. The 24 saved target images reproduce the original per-view mean errors and counts of pixels with RGB MAE strictly above 0.05. Eight private boards pair the original photographs, original renders, error maps and accumulated opacity. Their names and SHA-256 digests are in the summary. [View measurements](image-quality.csv) · [Target count readback](original-target-error-parity.csv) · [Board digests](summary.json)

## Why so few rows participated

1. **Initial placement relative to the cameras has the strongest direct evidence.** Regenerating PCG64 seed 43171 reproduced the five original FP32 arrays byte for byte. The centre box ranged from (−1.7, −0.8, −0.05) to (1.7, 1.6, 1.75) metres, with initial scale axes between 0.03 and 0.105 m. Position Adam used a learning rate of 0.008 in metre-valued coordinates. With the same arrays, 151 M2 and 180 M4 DTU centres fell inside a source frame. Under the archived Stage C family 057 source cameras, 2,441 centres fell inside for either source set. The respective unions of analytic 3σ ellipses intersecting the frame were 1,046, 1,329 and 3,144. This zero-fit control measures camera geometry. The 057 archive records initial source losses and final renders, with no pixel-level initial RGB file. [Source-camera unions](source-union.csv) · [Scale, anisotropy and opacity quantiles](state-quantiles.json)
2. **A few broad footprints carried much of the image.** At the initial set1/M2/s00 checkpoint, 99 centres lay inside the frame. An independent first-order ellipse calculation found 690 intersections, including 444 offscreen centres whose intersecting ellipse bounds spanned more than one image dimension. The CUDA rasterizer assigned tiles to 370 rows, 348 made a positive contribution, and 8 supplied half the accumulated weight. Analytic ellipses use the unclamped pinhole Jacobian. Native radii and tile assignment include backend clipping. For the colour probe, differentiating the sum of the linear red image with respect to each Gaussian’s red value measured its accumulated transmittance-weighted opacity across pixels. Its sum agreed with rendered opacity within 0.030 over each whole 65,536-pixel image. [Projected geometry](projected-views.csv) · [Native per-view weights](native-views.csv) · [gsplat v1.5.3 rasterization](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py)
3. **The original fixed topology and optimization budget left many rows unchanged.** Positive source photometric gradient and positive colour-probe weight identified the same rows at the eight named checkpoints. The original logs contain whole-group gradient norms for all updates. At the first update, 20 independently reconstructed whole-group norms agreed within 5.40 × 10⁻⁹. Historical per-row gradients and weights between endpoints are unavailable. The evidence supports changing where geometry starts. It does not isolate the separate causal effects of initial scale, topology or optimizer duration. [First-step readback](first-step-group-gradient-parity.csv) · [Endpoint gradients and weights](native-source-union.csv)

## Camera check and the largest residual

The two implementations used the historical 0.05–30 m clip interval and the same half-pixel conversion to integer-centre coordinates. Across 78 camera/checkpoint combinations, 37,868 rows had positive native radii on both axes. Camera Z agreed within 3.55 × 10⁻⁷ m. The pre-recorded per-axis tolerance was `0.003 + 0.000005 × |reference coordinate|` pixels, with an absolute Z tolerance of 10⁻⁵ m. [Frozen checks](report-config.json) · [View-by-view results](projection-parity.csv) · [gsplat coordinate transform and pinhole projection](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/include/Utils.cuh)

One axis exceeded the separate 0.001 px review threshold. On set6/M2/final/v02, Gaussian 1156 (zero-based) projected to (−62.026536, −2545.580307) in FP64 and (−62.026615, −2545.582520) on CUDA. Its depth was 0.05763455 m, only 7.635 mm past the near plane. FP32 evaluation of the original camera transform with contracted multiply-adds reproduced this point’s CUDA X, Y and Z exactly. The largest difference among 18,204 in-frame centres was 0.000340 px. Across all rows the signed mean deviations were −0.00000152 px in X and −0.00000039 px in Y. These measurements explain the isolated residual as rounding amplified by perspective and give no evidence of a systematic half-pixel offset, axis reversal or scale mismatch in the tested cameras. [Exact witnesses](roundoff.json) · [FP32 arithmetic comparisons](roundoff-routes.csv)

## Reproduction and original records

The eight successful R0 records comprise four source-gradient and colour probes, one native CUDA projection, one CPU image/geometry readback, the joint report and the FP32 follow-up. The resource table distinguishes recorder wall time, worker timing and sampled memory. WDDM per-process GPU memory was unavailable. Device-wide and PyTorch allocator measurements are reported separately. [Resource table](all-run-resources.csv) · [Final file inventory and original run hashes](public-members-v2.json)

An owner of the named sealed inputs can replay the Mac steps from the checkout using the project environment. Each invocation assigns a fresh recorder identity. The probe packages can also be regenerated from the original D1 selections using `scripts/r0_handoff.py`. Numerical outputs are CC BY 4.0, and protocol configurations and scripts are Apache-2.0. Original photographs, comparison boards and checkpoint arrays stay in the private archive.

```sh
results=docs/results/reconstruction-recovery/r0
python3 scripts/r0_record.py --store "$STORE" --config "$results/cpu-config.json"
python3 scripts/r0_record.py --store "$STORE" --config "$results/report-config.json"
python3 scripts/r0_record.py --store "$STORE" --config "$results/roundoff-config.json"
```
