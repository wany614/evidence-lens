# Frozen risk scoring on DTU photographs

DTU 实拍外测 set6 采用发布包四个冻结模型。MS-L1 在 M2／M4 的共同域 AP 为 0.9548／0.9109，共同域错误基率为 94.71%／89.80%，低风险组错误为 89.16%／82.34%。单个外测对象的排序与拒判须分开解读。数值与代码可公开，实拍图板仅供本地研究。

The 256-step arm scores six target views per object and reconstruction. M2 uses the two endpoint sources. M4 uses all four. B-L1-transfer reads s00. The three multisource methods read every source allowed by their reconstruction. All coefficients, preprocessing and thresholds come from Evidence Lens v0.3.0-rc.1.

Set1 was used for integration. Set6 was the registered external acquisition. One external object supports a within-object comparison. The six cameras and the four methods share that object. A cross-object interval and performance on other physical scenes remain undetermined.

| Role and object | Arm | Condition | Method | Common AP (views) | Common prevalence | RGB errors / E | Low / V | Errors / low | Unknown / E |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| debug set1 | main-256 | M2 | B-L1-transfer | 0.8643 (6/6) | 86.41% | 89.66% | 53.85% | 91.96% | 0.01% |
| debug set1 | main-256 | M2 | MS-L1 | 0.9140 (6/6) | 86.41% | 89.66% | 3.24% | 73.45% | 33.28% |
| debug set1 | main-256 | M2 | MS-frozen | 0.9249 (6/6) | 86.41% | 89.66% | 7.12% | 67.14% | 33.28% |
| debug set1 | main-256 | M2 | MS-ft | 0.9328 (6/6) | 86.41% | 89.66% | 10.10% | 67.03% | 33.28% |
| debug set1 | main-256 | M4 | B-L1-transfer | 0.8547 (6/6) | 85.54% | 87.53% | 53.33% | 88.45% | 0.00% |
| debug set1 | main-256 | M4 | MS-L1 | 0.9313 (6/6) | 85.54% | 87.53% | 3.54% | 52.61% | 17.10% |
| debug set1 | main-256 | M4 | MS-frozen | 0.9285 (6/6) | 85.54% | 87.53% | 9.24% | 64.63% | 17.10% |
| debug set1 | main-256 | M4 | MS-ft | 0.9309 (6/6) | 85.54% | 87.53% | 11.95% | 65.73% | 17.10% |
| external set6 | main-256 | M2 | B-L1-transfer | 0.9454 (6/6) | 94.71% | 94.77% | 51.81% | 95.02% | 0.49% |
| external set6 | main-256 | M2 | MS-L1 | 0.9548 (6/6) | 94.71% | 94.77% | 2.38% | 89.16% | 28.74% |
| external set6 | main-256 | M2 | MS-frozen | 0.9558 (6/6) | 94.71% | 94.77% | 1.79% | 86.17% | 28.74% |
| external set6 | main-256 | M2 | MS-ft | 0.9575 (6/6) | 94.71% | 94.77% | 1.74% | 84.97% | 28.74% |
| external set6 | main-256 | M4 | B-L1-transfer | 0.9014 (6/6) | 89.80% | 91.00% | 48.24% | 91.53% | 0.00% |
| external set6 | main-256 | M4 | MS-L1 | 0.9109 (6/6) | 89.80% | 91.00% | 0.60% | 82.34% | 20.02% |
| external set6 | main-256 | M4 | MS-frozen | 0.9101 (6/6) | 89.80% | 91.00% | 0.49% | 75.52% | 20.02% |
| external set6 | main-256 | M4 | MS-ft | 0.9124 (6/6) | 89.80% | 91.00% | 0.45% | 72.99% | 20.02% |

## What the counts measure

V is the independently valid photographic crop. E contains its pixels with valid predicted RGB. Y counts pixels of E where the mean absolute display-RGB channel error exceeds 0.05. S contains finite scores on E, U is its complement on E and low means finite risk strictly below the released threshold. All ratios keep their original numerator and denominator.

Common AP uses E intersected with the four finite score masks. Own AP uses each method’s deployment mask. Exact score ties are grouped. Only views containing positive and negative labels have an AP. Views with defined AP carry equal weight within an object, and objects with defined AP carry equal weight in their cohort. A zero count of RGB errors gives a zero incidence when E is positive. Empty rate denominators give N/A.

The object-only region is unavailable for photographs without an independently verified image-space mask. The full crop includes background. The auxiliary 3×3 test compares each centre prediction with the closest colour in its valid reference neighbourhood. It can hide misalignment or texture errors. Predicted expected camera depth and source support are engineering diagnostics. Surface-depth supervision is unavailable.

The accompanying tables distinguish failed acquisition, missing reference, invalid RGB and unknown risk. A poor reconstruction stays in the registered matrix. The source-only scoring run and the separate target-photograph evaluations are pinned in [provenance](provenance.json).

Read the [registered matrix](matrix.csv), [individual targets](target-metrics.csv), [object totals and rates](object-metrics.csv), [cohort estimates](cohort-metrics.csv), [paired changes](paired-conditions.csv), [engineering diagnostics](render-diagnostics.csv) and [recorded resources](resource-readback.csv).

[Risk–coverage points](risk-coverage.csv) preserve whole score groups and the actual frozen operating point. The [descriptive score bins](score-bins.csv) use ten equal-width bins.

The [reconstruction ledger](reconstruction.csv) records measured GPU update time, rendering time, source pixels and PyTorch CUDA allocator peaks. The [source ledger](source-supervision.csv) gives initial and final source-view error and updates per source.
