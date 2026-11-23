# Development-only reconstruction budget

Each endpoint uses DTU set1, the same source photographs, calibration, starting parameters and 4,096 Gaussians. The main and diagnostic endpoints ran separately. The 1024-step arm measured source and target RGB error. It added zero risk scores.

| Condition | Updates | Mean source MAE | Target MAE | Target errors / E |
|---|---:|---:|---:|---:|
| M2 | 256 | 0.0746 | 0.2603 | 352,553 / 393,216 (89.66%) |
| M2 | 1024 | 0.0649 | 0.2553 | 345,243 / 393,216 (87.80%) |
| M4 | 256 | 0.1020 | 0.2037 | 344,182 / 393,216 (87.53%) |
| M4 | 1024 | 0.0816 | 0.2342 | 345,034 / 393,216 (87.75%) |

The [four endpoint totals](endpoints.csv), [individual targets](target-budget.csv) and [source views](source-budget.csv) preserve the original counts and timing. Each 256-step target also agrees with the frozen independent RGB evaluation on reference and prediction identity, V, E, Y and the sum of absolute mean RGB error.
