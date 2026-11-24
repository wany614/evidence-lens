# Frozen evaluation on six new DTU subjects

The collective opening covered 288 scored entries out of 288 scheduled. Each physical subject contributes two conditions and six target views per condition. Ground truth is the full centre-cropped photograph at the registered camera. A pixel is an error when its mean absolute display-RGB difference exceeds 0.05.

## Reconstruction and the released MS-L1 operating point

| Object | Sources | Source MAE | Target MAE | RGB errors / E | Low / V | Errors / low | Unknown / E |
|---|---|---:|---:|---:|---:|---:|---:|
| set2 (rabbit statue) | M2 | 0.0124 | 0.1842 | 270858/393216 | 62650/393216 | 21880/62650 | 17362/393216 |
| set2 (rabbit statue) | M4 | 0.0240 | 0.1166 | 196383/393216 | 84950/393216 | 25010/84950 | 16995/393216 |
| set4 (bird doll) | M2 | 0.0085 | 0.1937 | 241930/393216 | 121143/393216 | 48380/121143 | 22072/393216 |
| set4 (bird doll) | M4 | 0.0216 | 0.1200 | 162016/393216 | 137570/393216 | 34175/137570 | 40400/393216 |
| set10 (cardboard box) | M2 | 0.0113 | 0.2039 | 240807/393216 | 101291/393216 | 31439/101291 | 22913/393216 |
| set10 (cardboard box) | M4 | 0.0195 | 0.0863 | 139685/393216 | 174081/393216 | 27326/174081 | 31390/393216 |
| set11 (power supply) | M2 | 0.0040 | 0.3440 | 349196/393216 | 50713/393216 | 42100/50713 | 18428/393216 |
| set11 (power supply) | M4 | 0.0156 | 0.0925 | 162209/393216 | 196848/393216 | 59789/196848 | 21736/393216 |
| set12 (beverage cans) | M2 | 0.0083 | 0.1671 | 189791/393216 | 167689/393216 | 46597/167689 | 23025/393216 |
| set12 (beverage cans) | M4 | 0.0167 | 0.0962 | 116146/393216 | 188499/393216 | 34711/188499 | 31714/393216 |
| set31 (pumpkin) | M2 | 0.0054 | 0.1391 | 218410/393216 | 142868/393216 | 45348/142868 | 27050/393216 |
| set31 (pumpkin) | M4 | 0.0151 | 0.0916 | 136315/393216 | 201799/393216 | 36727/201799 | 31149/393216 |

## Four frozen methods on identical current candidates

Each subject receives equal weight. AP ranks RGB errors, with prevalence averaged over identical defined views. At 20% coverage each method accepts the same per-view quota within `within_new`. The last two columns use each released threshold on its own complete application domain.

| Sources | Method | Common AP | Same-domain prevalence | Error at 20% | Low / V | Errors / low |
|---|---|---:|---:|---:|---:|---:|
| M2 | B-L1-transfer | 0.7750 | 64.69% | 48.75% | 24.42% | 46.87% |
| M2 | MS-L1 | 0.8185 | 64.69% | 42.12% | 27.40% | 41.41% |
| M2 | MS-frozen | 0.8126 | 64.69% | 41.81% | 20.90% | 41.16% |
| M2 | MS-ft | 0.8172 | 64.69% | 40.94% | 24.49% | 41.55% |
| M4 | B-L1-transfer | 0.5256 | 38.63% | 21.68% | 28.83% | 23.02% |
| M4 | MS-L1 | 0.6052 | 38.63% | 21.05% | 41.70% | 22.83% |
| M4 | MS-frozen | 0.6073 | 38.63% | 18.60% | 29.71% | 21.18% |
| M4 | MS-ft | 0.6193 | 38.63% | 16.94% | 37.93% | 21.54% |

## Counts at matched 20% coverage

Each cell gives errors among accepted pixels at the same per-view quota. The candidate population, its label counts, attainable views, cutoff tie sizes and tie bounds for every coverage are in the per-view table.

| Subject | Sources | B-L1-transfer | MS-L1 | MS-frozen | MS-ft |
|---|---|---:|---:|---:|---:|
| set2 | M2 | 41320/78642 | 31782/78642 | 32812/78642 | 32065/78642 |
| set2 | M4 | 28447/78642 | 23777/78642 | 22379/78642 | 21507/78642 |
| set4 | M2 | 37219/78642 | 35939/78642 | 34711/78642 | 34006/78642 |
| set4 | M4 | 21814/78642 | 21309/78642 | 21073/78642 | 20364/78642 |
| set10 | M2 | 33558/78642 | 21977/78642 | 20671/78642 | 19547/78642 |
| set10 | M4 | 10393/78642 | 6176/78642 | 4083/78642 | 3726/78642 |
| set11 | M2 | 69518/78642 | 69151/78642 | 67841/78642 | 67751/78642 |
| set11 | M4 | 14649/78642 | 19734/78642 | 13672/78642 | 12505/78642 |
| set12 | M2 | 22575/78642 | 19651/78642 | 19491/78642 | 18649/78642 |
| set12 | M4 | 11469/78642 | 17468/78642 | 15575/78642 | 12536/78642 |
| set31 | M2 | 25849/78642 | 20224/78642 | 21757/78642 | 21135/78642 |
| set31 | M4 | 15503/78642 | 10858/78642 | 10970/78642 | 9294/78642 |

These six physical subjects share the calibrated rig, lighting, crop and camera layout. They provide no measurement on a new rig or an arbitrary reconstruction asset. No independently verified object mask was available. Original thresholds, weights and score support were fixed before access to target photographs. Both the change from authored scenes to real photographs and the change in reconstruction recipe affect transfer.

[Per-view counts and mask hashes](method-views.csv) · [Per-object results](method-conditions.csv) · [All matched quotas](matched-conditions.csv) · [Per-view cutoff and tie details](matched-views.csv) · [Object-equal results](object-equal.csv) · [Unknown reasons](unknown-reasons.csv) · [Source fits](source-fit.csv) · [Scheduled and successful entries](system.csv) · [Evaluation and reference-access receipt](evaluation.json)
