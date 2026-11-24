# Choosing two additional photographs on the DTU development subjects

On set1, risk weighting selected cameras 17 and 40 and reduced full-crop target RGB MAE by 0.000257 compared with uniform visible coverage. On set6 both selected 17 and 31 and used the same reconstruction. The fixed four-camera layout had the lowest MAE of the primary layouts on both subjects. These two objects had been exposed in the previous recovery study. The measurements describe development behavior on this rig. [Per-subject paired measurements](paired.csv) · [All 108 view measurements and support hashes](views.csv)

| Development subject | Original M2 | Fixed M4 | Uniform M4 | Risk M4 | Risk minus uniform | Three random M4 runs, mean [min, max] |
|---|---:|---:|---:|---:|---:|---|
| set1 | 0.145344 | 0.085526 | 0.098788 | 0.098531 | −0.000257 | 0.105713 [0.096969, 0.110795] |
| set6 | 0.138064 | 0.088012 | 0.097731 | 0.097731 | 0.000000 | 0.109067 [0.098747, 0.128261] |

Each cell uses the same six independent target photographs and 393,216 comparable pixels out of 393,216 reference-valid pixels. All nine logical branches have identical per-view pixel-mask hashes within each subject. The full center crop includes the background. The primary endpoint pools the mean absolute difference in the three display-sRGB channels. Its value on set1 favors risk slightly, although the number of pixels exceeding 0.05 rises from 150,662 to 152,286. [Branch counts and full-precision means](branches.csv) · [Evaluation receipt](evaluation.json)

## What each selection observed

Each selector started with cameras 24 and 22. It could inspect the eight candidate poses 7, 10, 13, 17, 20, 23, 31 and 40. Four deterministic virtual poses were derived by interpolation inside this geometry. Two source self-renders exactly reproduced their archived RGB, expected camera-Z and opacity. The remaining renders provided the four query frames for MS-L1 and depth/opacity for candidate visibility. Uniform and risk weighting shared the same valid 3D samples, visibility test, pair enumeration, tie rule and fallback. The six evaluation poses and all candidate photographs were absent from the selection inputs. [Selector and visibility protocol](../../../../protocol/b2-nbv-v1.json) · [Visibility and unknown counts](query-support.csv) · [Utility for every one of the 28 pairs per subject](pair-utilities.csv)

| Subject | Uniform | Risk-weighted | Random 137101 | Random 137102 | Random 137103 |
|---|---|---|---|---|---|
| set1 | 10, 23 | 17, 40 | 7, 17 | 17, 20 | 10, 13 |
| set6 | 17, 31 | 17, 31 | 7, 10 | 7, 23 | 20, 31 |

Both subjects used the ordinary selection rule. Each of the ten trajectories sealed two camera IDs before obtaining either photograph. Each maintained its own observation receipt even when two trajectories selected the same pair. Set6's risk trajectory reused the exact ordered source bytes, initialization, seed, checkpoints and renders of uniform. All other selected M4 cases started from their four authorized source images. Set1 and set6 decision runs were `20261124T015805706693Z-09f9884ef63d482e` and `20261124T015806770790Z-ac34f106b6ff471b`. The independent endpoint evaluation was `20261124T022032641624Z-d7da793c7f6b4dff`. [Ten separate access ledgers](observations.csv) · [Acquisition audit](access-audit.json)

## Seed sensitivity and cost

The fixed M4 sensitivity experiment changed only the optimizer seed. The historical 43171 run and the two new 43172/43173 runs share the input RGBs, initialization array, camera, numerical worker, solver recipe apart from its seed, and 5,000-step endpoint. The range across the three resulting MAEs was 0.005505 on set1 and 0.001643 on set6. This measures variation in the fixed-layout controls. It supplies no uncertainty interval for the selected-layout comparison. [Endpoint and seed comparisons](paired.csv) · [Exact reconstruction origins](../reconstruction/README.md)

On set1 and set6 the CUDA probe worker took 1.0143 and 1.1185 seconds, respectively, for each combined sweep of eight candidate poses, four query poses and two replay checks. Per-camera-role rendering times were not instrumented in the original pass. Mac query feature extraction and scoring took 0.1552 and 0.1514 seconds. Total Mac selection took 0.1986 and 0.1939 seconds, including that scoring time. Ten trajectory-local source openings transferred 57,981,229 HTTP response-body bytes. The thirteen distinct development reconstructions and their endpoint renders accumulated 504.469 and 72.421 recorder seconds, respectively. [Probe timings and GPU allocation](probe-costs.csv) · [Per-fit timing and transfer ledger](../reconstruction/development-branches.json)
