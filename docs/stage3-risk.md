# Multi-source risk scoring

The primary experiment trains one shared model on all source images available to each reconstruction. M2 uses the two endpoint cameras. M4 adds two cameras between the same endpoints. Scoring modes hold each reconstructed scene and target prediction fixed. The one-source mode reads s00, the two-source mode reads s00 and s03, and the all-source mode reads the generator's complete source set. On M2, two and all refer to a single numerical result.

The published Stage B L1-full transfer uses s00 and its original 26 columns, coefficients and operating threshold (0.43483029305934906). Its score is identical in every mode for a fixed reconstruction. MS-L1, MS-frozen and MS-ft consume the 39 ordered columns in the protocol. Training used the all-source mode. The one-source ablation therefore evaluates extrapolation to an untrained source count.

## Native data and inference

The deployed case directory contains `case.json`, `prediction.json`, the source and render camera files, original source PNGs under `sources/`, and the reconstruction's source and six target arrays under `predictions/`. The camera uses OpenCV axes and integer pixel centres. Metres refer to expected optical-axis camera Z. The reader identifies each original asset by byte count and SHA-256. The input to the numerical extractor is one target prediction, its camera, and a mapping from allowed source ID to original RGB, source self-render and camera. Unverified depth or a target without a comparable source produces an unknown score. Comparable but inconsistent or low-opacity correspondences retain an explicit lack-of-support indicator.

For the two lightweight methods the CPU entry point writes `scores.npz` and `score.json`. The former stores FP32 risk, a boolean finite mask and uint8 state. State 0 means low, 1 means high and 255 means unknown.

```sh
python -m evidence_lens.multisource_model --case CASE --case-sha256 CASE_SHA --view v01 --scope all --method MS-L1 --package MODEL --model-sha256 MODEL_SHA --output NEW
```

For B-L1-transfer, provide the original model directory, its index SHA, the SHA of L1-full.json and its frozen threshold using `--weights-sha256` and `--tau`. The neural model package provides its own CPU score-native.py entry point.

## Recorded study

Run the controller with the training interpreter and `--project PROJECT --store STORE`. Each stage freezes its configuration before the existing recorder starts the worker. Data-pilot accepts only the first two engineering families and their original 64-update reconstructions. Data-train and data-development require the separate production budget, successful input readbacks, and each selected production condition's strengthened source audit. Confirmed scientific failures retain the scheduled reference and disposition.

```sh
python scripts/stage3_risk_control.py data-train --project PROJECT --store STORE
python scripts/stage3_risk_control.py data-development --project PROJECT --store STORE
python scripts/stage3_risk_control.py fit --project PROJECT --store STORE
python scripts/stage3_risk_control.py development --project PROJECT --store STORE
```

Each of the 8 training families contributes two reconstructions and six targets. Per native frame, all learned methods share up to 4,096 uniform scoreable, RGB-evaluable pixels, the saved sorted coordinates and their inclusion probability. Model fitting balances family, condition and view, estimates each fold's preprocessing from its training complement and groups all views of a family in the same fold. Separate development and test datasets retain full grids and independent error labels. Each learned method chooses one all-source development threshold. Equal scores move together; equal distances from 80% family- and condition-balanced low coverage favour the lower coverage.

After four formal neural fits, supply an explicit candidate roster with schema `el-stage3-risk-candidate-roster-1`, the registered catalog SHA and four `candidates`, each containing method, seed, run_id and manifest_sha256. `stage3_risk_postfit.py` records paired selection, the original development example, four standalone native CPU runs and device parity. The evaluation workers require the official opening SHA before opening a test dataset or resolving any test assets. The development and test numerical folders each contain original denominators, matched target comparisons, family-level AP and operating rates, risk–coverage points, separate reconstruction costs and an asset manifest. Full-resolution boards retain all six scheduled target views and all three source subsets for each family and condition.
