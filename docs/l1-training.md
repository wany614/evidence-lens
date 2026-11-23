# Training, freezing and using L1

L1-full and L1-geometry estimate the risk that a novel-view prediction exceeds 0.05 mean absolute display-sRGB error on a pixel with valid reference and predicted RGB. A model package contains the two fitted logistic regressions and their fitted preprocessing parameters. A development-frozen package also contains the four continuous methods’ operating thresholds. The base CPU scorer reads the package, the condition’s actual source PNG, calibrated cameras and predictions. [CPU input and feature specification](cpu-scoring.md)

## Run the frozen model

Install the package with `python -m pip install .` in a Python 3.11+ environment. Use the SHA-256 of `model/model.json` supplied with the release or frozen experiment. The output directory must be new or empty.

```sh
python -m evidence_lens.l1 \
  --model "$FROZEN_PAYLOAD/model" --model-sha256 "$MODEL_INDEX_SHA256" \
  --input "$CONDITION_SOURCE_PNG" --cameras "$CONDITION_CAMERA_JSON" \
  --predictions "$CONDITION_PREDICTION_DIR" --output "$NEW_SCORES"
```

The prediction directory uses the P2 schema. The scorer opens files by explicit role and checks the source PNG and camera content against `prediction.json`. If present, numeric prediction hashes must match. Each model component must match its package hash and the extractor’s column order. The optional `--run-id` identifies the scoring run and otherwise defaults to `EL_RUN_ID`. A direct Python caller can use `evidence_lens.l1.load_package`, `score_view` or the file-based `run`.

| Method in the receipt | Array name in each target NPZ | Low decision |
| --- | --- | --- |
| L1-full | `l1_full` | score < its frozen threshold |
| L1-geometry | `l1_geometry` | score < its frozen threshold |
| constant | `constant` | training prevalence < its frozen threshold |
| 1-alpha | `alpha_score` | 1 − legal alpha < its frozen threshold |
| L0 | `l0` | fixed rule returns 0 |

Each `v01.npz` through `v06.npz` contains FP32 scores, Boolean `<array>_valid` and uint8 `<array>_state` fields. State values are 0 for low, 1 for high and 255 for unknown. An unknown score is NaN. Both L1 methods require finite target RGB, alpha in [0,1] and finite positive expected optical-axis Z in metres from a validated provider. Low alpha with valid Z remains scoreable. Local source unavailability is represented by missing features. Incompatible depth validation disables L1 and preserves independently available baseline diagnostics. A declared failed target gets all-unknown maps. Incompatible global encoding, depth convention or camera calibration produces an unsupported receipt. Unknown maps are written when calibrated dimensions are available. Hash mismatches and undeclared missing prediction files fail validation.

`l1_score.json` records the model ID, exact input and model-file hashes and sizes, prediction producer, threshold values, state codes, per-view output hashes and timing for read, feature extraction, scoring and output. The run recorder provides total wall time and peak process-tree memory. Scientific evaluation applies states on the independently determined evaluable set E and uses all reference-valid pixels V for system coverage.

## Prepare a selected training or development family

The recorder verifies the sealed B0 trial selection, creates the family pin and stages the selected original P2 score and evaluation inputs. The family pin binds both run IDs, both run-manifest hashes and both exact receipt-file hashes to the trial. The worker checks the frozen split catalog, recomputes the staged file digests and verifies the evaluator read the same scoring inputs and numerical maps. The hash of the frozen split catalog is over its original bytes.

```sh
PYTHONPATH=src python scripts/train_l1.py prepare \
  --family "$FAMILY_PIN" --catalog "$FROZEN_FAMILY_CATALOG" \
  --scores "$SELECTED_P2_SCORE_PAYLOAD" --evaluation "$SELECTED_P2_EVALUATION_PAYLOAD" \
  --output "$NEW_FAMILY_EXTRACT"
```

| Input | Fields or files |
| --- | --- |
| Family pin | `schema=el-learning-family-1`, `scene_family_id`, `scene_id`, `trial_id`, `condition_id=B0`, `split=train` or `dev`, train `cv_fold` from 0 through 3, `split_manifest_sha256`, `selection_sha256`, `score_run_id`, `score_manifest_sha256`, `score_receipt_sha256`, `evaluation_run_id`, `evaluation_manifest_sha256`, `evaluation_receipt_sha256` |
| Catalog | `el-formal-families-1` with frozen family, scene, trial, split and CV assignment |
| P2 score payload | `score.json`, `feature_schema.json`, each completed target’s `vNN.npz` |
| P2 evaluation payload | `evaluation.json`, `labels/v01.npz` through `labels/v06.npz` |

The extract’s `family.json` (schema `el-learning-extract-1`) retains the pin, source PNG digest, read audit, feature and label versions, and each target’s numeric file SHA-256, byte size, image shape, V, E, complete positives, eligible population, sampled row and positive counts, inclusion probability and seed. Each `vNN.npz` stores sorted row-major `index` (int64), ordered feature matrix `X` (FP32), `Y` (uint8), `scoreable` (Boolean) and `alpha_score` (FP32). Use `divmod(index, width)` to recover row and column. Failed views retain full reference denominators, zero evaluable rows and the original failure in the P2 receipts.

For each of the six training targets, the worker uniformly draws up to 4,096 eligible pixels from E without replacement, using NumPy PCG64. Its seed is the unsigned big-endian integer formed from the first 16 SHA-256 bytes of UTF-8 `9184\n{split_manifest_sha256}\n{scene_family_id}\n{camera_id}`, with LF separators and no final newline. If there are at most 4,096 eligible pixels, all are retained. Both models and all C candidates share these exact coordinates. Development retains every row of E including per-method unknowns. The source self-render provides features.

## Fit on the complete training split

The training environment uses the optional `train` dependencies. Each split directory contains `catalog.json`, `dataset.json` and the named family extracts. Its index has schema `el-learning-index-1`, `split`, `condition_id=B0`, `split_manifest_sha256`, and a `families` array of `{scene_family_id, extract_manifest_sha256}` records. The worker verifies all 16 train or all four development assignments using the metadata catalog before opening the corresponding feature files. Only the corresponding family directories are read.

```sh
python -m pip install '.[train]'
PYTHONPATH=src python scripts/train_l1.py fit --dataset "$TRAIN_SPLIT_DIR" \
  --run-id "$FIT_RUN_ID" --output "$NEW_FIT_PAYLOAD"
```

Each represented training family receives equal total weight. Each of its nonempty views receives equal total weight within that family. In a fold of N rows from F represented families, each row of view v in family f receives `N / (F * nonempty_views_f * rows_fv)`. Weights have mean 1. Each fold computes finite-value medians on its training rows, fills wholly missing numerical columns with 0, then uses row-weighted mean and population variance. Constant numerical columns have scale 1. Frame one-hot and missing flags remain binary. Geometry drops `local_rgb_difference`, `source_rgb_difference`, `source_render_residual` and their three missing flags.

The fit uses scikit-learn 1.9.1 L2 logistic regression with an intercept, `lbfgs`, `l1_ratio=0`, no class weighting, a 1,000-iteration limit and tolerance 1e-6. Each model tests C of 0.1, 1 and 10. The first attempt uses the catalog’s four family fold IDs. If a training fold lacks either class, sorted family names are assigned by ordinal modulo 3, then 2 as needed. Validation AP uses whole equal-score groups on the identical samples. Each view needs both classes. Effective views are averaged within family and effective families are averaged equally. An AP difference up to 1e-6 selects the smaller C. [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

`training_features.parquet` contains row/column, labels, features, camera and source IDs, inclusion probability and full-fit weight. `training_dataset_manifest.json` binds the training table, source runs and extracts, split/feature/label versions and actual CV folds. `cv_results.json` records every candidate, fold-local preprocessing, coefficients, iterations/convergence, per-view positive and negative counts and APs. `fit.json` reports actual class counts, selected-package hashes and maximum numerical difference after reloading the export and comparing against scikit-learn on every training row. The required maximum absolute error is 1e-6. `model_card.md` gives the measured training results for that fit. Insufficient training classes or undefined selection AP leaves a diagnostic receipt and CLI exit code 2.

## Freeze the development operating points

```sh
PYTHONPATH=src python scripts/train_l1.py freeze --dataset "$DEVELOPMENT_SPLIT_DIR" \
  --fit "$FIT_PAYLOAD/model" --run-id "$THRESHOLD_RUN_ID" --output "$NEW_FREEZE_PAYLOAD"
```

For each method, coverage is low pixels divided by all reference-valid pixels within each development family, then averaged over families with defined denominators. All six planned views contribute their V, including prediction failures. Candidate thresholds accept complete equal-score groups from lowest to highest, including accepting none and all scoreable pixels. Choose the attainable coverage nearest 0.80, resolving exact distance ties toward the lower coverage. Unavailable scores stay unknown. The constant baseline is the mean of the training families’ complete E error rates. It has only the none/all operating choices.

`development.json` records the fit identity, pinned input files, frozen model-index SHA-256, attained coverage, nearest candidates and family/view counts and errors for all four continuous methods. `development_scores/<family>/vNN.npz` retains each E pixel’s coordinate, Y and numerical scores for threshold recomputation. `model/model.json` identifies the frozen release and binds `feature_schema.json`, `L1-full.json`, `L1-geometry.json` and `thresholds.json` by SHA-256. Apply the same FP32 score and strict float64 threshold comparison to subsequent conditions. Formal results and resource measurements come from the recorded runs.

## Record the scientific runs

The controller uses the recorder environment for manifests and process-tree telemetry. Give `fit` the training interpreter and give `threshold` and `score` the base CPU interpreter. It materializes the complete named split with `formal_batch.dataset`. Each family’s numeric extract is a separately declared input tied to the corresponding sealed `dataset_build` parent. Development also binds the exact fitted model as a parent.

```sh
PYTHONPATH=src .venv/bin/python scripts/formal_learning.py fit \
  --store "$STORE" --worker-python .venv-train/bin/python
PYTHONPATH=src .venv/bin/python scripts/formal_learning.py threshold \
  --store "$STORE" --fit-run "$FIT_RUN_ID" --worker-python .venv-cpu/bin/python
PYTHONPATH=src .venv/bin/python scripts/formal_learning.py freeze-method \
  --store "$STORE" --threshold-run "$THRESHOLD_RUN_ID" --method-code-commit "$COMMIT"
PYTHONPATH=src .venv/bin/python scripts/formal_learning.py score \
  --store "$STORE" --family "$FAMILY" --condition "$CONDITION" --worker-python .venv-cpu/bin/python
```

The immutable selected fit and threshold receipts are `STORE/formal/<catalog-sha>/learning/fit.json` and `threshold.json`. When a caller is interrupted after the recorder seals a result, the same command recovers the matching stage. `--adopt-run` allows explicit recovery by run ID and `--retry-of` identifies an executed failure. A different scientific request cannot replace the first selected result.

`freeze-method` checks that the committed numerical and comparison source bytes match the registered study and the executed learning snapshots. It publishes `STORE/protocol/method-freeze-v1.json` with catalog, study, comparison, commit, model, fit and threshold hashes. Held-out `score` verifies the gate before opening the selected condition. R1 uses the selected half-size preparation’s source and calibration. B0 and Q1 use the selected reference’s exact source and camera. The scoring subprocess receives only the condition input, cameras, available predicted numeric arrays and frozen model. The learned-score selection is `STORE/formal/<catalog-sha>/learned-scores/<trial>.json`. The registered first test family also permits `--timing-repeat 1`, `2` or `3`. Timing selections are archived under `learned-scores/timing/<repeat>/`.

Run the six-view development deployment check after the method freeze. The registered audit uses the lexicographically first development family, `scene_017_cycle_stand`. It makes a new CPU deployment from the selected B0 source, cameras and predictions. A separate recorded worker compares all four continuous scores on E with their saved development arrays, verifies exact validity and frozen states, and checks L0 against the original P2 rule throughout the image. It reports V−E by view and the raw low/high/unknown counts there. The method operating counts use E and the complete V denominator.

```sh
PYTHONPATH=src .venv/bin/python scripts/verify_l1_development.py record \
  --store "$STORE" --threshold-run "$THRESHOLD_RUN_ID" --cpu-python .venv-cpu/bin/python
```

The audit writes a selected `archive_verify` pin to `STORE/formal/<catalog-sha>/learning/development-parity.json`. Its `payload/development_parity.json` identifies both original P2 runs, the frozen threshold run, the new CPU deployment and the exact model index. The public inference process checks content identity before semantic support, so wrong PNG or camera bytes fail even if they also use an unsupported encoding or camera convention.

To inspect V−E on a selected training or development B0 pilot, use `PYTHONPATH=src .venv/bin/python scripts/verify_l1_development.py pilot --store "$STORE" --family "$FAMILY" --output "$NEW_MASK_CHECK"`. The output records every view’s complete V, E, failed-RGB count and raw low/high/unknown L0 partition of V−E. Each count is tied to the selected source manifests.

## Grade a standalone release example

Stage the five model files under `BUNDLE/model/` and the condition’s deployable assets under `BUNDLE/example/input.png`, `BUNDLE/example/cameras.json` and `BUNDLE/example/predictions/`. Seal the expected result from a selected recorded five-method score receipt. The default graded view is v01. Repeat `--view` to include more targets.

```sh
python -m evidence_lens.l1_demo seal --bundle "$BUNDLE" --scores "$RECORDED_P6_SCORES" \
  --score-receipt-sha256 "$L1_SCORE_JSON_SHA256"
python -m evidence_lens.l1_demo grade --bundle "$BUNDLE" --manifest-sha256 "$DEMO_SHA256" \
  --python "$CLEAN_VENV/bin/python" --output "$NEW_GRADE"
```

`seal` checks every deployable input and model hash against the archived score, copies the selected original numerical score NPZs into `expected/` and returns the digest of `demo.json`. Install the release wheel and its NumPy/Pillow dependencies in the clean virtual environment before grading. The grader reads the expected results in its own process. It copies only deployable inputs into temporary storage, then launches the installed scorer with Python’s isolated mode. It compares all five FP32 scores within 1e-6, exact availability masks and state codes, and L0 reasons. `grade.json` lists counts and maximum differences by method and view, the original score run, the released model and the measured runtime package versions.
