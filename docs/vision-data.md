# Stage B learning data and native evaluation

Six independent target renders per scene supply the training views. The primary error label is the display-sRGB three-channel mean absolute error strictly greater than 0.05. The data worker reproduces the original pixel labels from FP32 reference and predicted RGB, camera-Z and the other recorded arrays, and checks them against the sealed labels. Source self-rendering contributes correspondence features. It contributes no additional supervised view.

## Learning cohorts and commands

The original 16 B0 training and four B0 development families have separately sealed material and transport indices. The transport records bind the exact original selection, the six views’ files and the older learning extraction. The registered 12 new training and four new development families have separate trial selections. The first four new training families form the engineering pilot. Eight additional confirmation families have a separate opening ceremony and scoring entry point. Four previously reported test families belong to historical regression.

Run from the research checkout with its experiment Python environment. The argument `--project` defaults to the current directory, `--store` to `.research`, and the cold shelf to `cold-shelf` next to the resolved store. When operating from a staged source bundle, substitute that bundle’s script path.

```sh
PYTHONPATH=src .venv-train/bin/python scripts/vision_pipeline.py pilot
PYTHONPATH=src .venv-train/bin/python scripts/vision_pipeline.py full --budget "$BUDGET_RUN_ID=$BUDGET_MANIFEST_SHA256"
```

The pilot command waits for each named scientific selection to exist before allocation. It records the trial registry, material index, four-family transport, and pilot-only preprocessing. The full command requires all registered learning selections and the sealed pilot budget decision. It joins the checked original and new splits, fits the two logistic regressions on the 28-family training sample, emits five development score maps per native target and freezes their operating points. Each stage stores its selected run ID, measured resources and content digest. Running the command again recovers its original selection. The split records live under `STORE/stage2/CATALOG_SHA/learning/STAGE/selected.json`. An unselected stage reports its missing named dependency. A genuine failed recorder attempt retains its ID and can be explicitly retried with `--retry STAGE=FAILED_RUN_ID` when the recorder admits a scientific retry.

The formal budget receipt is `payload/budget.json`. Its `el-stage2-pilot-budget-1` document gives the new catalog SHA, the four registered pilot families, the prior pilot dataset SHA, nonempty actual measurements, status `approved`, and decision `proceed`. The root controller supplies the sealed run pin. Transport failures pending repair have their own engineering records. A scientific terminal failure requires an explicitly selected failed run and a verified valid reference.

## Data for the Windows worker

Read the requested split's signed `dataset.json` (`el-vision-dataset-1`). It lists its exact `columns`, `feature_schema_digest` and every target under `views`. Each row has `family`, `condition`, `view`, `height`, `width`, `cv_fold`, `status`, `reason`, `rgb`, `samples` and `full`. File records contain `path` (portable split-relative), `sha256` and `bytes`. Use the array's native row-major, top-left coordinates. The DINO encoder alone may resize the full predicted RGB. The official target and every risk/evaluation array retain the original 512 × 512 coordinates.

| File | Contents |
| --- | --- |
| `rgb.npy` | Complete condition-specific target **prediction**. Shape H × W × 3, float32, display-sRGB. RGB values retain their original binary32 values and invalidity. |
| `samples.npz` | `index`, int32[N], strictly increasing and unique. `X`, float32[N,26]. `Y`, uint8[N], valid binary labels. Flattened index equals row × width + column. |
| `full.deploy` | Development/confirmation NPZ with exactly `X` float32[H,W,26], `scoreable` bool[H,W], `l0` and `alpha_score` float32[H,W]. Inference reads the RGB and this file. A worker can load one view and process bounded native row blocks. |
| `full.truth` | Separate sealed evaluation NPZ. `reference_valid` and `rgb_evaluable` are bool masks V and E. `rgb_high_error` is uint8 with 0 or 1 inside E and 255 elsewhere. Positive labels are obtained with `== 1`. Also preserves continuous RGB error, camera-Z support/error, edge and visibility. On a scientific terminal failure its masks are freshly derived from the pinned original reference and the recorded absence of predictions. |

Training rows have `full=null`. A failed view retains its slot, known V and failure reason, has no RGB/deployment asset, E=0 and no supervised rows. Report the acquisition stage and the count of invalid outputs separately. A complete numeric risk array encodes an unsupported element as NaN. Finite model scores lie in [0,1].

Eleven numeric columns precede four frame one-hot flags and eleven missing-numeric flags. The authoritative names and order are `evidence_lens.l1.ALL_COLUMNS`. IEEE NaN plus an explicit missing flag preserves unmeasured per-feature values. The train-only `preprocessing.json` (`el-vision-preprocessing-1`) gives the exact 11 medians, weighted means and scales, the ordered columns and the training dataset SHA. Impute and normalize the numeric columns in float64, preserve the flag values, and cast the resulting 26 columns to float32 for the neural head. Both DINO branches use these same statistics. LR refits preprocessing inside each training fold.

Within each registered family, condition and view, uniformly sample at most 4,096 members of E intersected with native deployable scoreability, without replacement. B0 reuses the exact original `9184\nCATALOG_SHA\nFAMILY\nVIEW` seed input, with SHA-256's first 16 bytes interpreted big-endian as the PCG64 seed. R1 and Q1 include the condition before the view in that final component. The same indices and features are used for both LR variants, both DINO branches and all their seeds. The extractor records inclusion probability and population counts per view. Each supervised family gets equal total weight, divided equally among its supervised views and then among each view's retained pixels. The constant baseline uses each training family's full E prevalence followed by an equal-family mean.

## Full-field comparison and threshold selection

A neural `scores.json` has schema `el-vision-scores-1`, its `method`, `model_sha256` (bytes of that model package's `model.json`), `dataset_sha256`, and all planned `views`. Row fields are `family`, `condition`, `view`, `status`, `reason`, `rgb_sha256`, `deploy_sha256` and `score`. Scored per-view NPZ files contain native `risk`, `valid`, and frozen `state` 0/1/255. Missing rows require a recorded `render_failed` or `unsupported` disposition with reason. The original five share one package and a per-view NPZ with `constant`, `alpha_score`, `l0`, `l1_full`, and `l1_geometry`.

After both selected seed-A neural development outputs arrive, run the independent all-seven development freeze against their three explicit score index pins. These examples use shell variables containing the individual paths and file hashes. The private recorder can invoke the same workers with a fresh output target and a genuine run ID.

```sh
PYTHONPATH=src python scripts/vision_evaluate.py freeze-dev \
  --dataset "$DEV" --sha256 "$DEV_SHA" \
  --scores "$ORIGINAL_SCORES" "$ORIGINAL_SHA" --scores "$FROZEN_SCORES" "$FROZEN_SHA" \
  --scores "$FT_SCORES" "$FT_SHA" --baseline-freeze "$FIVE_THRESHOLDS" "$FIVE_THRESHOLDS_SHA" \
  --catalog "$ORIGINAL_CATALOG" "$ORIGINAL_CATALOG_SHA" --catalog "$B_CATALOG" "$B_CATALOG_SHA" \
  --run-id "$THRESHOLD_RUN_ID" --output "$NEW_THRESHOLDS"

PYTHONPATH=src python scripts/vision_evaluate.py evaluate \
  --split development --dataset "$DEV" --sha256 "$DEV_SHA" \
  --scores "$ORIGINAL_SCORES" "$ORIGINAL_SHA" --scores "$FROZEN_SCORES" "$FROZEN_SHA" \
  --scores "$FT_SCORES" "$FT_SHA" --thresholds "$SEVEN_THRESHOLDS" "$SEVEN_THRESHOLDS_SHA" \
  --catalog "$ORIGINAL_CATALOG" "$ORIGINAL_CATALOG_SHA" --catalog "$B_CATALOG" "$B_CATALOG_SHA" \
  --output "$NEW_EVALUATION"
```

Development B0 thresholds use each method's own supported pixels inside E. They select the attainable family-mean L/V nearest to 0.8, keeping whole equal-score groups and selecting lower coverage on exact ties. The original L0 binary rule remains fixed at 0.5. Low is a finite score strictly less than the threshold. The all-seven freeze can explicitly recheck that the five original thresholds remain identical. Production thresholds and selected models are subsequently included in the signed method-freeze report.

Per-view reports include all system numerators, denominators, missing reasons, sensitivity labels at 0.025 and 0.10, and depth summaries. System rates pool native counts within each family before an equal-family average. AP first evaluates each target with whole-score ties, then averages eligible target APs inside each family, then averages eligible families. A view containing a single class has an undefined AP with reason. The five-method common mask and seven-method common mask have separate AP, size, coverage and positive prevalence. Each method also has its own supported-domain AP. The paired comparison file lists the exact matched family results. Risk–coverage plots can use the native family curve points, chosen from whole score groups at 5% coverage intervals plus the frozen point. Signed native score and truth files permit reconstruction at every distinct score. `case_index.json` pins v01 for each planned family and condition.

For confirmation the dedicated scorer and evaluator take `--opening STORE COLD_SHELF` and the single preregistered new catalog. They authenticate the one opening file, the original successful experiment report, model package members, training receipt lineage and frozen values before consuming a test data index. Confirmation uses all eight new families under B0/R1/Q1. Original test material is refused by the learning and confirmation source worker.
