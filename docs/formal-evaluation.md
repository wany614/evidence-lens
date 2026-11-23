# Fixed held-out comparison and private report

P6 compares five frozen risk maps for each of four held-out families under B0, R1 and Q1. Its inputs include private Blender answers. The deployable L1 scorer runs first with the condition's source image, cameras, predictions and immutable model package. This tool consumes the independently produced scores.

## Before held-out scoring

Run the following in the project checkout with the recorder environment. Save the emitted digest and its full path-to-SHA map alongside the P5 method freeze. The canonical digest is SHA-256 of a UTF-8 JSON object mapping paths to file digests, sorted by path with compact separators and no trailing newline. The listing binds the study, analysis addendum, operating-point clarification, comparison logic, reporting code and imported numerical helpers.

```sh
PYTHONPATH=src .venv/bin/python scripts/report_study.py protocol
```

The formal B0 reference and independent readback can also be independently checked against both expected recorder-resolved configurations before a family is used. This reads only the exact selected B0 and readback run metadata and native camera. For tabletop it uses the two original pre-P3 seals and their respective historical configuration hashes.

```sh
PYTHONPATH=src .venv/bin/python scripts/verify_reference_config.py \
  --store .research --selection "$formal_reference_selection"
```

## One recorded comparison

After the real P5 method freeze, invoke the following for a named canonical quality selection and its independent L1 inference run. The environment variables in these examples stand for explicitly selected local files and sealed run pins. The selection command writes a new file.

```sh
PYTHONPATH=src .venv/bin/python scripts/report_study.py select \
  --quality "$quality_selection" --learned "$l1_run_id=$l1_manifest_sha256" \
  --method-freeze "$method_freeze" --out "$comparison_selection"

PYTHONPATH=src .venv/bin/python scripts/report_study.py record \
  --store .research --selection "$comparison_selection"
```

For Q1, also pass `--paired-quality "$same_family_B0_selection" --paired-learned "$B0_l1_run_id=$B0_l1_manifest_sha256"` to `select`. The configuration records trial, family, condition, study model, method freeze and comparison version. The recorder pins the individual parent manifests and binds each applicable receipt directly to the parent artifact. The structured bundle associates each original numeric file with its source role, parent-relative path, exact bytes and SHA-256. `record` accepts an explicit `--retry-of` and `--storage-policy`. Each attempt has a new recorder run ID.

The two lower-level entry points permit other orchestrators to stage exactly the same inputs.

```sh
PYTHONPATH=src .venv/bin/python scripts/report_study.py stage \
  --store .research --selection "$comparison_selection" --out "$empty_staging_directory"

PYTHONPATH=src .venv/bin/python -m evidence_lens.formal_evaluation \
  --bundle "$empty_staging_directory" --output "$empty_output_directory" --run-id "$run_id"
```

A manual selection uses schema `el-formal-comparison-selection-1`, kind `heldout_study`, `scene_family_id`, `condition_id`, explicit `parents`, `source`, `cameras` and `protocol_files`. Each role in `parents` has `run_id` and `manifest_sha256`. Normal parents are `reference`, `prediction`, `rules`, `labels`, `learned`, `model`, `readback` and `generation`. R1 and Q1 also supply `preparation`. Q1 adds `paired_prediction`, `paired_rules` and `paired_learned` from the same family's selected B0. Source and camera entries specify `parent_role` and a sealed run-relative `path`, with an optional explicit `sha256`. Each entry in `protocol_files` for `catalog`, `study` and `method_freeze` has `path` and `sha256`. Supply `quality_selection` and, for Q1, `paired_quality_selection` with these same two fields to bind the canonical scientific selections. Relative paths resolve against the selection file.

For a whole missing render, supply the actual failed predictor run, `render_failure_reason`, the model and the P2 evaluation made with that reason and the six original references. Rules and deployed scores are absent for that failed attempt. Each known full V is retained. If a matched B0 render is wholly unavailable, use `paired_render_failure_reason` and its failed B0 predictor pin. A per-view failure comes from the selected prediction, scoring and label receipts.

## Outputs and units

The P6 payload contains the following private records.

| File | Meaning |
| --- | --- |
| `formal_evaluation.json` | Parent seals, method and source hashes, every planned target's status, all numerator/denominator counts, missing reasons, scoreable masks' digests, common and own AP, pooled family system metrics, signed depth quantiles and paired diagnostics. |
| `curve_schema.json` and `curves/{v01,…,v06,family}.npz` | Complete attainable points for every method. Each method prefix indexes `score`, `accepted`, `errors`, `V`, `frozen_tau` and `frozen_index`. Coverage is `accepted/V` and risk is `errors/accepted`. The accept-none row has undefined risk. Ties remain intact. |
| `reason_visibility.json` | L0 rule reasons crossed with independent Blender visibility, including unknowns, failed outputs and row/column totals for each target and family. |
| `board/board_v01.png`, native panels and `board_v01.json` | Preselected v01 for every family and condition. Fixed error and risk scales, actual source, image prediction, error outcome, L0, continuous L1-full/geometry and physical visibility. A failed prediction keeps its assigned panel. |

Reference-valid RGB, including valid background, defines V. E adds finite prediction RGB. True error Y on E uses strict per-pixel RGB channel-mean absolute difference greater than 0.05 in display sRGB. Each method's low, high and unknown counts partition E. Unknown errors remain in the detection denominator. The primary R1 outcome is confirmed strict errors divided by all V. The ordinary true error incidence divides by E. Empty denominators produce JSON null and a reason.

Common-mask AP uses the five-way intersection with E, the same two-class target set and whole score ties. Own-mask AP is separate. System ratios pool numerator and denominator within family, and families receive equal weight. AP averages the eligible targets within a family before averaging eligible families. The family risk curve recombines target score groups at common thresholds across all six targets. Continuous signed depth summaries describe the supported render-depth population.

Q1 and B0 use the same condition input and cameras. The comparison verifies the selected B0 generation and Q1 derivation, both PLY byte digests, the recorder input digests and the PLY each renderer actually consumed. The paired RGB response uses jointly finite predicted RGB at the same target. Q1 error against Blender is also reported. The bitwise comparison covers native protected PLY attributes, alpha, expected depth, contribution and eligibility masks, geometry features, L0 and its reason, 1-alpha and the geometry-model score. It reports counts of unequal values, finite-mask disagreements, maximum absolute difference and the number of numerical differences above 1e-6. It measures the full model's response separately. The source diagnostic applies the same geometry-feature and L0 functions from the source camera to itself on each native self-render. Only the six registered targets enter the quality metrics.

## Reproducible final report

Create an explicit selection with schema `el-formal-report-selection-1`, kind `heldout_study`, the three `protocol_files` pins, `inventory_scope`, `trials` and `evidence`. `trials` must explicitly name the four frozen test families under all three conditions. Each row has `scene_family_id`, `condition_id`, `run_id` and `manifest_sha256` for its P6 result. A completed P6 bookkeeping run can retain a scientific condition whose renderer failed.

Each `evidence` entry has one unique sealed `run_id`, its `manifest_sha256` and a `purpose` of `scientific`, `timing_repeat`, `engineering`, `transfer`, `archive`, `recovery` or `training`. Include the actual model fit and threshold selection IDs, selected generation/render/scoring runs, retries, timing repeats, transfer, archive and recovery runs. Optional `startup` and `timing_index` describe cold/warm and repeated runs. The aggregator can include explicitly named compact documents with `documents` mapping an output basename to the selected run-relative file. Standard fit, CV, model, generation, prediction and score receipts are picked from these exact pins. The declared inventory scope appears verbatim in the report. Gaps appear as unavailable measurements.

```sh
PYTHONPATH=src .venv/bin/python scripts/report_study.py assemble \
  --store .research --selection "$report_selection" --out "$report_input_directory"

PYTHONPATH=src .venv/bin/python scripts/report_study.py build \
  --bundle "$report_input_directory" --output "$report_output_directory" --run-id "$report_run_id"
```

For a recorded `report` stage, bind the staged directory as input `report`, pass the explicit sealed result parents and execute `{python} {source}/scripts/report_study.py build --bundle {input:report} --output {out} --run-id {run_id}`. The payload includes Markdown, all 12 fixed boards, three condition-specific 4-family coverage figures, a paired outcomes figure, exact underlying numerical data, raw and pooled tables and a per-figure source-hash manifest. A plotted coverage point always corresponds to an actual whole-tie endpoint. An omitted accept-none y-coordinate has zero accepted pixels. The complete numeric curves preserve every attainable point at full precision.

The synthetic matrices and production-branch unit fixtures live in `tests/test_formal_evaluation.py` and `tests/test_formal_freeze_guards.py`. They exercise two-class AP, tied scores, a missing camera, an entirely failed condition, family-equal reduction, whole-pixel RGB errors, geometry equality, figures, a four-by-three report, truncated selections and tamper detection. The R1 registration tests change one 8-bit channel value or one camera parameter by half a pixel. The Q1 tests substitute an original generation, an original PLY or a renderer's recorded geometry parent. All test inputs are local generated fixtures.

### Expand the complete run roster

The `roster` command resolves the scientific and engineering pins below into the input for `assemble`. It includes every attempt inside each explicitly named scientific or timing selection. It verifies that timing trials identify the preselected warm-up and remain separate from the scientific comparisons. The canonical source/baseline chains, model fit and threshold selection remain individually pinned.

```sh
PYTHONPATH=src .venv/bin/python scripts/report_study.py roster \
  --store .research --roster "$study_roster" --out "$report_selection"
```

| Field in `el-study-report-roster-1` | Contents |
| --- | --- |
| `kind`, `protocol_files`, `inventory_scope` | Same fields as the explicit report selection above. |
| `trials` | Twelve objects containing `scene_family_id`, `condition_id`, `comparison` as a run pin and `quality` as that condition’s canonical scientific selection file pin. A P6 record can retain a totally failed render without a selected successful chain. |
| `scientific_markers` | Canonical science selection file pins for training and development conditions. The held-out selections already appear in `trials`. |
| `model_fit`, `threshold_select` | Each is a run pin or a file pin to its `el-scientific-stage-selection-1` marker. The P5 model must descend from the selected fit. |
| `timing_markers` | Nine Windows `el-timing-repeat-1` file pins. Each expands to the individual generation or derivation, render and retry attempts and retains its post-warm-up repeat number. |
| `cpu_timing` | Nine standalone CPU timing markers, or explicit model-score run pins. The SEALED configuration supplies the repeat index and quality-exclusion flag. |
| `evidence` | Extra sealed `run_id`, `manifest_sha256` and `purpose` entries, e.g. a model parity, transfer or independent recovery audit. |
| `proofs` | Independently sourced file pins with `id`, `kind` and `measures`. An independent cloud/blank-store receipt can live outside the active research store. |

A file pin is `{"path":"…","sha256":"…"}`. A run pin is `{"run_id":"…","manifest_sha256":"…"}`. Relative file pins in a roster resolve against that roster's directory. The cloud proof can select `{"verified_runs":"/index/runs","index_receipts":"/index/receipts","logical_files":"/logical_files","container_bytes":"/container_bytes","download_seconds":"/download_seconds"}`. The independent replay proof can select `{"compared_files":"/compared_files"}`. Each value comes from the actual hash-verified JSON. An array length can be selected with `{"pointer":"/items","op":"length"}`. The proof’s original bytes and selectors stay in the private report. Include standalone transfer/archive or storage receipts using this same interface when their measurements reside outside recorded runs.

The `public-numerical/` output contains family-level measurements, condition changes, stage times, selected numerical evidence, selection IDs and digests. Its exporter uses a fixed field allowlist. The private research archive retains the original images and geometric arrays. The sealed report identifies their producer runs and includes proofs and fixed visual boards for owner review. Each released numerical file has a material class and checksum in the matching release manifest. [Asset notices and provenance](licenses.md)
