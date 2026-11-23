# CPU scoring and independent evaluation

The P2 CPU path scores six target views. It uses an exact source image, seven calibrated cameras and the producer's seven predicted renders. The independent evaluator accepts a validated Blender reference package and computes the actual RGB and rendered-depth errors. Both commands write into a new or empty output directory. The run recorder can pass its empty payload directory directly as the output.

## Score a prediction

```sh
python -m evidence_lens.risk_model \
  --input source/input.png --cameras source/cameras.json \
  --predictions render/prediction --output result/score
```

The input is the actual 8-bit RGB or fully opaque RGBA PNG for this condition. The scorer decodes its RGB code values to FP32 in [0,1]. It uses the source camera's own dimensions when reading the source self-render and the source image. The source and target resolutions may differ.

The camera JSON has schema `el-camera-1`, `unit="m"`, `pixel_center="integer"` and a `views` object keyed by `source`, then `v01` through `v06`. Each camera records positive integer width and height, 3×3 pinhole K and rigid 4×4 camera-to-world pose. Camera columns point right, down and forward. Rows increase downward. A pixel centre at array index (row v, column u) projects to integer (u,v). Sampling converts to edge coordinates by adding 0.5, checks the half-open field of view [0,W) × [0,H) and takes the floor. For a scale s with no crop, the integer-centre principal point becomes s(c+0.5)−0.5.

The prediction directory contains `prediction.json` and one `{view}.npz` per view. Every NPZ contains exactly three named FP32 arrays.

| Array | Shape | Meaning |
| --- | --- | --- |
| `rgb` | H×W×3 | Display sRGB in [0,1] for each finite channel, in RGB order. Any non-finite channel marks the pixel RGB invalid. |
| `z` | H×W | Expected positive camera-axis Z in metres. Non-finite or nonpositive values mark unavailable depth. Alpha zero requires unavailable depth. |
| `alpha` | H×W | Accumulated opacity in [0,1]. A non-finite or out-of-range value marks invalid contribution. |

The producer metadata has the following fields.

| Field | Contract |
| --- | --- |
| `schema`, `producer_run_id` | `el-prediction-1` and a nonempty producer ID |
| `input_sha256`, `camera_sha256` | SHA-256 of the exact input PNG and camera JSON bytes |
| `rgb_encoding`, `depth_semantics`, `alpha_semantics` | `display_srgb_0_1`, `expected_camera_z_m`, `accumulated_opacity` |
| `depth_validation` | Two Boolean results, `off_axis_plane_passed` and `alpha_normalization_passed`, from the actual provider's calibration |
| `depth_validation_receipt_sha256` | Optional SHA-256 of the provider's full calibration report. The report lives in the producer archive. |
| `views` | Optional map of view IDs to objects with `sha256`. When supplied, each available NPZ must match the producer digest. |
| `failed_views` | Optional map of intentionally missing view IDs to their actual failure reasons. The corresponding files must be absent. An undeclared missing view fails the input check. |

Each target produces `v01.npz` through `v06.npz` when available. `l0` and `alpha_score` are FP32 maps, with NaN for unknown. Each map has its own Boolean validity mask. For L0, 0 means low risk and 1 means high risk. The diagnostic 1−alpha operating point flags score >0.5. This is the P2 operating point for the built-in baseline. The map's numerical values retain its full ranking.

The raw L0 diagnostic can flag low contribution or a geometry conflict at a pixel with invalid predicted RGB. The evaluator assigns risk decisions on E, where both images have valid RGB. Its board marks pixels outside E in magenta and unknown L0 decisions within E in gray.

The L0 reason codes follow their evaluation order. Target alpha below 0.5 gives a warning even when target depth is unavailable. An unavailable global calibration disables projected-depth comparisons for every pixel.

| Code | Rule outcome | Risk |
| --- | --- | --- |
| 1 | Invalid target alpha | Unknown |
| 2 | Low target contribution | High |
| 3, 4 | Unvalidated depth, or locally invalid target depth | Unknown |
| 5 | Projected representative point outside the input image or behind the source camera | High |
| 6, 7 | Invalid source alpha, or low source contribution | Unknown, or high |
| 8 | Invalid sampled source depth | Unknown |
| 9 | Representative point behind the source expected depth by >0.10 of projected source Z | High |
| 0 | Supported under the preceding rules | Low |

The numerical feature arrays use FP32, NaN for missing and an explicit Boolean missing mask per feature. `feature_schema.json` gives the full fixed column order, units, field-of-view one-hot flags and the eligibility mask for future model work. `c_behind` and `c_front` use their respective projected and source-render depth denominators. Local 3×3 features include the centre, truncate at the boundary and require at least three valid pixels. The source-image consistency feature requires both alphas ≥0.5 and both relative depth conflicts ≤0.10. The source self-render residual uses any valid projected RGB correspondence.

`score.json` records the precise input byte hashes, per-view score hashes, L0 reason dictionary and elapsed times for reading each target, feature computation, scoring and saving. The recorder separately measures process resources. The source self-render diagnostic includes the input/self-render comparison PNG, numeric residual and finite-RGB count.

## Evaluate with the Blender reference

Run from the same condition and camera files used by the scorer. The following paths and IDs represent caller-supplied run assets.

```sh
python -m evidence_lens.evaluation \
  --input source/input.png --cameras source/cameras.json \
  --predictions render/prediction --scores result/score \
  --references reference-run/payload --output result/evaluation \
  --run-id "$EVAL_RUN_ID" --trial-id "$TRIAL_ID" \
  --scene-family-id scene_001_tabletop --scene-id scene_001_tabletop/v001 \
  --condition-id B0 --parquet
```

`--references` points to the directory containing `reference.json`, `validation/results.json` and `reference/v01.npz` through `reference/v06.npz`. The evaluator checks the reference validation receipt, image and camera hashes, scene identity, RGB and depth semantics, and individual file hashes. It checks that world Position projects back to the image within one pixel and agrees with camera-Z within 0.1 mm plus a relative tolerance of 0.0001. Predictions and scores must share the same source image, cameras, producer metadata and per-view data bytes.

The reference stores display-sRGB FP32 `rgb`, metric FP32 camera `z`, metric FP32 world `position`, integer `object_id`, uint8 `visibility` and binary `edge`. Visibility codes 0, 1, 2 and 3 mean background/invalid, observed, disoccluded and outside the source field of view. The actual-error labels depend on corresponding numerical image and depth differences. Visibility and edges describe regions for interpretation.

| Output | Interpretation |
| --- | --- |
| `labels/{view}.npz` | Continuous FP32 per-pixel RGB MAE, absolute/signed/relative depth residuals, independent labels and masks. Unavailable residuals are NaN. Binary label arrays use 0/1 and 255 for unavailable. |
| `metrics.jsonl`, `metrics.csv` | Per-view, per-region raw numerator, denominator, value, unit, mask digest, status/reason and source artifact digests. The optional `--parquet` switch also writes the typed recorder schema using the `record` install extra. |
| `curves/{view}.npz` | Each method's attainable accepted-pixel count, error count, coverage relative to V and error rate. Thresholds enter in ascending complete equal-score groups. The zero-acceptance point has undefined error rate. |
| `reason_visibility_counts.json` | Cross-tabulation of L0 reasons and true visibility on valid reference RGB, with row and column totals per view and for the family. |
| `board_v01.png`, `board_v01.json` | The preselected first target, actual source, reference, prediction, RGB error, L0 and true visibility. Error colors use the same 0–0.20 scale across images. The manifest binds the board to its numerical sources. |
| `v01_l0_state.png`, `v01_rgb_error.png` | Full-resolution, indexed PNGs of the risk and error panels for an available v01 prediction. Risk indices 0, 1 and 2 represent low, high and unknown on E. Error indices 0–254 linearly quantize RGB MAE over 0–0.20 and saturate above 0.20. Both PNGs use 255 outside E. Their hashes appear in the board manifest. The label NPZ retains full FP32 error. |
| `evaluation.json` | Run IDs, input SHA-256 hashes, depth validation status, six planned views, actual completed views, per-view masks and rule-reason counts. |

V contains finite reference RGB, including valid background. E is the part of V with finite predicted RGB. An RGB error is the mean absolute difference of the three display-sRGB channels. The main positive label uses error >0.05. The predetermined sensitivity thresholds are 0.025 and 0.10. Rendered-depth evaluation additionally requires a true surface, verified positive predicted Z and valid alpha ≥0.5. Its threshold in metres is max(0.02, 0.02 × true Z).

| System metric | Numerator | Denominator |
| --- | --- | --- |
| Error area | Actual errors | E |
| High-risk recall | Errors in high-risk pixels | All errors in E, including unknown pixels |
| Warning precision | Errors in high-risk pixels | High-risk pixels in E |
| Low-risk error | Errors in accepted pixels | Low-risk pixels in E |
| Overall low-risk coverage | Low-risk pixels in E | V |
| Unknown fraction | Unknown pixels in E | E |
| Output failure | Missing or invalid predicted RGB in V | V |

Metrics retain empty regions. A zero denominator gives a null value and a reason. Medians and AP retain their sample count and use a null numerator. P2 common-set AP uses the intersection of eligible L0 and 1−alpha scores in E. A single-class intersection yields `ranking_not_evaluable`. Per-method AP uses a separate diagnostic field. Precision–recall and risk–coverage calculations preserve complete equal-score groups.

For a documented whole-render failure, replace the pair of `--predictions` and `--scores` arguments with `--render-failed "$REASON"`. With V known and nonempty, each affected view records zero low-risk coverage and unit output failure. Scientific-quality values with no measurements remain null. A device awaiting execution has a pending state in the project ledger.

The deterministic verification cases run via `python -m pytest -q tests/test_cpu_p2.py` after installing the test extra. Their tiny arrays live in temporary directories.
