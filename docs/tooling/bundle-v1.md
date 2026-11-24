# Evidence Bundle v1

A bundle is a directory with an integrity-checked `bundle.json`, `cameras.json`, `producer.json`, source photographs and numerical renders. A producer preparing data from its own reconstruction may write this contract directly. Each asset record is exactly `{"path":"relative/member","bytes":INTEGER,"sha256":"64 lowercase hex"}`. Hash the actual bytes, including the exact JSON encoding. Numeric values in the schema below describe the saved image grid.

## Bundle manifest

| Field | Contract |
| --- | --- |
| `schema` | `el-evidence-bundle-1` |
| `creation` | `run_id`, `implementation`, `code_sha256`, `config_sha256`, and `original_case_sha256` (null for native production). An authenticated historical supplement may add `legacy_proof_sha256`. |
| `input_domain` | Human-readable description of the acquired or generated inputs. |
| `condition` | `M2` with exactly two sources, or `M4` with exactly four. |
| `source_order`, `primary_source_id`, `endpoint_source_ids` | Explicit ordered source IDs, one member as the primary, and two distinct common endpoints drawn from those sources. All IDs are nonempty printable strings. Declare decimal IDs such as `"24"` and `"7"` as strings. Filesystem or lexical order does not select a source. |
| `target_order` | Nonempty ordered list of unique target IDs disjoint from the sources. IDs are metadata and never interpreted as paths. A failed target retains its ID and camera. |
| `camera`, `producer` | Asset records for `cameras.json` and `producer.json`. |
| `sources` | Each authorized ID maps to the corresponding RGB8 PNG asset under `sources/`. Different roles have different paths. |
| `geometry` | `prepared_camera_and_z_unit="m"`, positive `meters_per_original_unit`, `metric_evidence`, `crop`, `resize`, `undistortion`, and `scoring_resampling="none"`. Describe the recorded upstream conversion. Both camera translations and saved expected Z must already be in metres. The scorer never infers or reapplies the scale. |
| `risk_profile` | `id="v0.3.0-rc.1/original-thresholds"`, `sha256=99587dd70b67a84debb81fa1dc84e93729c8980bc4531edecfb56112a15eb191`, `release_index_sha256=150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d`. |

The frozen profile identifies the feature schema, model, preprocessing and original threshold for each of the four methods. The checkpoint and prior profile specify each upstream identity separately. The endpoint distance reported by the scorer is the Euclidean distance between the declared camera centres in metres. It is descriptive and introduces no extra numerical normalization. [Machine-readable profile](../../src/evidence_lens_bundle/assets/profile.json)

## Camera grid

`cameras.json` has schema `el-camera-1`, `axis_convention="OpenCV x right, y down, z forward"`, `image_origin="top_left"`, `pixel_center="integer"`, `unit="m"` and a `views` map containing exactly the source and target IDs. Each camera supplies integer `width` and `height` (both 256), finite 3×3 pinhole `K`, finite rigid right-handed 4×4 camera-to-world `c2w`, `projection="pinhole"` and `distortion="none"`. Optional `w2c` must invert `c2w`; optional `K_edge` must agree with the declared integer-centre intrinsics. Both transforms use column-vector camera/world conventions. Apply lens correction, frame conversion, crop, resize and calibration updates before writing the bundle.

## Renderer attestation

`producer.json` has schema `el-render-evidence-1`. It records `generator` and `renderer`, each with its actual `run_id` and original `receipt_sha256`. It also records:

- `checkpoint_sha256` for the single reconstruction used for all source self-renders and target predictions.
- `camera_sha256` for the distributed prepared camera document, `source_rgb_sha256` for every authorized source PNG, and the same `source_order` and `target_order` as the bundle. These identities bind the roster, inputs and arrays. Pin the top-level bundle digest from a trusted producer to detect replacement of the whole manifest.
- `semantics={"rgb":"display_srgb_0_1","z":"expected_camera_z_m","alpha":"accumulated_opacity"}`.
- Boolean `depth_verified` and `depth_probe_sha256`. A positive assertion requires a probe digest. An unverified value yields unknowns. The legacy recovery proof also authenticates and inspects the original probe.
- `frames` covering every source and target. A ready frame is `{"status":"ready","file":ASSET}` with its numerical `.npz` under `renders/`. A declared failed render is `{"status":"render_failed","reason":"..."}` and has no fabricated file.

A ready NPZ contains exactly float32 `rgb[H,W,3]`, `z[H,W]` and `alpha[H,W]`. Use NumPy row/column order. Valid RGB is display-sRGB in [0,1], positive depth is opacity-normalized expected camera Z in metres, and valid alpha is in [0,1]. Positive expected depth with zero opacity is rejected. Nonfinite predicted values can be represented and give explicit pixel-level reasons. A missing source self-render can remove evidence from one source while another still contributes. The original source photograph itself is required.

## Failure and access boundaries

- Malformed or missing inputs, hash mismatches, undeclared fields and contradictory geometry produce a structured `invalid` error and exit 2. Supported input syntax with an unsupported declared domain or model option produces `unsupported` and exit 3. The CLI writes a successful output atomically and refuses to overwrite an earlier result.
- Failed target renders retain their entire 256² canvas as unknown with an explicit render-failure reason. Individual reasons distinguish unverified depth, invalid target RGB/alpha/Z, projections outside the source frames, and absence of comparable source evidence. The single-source model follows its original eligibility rule.
- The deployment interface accepts no reference path. Put independent reference photographs, scans, validity masks and scientific error labels in a separate research package. The report consumes only its score receipt and declared output members. A `validate` call reads every declared deployment member. The reader does not discover extra photos by scanning neighbouring directories.
- A deployment receipt counts the actual target canvas. `low + high = scoreable` and `scoreable + unknown = canvas`. Reference-derived V/E, observed error rates, AP and MAE belong to independent evaluation; they cannot be read from this bundle alone.

The multisource extractor uses the original 39-column kernel in declared source order. The single-source transfer uses the released 26-column extractor on the primary source. The original Stage C Case path keeps its fixed M2/M4 IDs and sort rule. [Shared numerical implementation](../../src/evidence_lens/multisource.py)
