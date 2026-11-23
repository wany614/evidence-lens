# Module boundaries

| Module | Reads | Writes |
| --- | --- | --- |
| recorder | Stage configuration, explicitly declared inputs, pinned parent manifests, project sources and runtime counters | One private run with frozen launch context, byte hashes, timestamps, process resources, logs, outcomes and completion marker |
| blender_reference | The procedural world and camera configuration | Frozen scene, source PNG, camera records, reference RGB/Position/camera-Z, visibility, mask provenance and validation measurements |
| generator_adapter | One source RGB PNG, its source intrinsics, pinned SHARP code and checkpoint | A metric PLY in the source camera frame, source hashes, model versions, stage timings and memory |
| renderer_adapter | Generated PLY/metadata, camera transforms, source PNG for self-render error only | Source self-render and six target RGB/Z/alpha views, GPU probe results, file hashes and resources |
| features | Exact source RGB and same-condition predicted source view, target predictions and two known cameras | Reprojection, predicted depth consistency, local features and missing/validity flags |
| risk_model | Allowed deployment files and L0 thresholds | Rule score, reason, and high/low/unknown states per target plus a record of the actual files read |
| labels | Reference RGB/geometry and corresponding prediction | Pixelwise RGB error, error labels, depth residuals where verified, supporting masks |
| evaluation | Separate reference, prediction and scoring records | Counted metrics, fixed comparison board and source hashes |
| archive | Frozen producer bundles, pinned shared assets, local store and external receipts | Verification or quarantine receipts, immutable received copies, rebuildable SQLite and archive files |

Projection uses metric OpenCV camera coordinates, x right, y down, z forward. Camera JSON stores integer pixel-centre K. Its equivalent boundary-origin K adds 0.5 to cx and cy. Source lookups use floor on the projected boundary-origin coordinates. At scale s without a crop, the integer-centre principal point becomes s(c+0.5)−0.5. gsplat v1.5.3 adds half a pixel at sampling, so its adapter receives boundary-origin K. The PLY frame is the source camera's frame. Target rendering therefore uses inverse(target c2w) multiplied by source c2w. These conventions also apply at the input pose.

All prediction arrays use finite display-sRGB values in [0,1] when valid, FP32 RGB in H×W×3 order, metric FP32 expected positive camera-Z in H×W, and unitless FP32 accumulated opacity in H×W. Zero contribution has undefined depth. The renderer normalizes accumulated depth exactly once through RGB+ED after its calibration check. Blender's world Position and source ray casts are evaluation inputs. The Windows worker handoff contains only the exact input PNG and camera file from the reference producer. Its code allowlist omits the reference recipe.

RGB error uses the per-pixel mean absolute channel difference. The initial positive threshold is strictly greater than 0.05. RGB evaluation includes valid background and low-contribution pixels. V counts reference-valid RGB pixels, E counts reference- and prediction-valid RGB pixels. A low-risk acceptance counts against V. Precision, recall, unknowns and failure rates retain their own denominators.
