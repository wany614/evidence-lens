# D1 photographic evaluation

D1 scores the original v0.3.0-rc.1 CPU release on DTU’s rectified photographs. The registered matrix uses set1 for integration and set6 for external confirmation. Each reconstruction has six 256 × 256 targets. The default release method is MS-L1. The release thresholds, feature preprocessing and four models are fixed. The protocol and exact source/target camera identities live in [the external study registration](../../protocol/dtu-external-v1.json).

| Method | Scorer sources |
| --- | --- |
| B-L1-transfer | s00 |
| MS-L1, MS-frozen, MS-ft | All sources of the selected M2 or M4 reconstruction |

The release scorer receives a CASE directory with original source photographs, registered camera calibration, source self-renders and target predictions. The CPU API and its four model files are verified against the distributed release index. Independent photographic references enter only the subsequent evaluation. The 1024-step diagnostic on set1 compares source and target reconstruction errors without additional risk scoring.

## Recorded commands

Run the controller from the project checkout. Use a research Python environment with NumPy, Pillow, PyArrow, SciPy, scikit-learn and the recorder. `CPU_PYTHON` is the absolute path to the frozen release's CPU interpreter with PyTorch and safetensors. Supply an empty local state directory on first execution and keep it for idempotent reruns. Every derived stage records its explicit input digests, source snapshot, parents and elapsed time. `--retry-of RID` selects one recorded failed attempt for its single authorized retry.

```sh
python scripts/dtu_score_record.py restore --project "$PWD" --store "$STORE" --state "$STATE"
python scripts/dtu_score_record.py score --project "$PWD" --store "$STORE" --state "$STATE" \
  --cpu-python "$CPU_PYTHON" --role debug --object-id set1 --condition M2 \
  --assets "$ASSET_PIN" --case "$CASE_PIN" --case-member payload/case/case.json \
  --case-sha256 "$CASE_SHA" --authority "$CAMERA_PLAN_PIN" \
  --authority-member payload/camera-plan.json --authority-sha256 "$CAMERA_PLAN_SHA"
python scripts/dtu_eval_record.py evaluate --project "$PWD" --store "$STORE" --state "$STATE" \
  --request "$EVAL_REQUEST" --request-sha256 "$EVAL_REQUEST_SHA"
python scripts/dtu_eval_record.py replay --project "$PWD" --store "$STORE" --state "$STATE" \
  --request "$REPLAY_REQUEST" --request-sha256 "$REPLAY_REQUEST_SHA"
python scripts/dtu_eval_record.py diagnostic --project "$PWD" --store "$STORE" --state "$STATE" \
  --request "$BUDGET_REQUEST" --request-sha256 "$BUDGET_REQUEST_SHA"
python scripts/dtu_eval_record.py report --project "$PWD" --store "$STORE" --state "$STATE" \
  --request "$REPORT_REQUEST" --request-sha256 "$REPORT_REQUEST_SHA"
```

CLI pins are `RUN_ID=MANIFEST_SHA256`. Request member pins contain `run_id`, `manifest_sha256`, `member` and `sha256`. The controller looks up those exact members in `STORE/runs` or the explicit read-only `--shelf`. It verifies the immutable seal and the content of every consumed member.

- `el-dtu-eval-request-1` specifies `role`, `object_id`, `condition`, `arm`, `status`, `authority`, `references`, `case`, `case_provenance`, `readback`, `release` and `scores`. A terminal failure carries its original `failure` and `reason`. A valid reference still establishes V when a render fails. A failed scorer leaves the evaluable image in E and every pixel in U. A missing reference has unknown V.
- `el-dtu-evaluation-replay-request-1` specifies the same role, object, condition, arm and authority, plus the original `evaluation`. Independent RGB arithmetic, bounded-neighbour comparison and scikit-learn AP verify every stored supervision pixel, confusion count and defined two-class AP.
- `el-dtu-source-budget-request-1` uses the independently registered diagnostic `authority`, set1 role/condition, the original 256-step `evaluation`, factual `references`, and `case`, `case_provenance`, `readback` from the 1024-step endpoint. Initial parameters, topology, source photographs and calibrations must match.
- `el-dtu-report-request-1` includes the explicit `objects`, four 256-step `cells` with evaluation/replay pins, `authorities` including the complete external reference gate, and `declared_stages` with `stage`, `original` and optional object/condition/arm. Registered reconstruction-only diagnostics use `debug_1024=true` and one `diagnostic` pin per condition. Public tables and private panels have separate directories.

Set6 scoring requires the sealed source opening. The complete photographic evaluation requires the later reference gate that identifies both terminal scoring conditions. The reference conversion, source opening, case, release and scores must match the request. The original DTU photographs, prediction arrays, risk maps and case panels are for private research inspection under the registered dataset terms.

## Reading the output

The primary event is mean absolute RGB channel error above 0.05 on display sRGB. V is the photographic valid-reference domain. E is its subset with valid prediction RGB. S is the finite score domain within E for each method. Low risk uses `risk < tau` on S. High risk includes ties at the threshold. U is `E − S`. The counts preserve `Y = LY + HY + UY` and `E = L + H + U`. Main coverage is `L/V`, low-group error is `LY/L`, unknown is `U/E`, error incidence is `Y/E` and comparison support is the common intersection of four finite score domains within E. Each table keeps the numerator and denominator. Empty denominators are reported as missing, including whole absent object masks.

Whole-score ties are grouped for AP and risk–coverage. Common and method-specific AP require both label classes in a view. Object AP averages the defined views, and cohort AP averages the defined objects. The auxiliary 3 × 3 metric takes the minimum colour error over valid in-domain neighbours including the centre. It uses the same centre prediction, with no image alignment. Descriptive Brier and ten fixed equal-width bins use finite scores. Expected camera depth, alpha and support give engineering diagnostics. An independently validated image-space segmentation or dense surface depth is unavailable for this protocol.
