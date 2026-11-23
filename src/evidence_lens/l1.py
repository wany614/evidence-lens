"""Load and apply an exported Evidence Lens L1 model with NumPy and Pillow only."""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from .features import FEATURES, FRAME_STATES, extract_features, feature_schema
from .risk_model import score_features
from .scoring_io import TARGETS, PredictionBundle, ReadAudit, file_hash, load_cameras, load_source_png, new_output, rgb_valid, write_json


MODELS = ("L1-full", "L1-geometry")
KEYS = {"L1-full": "l1_full", "L1-geometry": "l1_geometry", "constant": "constant", "1-alpha": "alpha_score", "L0": "l0"}
COLOR_FEATURES = {"local_rgb_difference", "source_rgb_difference", "source_render_residual"}
NUMERIC = list(FEATURES)
ALL_COLUMNS = feature_schema()["columns"]
FEATURE_SCHEMA = json.loads(json.dumps(feature_schema()))


def schema_digest():
    data = json.dumps(FEATURE_SCHEMA, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def columns(method):
    if method not in MODELS:
        raise ValueError(f"unknown L1 method {method}")
    nums = [k for k in NUMERIC if method == "L1-full" or k not in COLOR_FEATURES]
    return nums, [*nums, *("frame_" + s for s in FRAME_STATES), *("missing_" + k for k in nums)]


def feature_rows(features, indices):
    return np.column_stack([features[key].ravel()[indices] for key in ALL_COLUMNS]).astype(np.float32, copy=False)


def transform(raw, params):
    """Impute and standardize only numeric columns. Binary indicators keep their original values."""
    x = raw[:, [ALL_COLUMNS.index(k) for k in params["columns"]]].astype(np.float64)
    n = len(params["medians"])
    z = x[:, :n]
    z[:] = np.where(np.isfinite(z), z, params["medians"])
    z -= params["means"]
    z /= params["scales"]
    return x


def predict_rows(raw, params):
    x = transform(raw, params)
    logit = x @ np.asarray(params["coefficients"], dtype=np.float64) + params["intercept"]
    return np.exp(-np.logaddexp(0.0, -logit)).astype(np.float32)


def _validate_model(params, method):
    nums, cols = columns(method)
    if (params.get("schema") != "el-l1-coefficients-1" or params.get("method") != method
            or params.get("columns") != cols or params.get("numeric_columns") != nums
            or params.get("feature_schema_digest") != schema_digest()):
        raise ValueError(f"incompatible schema or column order for {method}")
    for key, length in (("medians", len(nums)), ("means", len(nums)), ("scales", len(nums)), ("coefficients", len(cols))):
        arr = np.asarray(params[key], dtype=np.float64)
        if arr.shape != (length,) or not np.isfinite(arr).all() or (key == "scales" and np.any(arr <= 0)):
            raise ValueError(f"invalid {method} {key}")
    if (not isinstance(params.get("intercept"), (float, int)) or not np.isfinite(params["intercept"])
            or params.get("classes") != [0, 1] or params.get("C") not in (0.1, 1.0, 10.0)):
        raise ValueError(f"invalid {method} intercept or class order")
    return params


def load_package(directory, *, require_frozen=True, expected_index_sha256=None):
    directory = Path(directory)
    audit = ReadAudit()
    main = json.loads(audit.read("model_index", directory / "model.json"))
    if expected_index_sha256 is not None and audit.sha("model_index") != expected_index_sha256:
        raise ValueError("model.json differs from the pinned model index hash")
    if main.get("schema") != "el-l1-package-1" or main.get("feature_schema_digest") != schema_digest():
        raise ValueError("incompatible L1 package or feature schema")
    status = main.get("status")
    if status not in ("fit", "frozen") or (require_frozen and status != "frozen"):
        raise ValueError("deployment requires a development-frozen L1 package")
    if (not isinstance(main.get("fit_run_id"), str) or not main["fit_run_id"].strip()
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(main.get(k))) for k in ("split_manifest_sha256", "training_manifest_sha256"))
            or (status == "fit" and main.get("model_id") != main["fit_run_id"])):
        raise ValueError("invalid L1 fit identity or manifest pins")
    names = ["feature_schema.json", *(name + ".json" for name in MODELS)]
    if status == "frozen":
        names.append("thresholds.json")
    if set(main["files"]) != set(names):
        raise ValueError("model package file list is incomplete or unsupported")
    docs = {}
    for name in names:
        data = audit.read(name, directory / name)
        if audit.sha(name) != main["files"][name]:
            raise ValueError(f"model digest mismatch: {name}")
        docs[name] = json.loads(data)
    if docs["feature_schema.json"] != FEATURE_SCHEMA:
        raise ValueError("model feature definitions do not match this extractor")
    models = {name: _validate_model(docs[name + ".json"], name) for name in MODELS}
    constant = main["constant_risk"]
    if not isinstance(constant, (float, int)) or not np.isfinite(constant) or not 0 <= constant <= 1:
        raise ValueError("invalid constant training baseline")
    thresholds = {}
    if status == "frozen":
        frozen = docs["thresholds.json"]
        if (frozen.get("schema") != "el-l1-thresholds-1" or frozen.get("fit_run_id") != main["fit_run_id"]
                or not isinstance(main.get("threshold_run_id"), str) or not main["threshold_run_id"].strip()
                or main.get("model_id") != main["threshold_run_id"] or frozen.get("threshold_run_id") != main["threshold_run_id"]
                or frozen.get("split_manifest_sha256") != main["split_manifest_sha256"]
                or set(frozen["thresholds"]) != {*MODELS, "constant", "1-alpha"}):
            raise ValueError("thresholds disagree with the fit")
        thresholds = frozen["thresholds"]
        if any(not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 1.000001 for v in thresholds.values()):
            raise ValueError("invalid operating threshold")
    return {"index": main, "models": models, "thresholds": thresholds, "file_hashes": audit.files}


def risk_state(score, tau):
    state = np.full(score.shape, 255, np.uint8)
    ok = np.isfinite(score)
    # Compare in float64 so a threshold between adjacent FP32 scores stays between them.
    state[ok] = (score[ok].astype(np.float64) >= tau).astype(np.uint8)
    return state


def score_view(package, features, target, depth_verified):
    baseline = score_features(features, depth_verified)
    shape = features["scoreable"].shape
    result = {**baseline, "constant": np.where(rgb_valid(target["rgb"]), np.float32(package["index"]["constant_risk"]), np.float32(np.nan))}
    indices = np.flatnonzero(features["scoreable"])
    for name in MODELS:
        result[KEYS[name]] = np.full(shape, np.nan, np.float32)
    for start in range(0, indices.size, 65536):
        chosen = indices[start:start + 65536]
        raw = feature_rows(features, chosen)
        for name in MODELS:
            result[KEYS[name]].ravel()[chosen] = predict_rows(raw, package["models"][name])
    for method, key in KEYS.items():
        result[key + "_valid"] = np.isfinite(result[key])
        result[key + "_state"] = risk_state(result[key], 0.5 if method == "L0" else package["thresholds"][method])
    return result


def _unknown_view(out, view, status, reason, camera):
    entry = {"status": status, "reason": reason}
    if camera is not None:
        maps = {}
        for key in KEYS.values():
            maps[key] = np.full(camera.shape, np.nan, np.float32)
            maps[key + "_valid"] = np.zeros(camera.shape, bool)
            maps[key + "_state"] = np.full(camera.shape, 255, np.uint8)
        file = out / f"{view}.npz"
        np.savez_compressed(file, **maps)
        entry.update(sha256=file_hash(file), l1_scoreable=0)
    return entry


def _unsupported(out, receipt, reason, cameras=None):
    receipt.update(status="unsupported", reason=reason)
    for view in TARGETS:
        receipt["views"][view] = _unknown_view(out, view, "unknown", reason, cameras[view] if cameras else None)
    write_json(out / "l1_score.json", receipt)
    return receipt


def run(input_path, camera_path, prediction_dir, model_dir, output, run_id=None, model_sha256=None):
    package = load_package(model_dir, expected_index_sha256=model_sha256)
    audit = ReadAudit()
    # Establish byte identity even when the matching producer declares an unsupported camera or encoding.
    # ReadAudit also detects a change between this preflight and the subsequent decode.
    metadata = json.loads(audit.read("prediction_metadata", prediction_dir / "prediction.json"))
    for role, key, path in (("input_png", "input_sha256", input_path), ("cameras", "camera_sha256", camera_path)):
        audit.read(role, path)
        if metadata.get(key) != audit.sha(role):
            raise ValueError(f"prediction.json: incompatible {key} (declared and supplied bytes differ)")
    receipt = {"schema": "el-l1-score-1", "run_id": run_id, "model_id": package["index"]["model_id"],
               "model_files": package["file_hashes"], "inputs": audit.files, "methods": KEYS,
               "state_codes": {"0": "low", "1": "high", "255": "unknown"},
               "thresholds": package["thresholds"], "views": {}}
    with new_output(output) as out:
        try:
            cameras = load_cameras(camera_path, audit)
        except (ValueError, KeyError, TypeError) as e:
            return _unsupported(out, receipt, f"incompatible camera bundle: {e}")
        try:
            image = load_source_png(input_path, cameras["source"], audit)
        except ValueError as e:
            return _unsupported(out, receipt, f"incompatible source PNG: {e}", cameras)
        try:
            predictions = PredictionBundle(prediction_dir, audit)
        except ValueError as e:
            if str(e) in {f"prediction.json: incompatible {k}" for k in ("schema", "depth_semantics", "alpha_semantics", "rgb_encoding")}:
                return _unsupported(out, receipt, str(e), cameras)
            raise
        source = predictions.read("source", cameras["source"])
        receipt.update(producer_run_id=predictions.meta["producer_run_id"], depth_verified=predictions.depth_verified,
                       depth_validation_receipt_sha256=predictions.depth_receipt, source_render_status="failed" if source is None else "completed",
                       source_render_failure=predictions.failed.get("source"),
                       status="scored" if predictions.depth_verified else "unsupported_l1_depth",
                       reason=None if predictions.depth_verified else "provider depth validation did not pass")
        for view in TARGETS:
            t = time.perf_counter()
            target = predictions.read(view, cameras[view])
            if target is None:
                receipt["views"][view] = _unknown_view(out, view, "render_failed", predictions.failed[view], cameras[view])
                continue
            start = time.perf_counter()
            features = extract_features(image, source, target, cameras["source"], cameras[view], predictions.depth_verified)
            features_done = time.perf_counter()
            scores = score_view(package, features, target, predictions.depth_verified)
            score_done = time.perf_counter()
            file = out / f"{view}.npz"
            np.savez_compressed(file, **scores)
            receipt["views"][view] = {"status": receipt["status"], "sha256": file_hash(file),
                                       "l1_scoreable": int(features["scoreable"].sum()), "target_read_seconds": start - t,
                                       "features_seconds": features_done - start, "scoring_seconds": score_done - features_done,
                                       "output_seconds": time.perf_counter() - score_done}
        write_json(out / "l1_score.json", receipt)
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("input", "cameras", "predictions", "model", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--run-id", default=os.environ.get("EL_RUN_ID"))
    p.add_argument("--model-sha256", help="pin exact model.json bytes to an independently supplied release hash")
    a = p.parse_args(argv)
    run(a.input, a.cameras, a.predictions, a.model, a.output, a.run_id, a.model_sha256)


if __name__ == "__main__":
    main()
