"""CPU-only L0 and 1-alpha inference. Run with python -m evidence_lens.risk_model."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .features import ALPHA_MIN, CONFLICT_MAX, extract_features, feature_schema
from .scoring_io import (
    TARGETS, PredictionBundle, ReadAudit, file_hash, load_cameras, load_source_png,
    new_output, rgb_valid, write_json,
)


REASONS = {
    0: "supported",
    1: "invalid_target_alpha",
    2: "low_target_contribution",
    3: "unvalidated_target_depth",
    4: "invalid_target_depth",
    5: "outside_input_view",
    6: "invalid_input_alpha",
    7: "low_input_contribution",
    8: "invalid_input_depth",
    9: "pred_depth_behind",
}
HIGH = (2, 5, 7, 9)


def score_features(features, depth_verified: bool):
    """Use the first applicable rule per pixel; NaN denotes an unavailable score."""
    alpha, source_alpha = features["target_alpha"], features["source_alpha"]
    frame, behind = features["frame_relation"], features["c_behind"]
    conditions = [
        ~np.isfinite(alpha), alpha < ALPHA_MIN, frame == 0, (frame == 2) | (frame == 3),
        ~np.isfinite(source_alpha), source_alpha < ALPHA_MIN, ~np.isfinite(behind),
        behind > CONFLICT_MAX,
    ]
    reason = np.select(conditions, [1, 2, 4 if depth_verified else 3, 5, 6, 7, 8, 9], default=0).astype(np.uint8)
    risk = np.full(reason.shape, np.nan, np.float32)
    risk[reason == 0] = 0
    risk[np.isin(reason, HIGH)] = 1
    alpha_score = np.float32(1) - alpha
    return {
        "l0": risk,
        "l0_valid": np.isfinite(risk),
        "l0_reason": reason,
        "alpha_score": alpha_score,
        "alpha_score_valid": np.isfinite(alpha_score),
    }


def rgb_preview(rgb):
    valid = rgb_valid(rgb)
    output = np.rint(255 * np.where(valid[..., None], rgb, 0)).clip(0, 255).astype(np.uint8)
    output[~valid] = [220, 35, 171]
    return Image.fromarray(output)


def _source_comparison(out, input_rgb, source):
    reference, prediction = rgb_preview(input_rgb), rgb_preview(source["rgb"])
    w, h = reference.size
    image = Image.new("RGB", (2 * w, h + 22), "white")
    image.paste(reference, (0, 22))
    image.paste(prediction, (w, 22))
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), "Input PNG", fill="black")
    draw.text((w + 4, 4), "Source self-render", fill="black")
    image.save(out / "source_comparison.png")
    valid = rgb_valid(source["rgb"])
    error = np.full(valid.shape, np.nan, np.float32)
    error[valid] = np.abs(source["rgb"][valid] - input_rgb[valid]).mean(axis=-1)
    np.savez_compressed(out / "source_comparison.npz", rgb_mae=error, rgb_evaluable=valid)
    return {
        "valid_pixels": int(valid.sum()), "total_pixels": int(valid.size),
        "mae": float(error[valid].mean(dtype=np.float64)) if valid.any() else None,
    }


def run_score(input_path: Path, camera_path: Path, predictions: Path, output: Path, run_id=None):
    audit = ReadAudit()
    cameras = load_cameras(camera_path, audit)
    input_rgb = load_source_png(input_path, cameras["source"], audit)
    bundle = PredictionBundle(predictions, audit)
    source = bundle.read("source", cameras["source"])
    with new_output(output) as out:
        write_json(out / "feature_schema.json", feature_schema())
        receipt = {
            "schema": "el-score-1", "run_id": run_id, "stage": "P2",
            "methods": ["L0", "1-alpha"], "l0_reason_codes": REASONS,
            "l0_high_codes": list(HIGH), "l0_low_code": 0, "missing_scores": "NaN",
            "1-alpha_operating_point": "high if score > 0.5; P2 diagnostic threshold",
            "producer_run_id": bundle.meta["producer_run_id"],
            "depth_verified": bundle.depth_verified,
            "feature_schema_sha256": file_hash(out / "feature_schema.json"),
            "source_comparison": _source_comparison(out, input_rgb, source) if source is not None
                                 else {"status": "render_failed", "reason": bundle.failed["source"]},
            "views": {},
        }
        for view in TARGETS:
            target = bundle.read(view, cameras[view])
            if target is None:
                receipt["views"][view] = {"status": "render_failed", "reason": bundle.failed[view]}
                continue
            features = extract_features(input_rgb, source, target, cameras["source"], cameras[view], bundle.depth_verified)
            scores = score_features(features, bundle.depth_verified)
            path = out / f"{view}.npz"
            np.savez_compressed(path, **features, **scores)
            receipt["views"][view] = {"status": "scored", "score_sha256": file_hash(path)}
        receipt["inputs"] = audit.files
        write_json(out / "score.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="the condition's exact source PNG")
    parser.add_argument("--cameras", required=True, type=Path, help="el-camera-1 JSON")
    parser.add_argument("--predictions", required=True, type=Path, help="directory containing prediction.json and per-view NPZ")
    parser.add_argument("--output", required=True, type=Path, help="new score directory")
    parser.add_argument("--run-id", help="ID assigned by the recording stage")
    args = parser.parse_args(argv)
    run_score(args.input, args.cameras, args.predictions, args.output, args.run_id)


if __name__ == "__main__":
    main()
